"""Data -> Review (every frame checked one by one), the reviewed-only export and the Train job preparation."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sortops.anomaly_train import AnomalyTrainer
from sortops.pipeline.annotation import InspectionStore, validate_review
from sortops.server import create_app

BOX = (0.4375, 0.4167, 0.15625, 0.2083)


def _fill(store: InspectionStore, run: str, cycle: int, start: int, n: int = 3, camera: str = "wrist") -> None:
    frame = np.full((480, 640, 3), 120, np.uint8)
    frame[200:300, 280:380] = (220, 60, 40)
    for i in range(n):
        store.save_shot(run, cycle, start + i, camera, "present", frame, frame[200:300, 280:380], BOX,
                        {"usable": i > 0, "sharpness": 50.0 + i}, 0.1 * i)


def _store(tmp_path: Path) -> InspectionStore:
    store = InspectionStore(tmp_path / "runs")
    _fill(store, "r1", 1, 1)
    _fill(store, "r1", 2, 10)
    store.auto_annotation("r1", 1, "good", "operator")
    store.save_annotation("r1", 2, {"verdict": "bad", "cameras": {"wrist": {
        "shot": 11, "piece": [0.4, 0.4, 0.2, 0.25], "defects": [{"type": "rect", "box": [0.45, 0.45, 0.05, 0.05]}]}}})
    return store


def test_queue_proposes_every_frame_and_keeps_reviews(tmp_path: Path) -> None:
    store = _store(tmp_path)
    queue = store.review_queue("r1")
    assert queue["total"] == 6 and queue["reviewed"] == 0
    by_shot = {i["shot"]: i for i in queue["items"]}
    # unusable frames are proposed excluded; the cycle's reference frame carries its drawn defects
    assert by_shot[1]["excluded"] and not by_shot[2]["excluded"] and by_shot[2]["piece"] == list(BOX)
    assert by_shot[11]["defects"] and by_shot[11]["piece"] == [0.4, 0.4, 0.2, 0.25] and by_shot[12]["defects"] == []
    store.save_review("r1", 2, 12, {"piece": [0.4, 0.4, 0.2, 0.2], "defects": [{"type": "rect", "box": [0.5, 0.5, 0.04, 0.04]}]})
    store.save_review("r1", 1, 1, {"piece": None, "excluded": True})
    queue = store.review_queue("r1")
    assert queue["reviewed"] == 2 and queue["kept"] == 1
    assert {i["shot"]: i for i in queue["items"]}[12]["defects"][0]["box"] == [0.5, 0.5, 0.04, 0.04]
    assert store.runs()[0]["reviewed"] == 2 and store.runs()[0]["kept_by_camera"] == {"wrist": 1}
    with pytest.raises(KeyError):
        store.save_review("r1", 2, 999, {"piece": list(BOX)})


@pytest.mark.parametrize("bad", [{"piece": None}, {"piece": [0, 0, 0, 1]}, {"piece": list(BOX), "defects": [{"type": "x"}]}])
def test_review_validation(bad: dict) -> None:
    with pytest.raises(ValueError):
        validate_review(bad)


def test_reviewed_only_export_takes_reviewed_frames_with_their_own_boxes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for shot in (2, 3):
        store.save_review("r1", 1, shot, {"piece": [0.42, 0.40, 0.18, 0.24]})
    store.save_review("r1", 2, 11, {"piece": [0.4, 0.4, 0.2, 0.25], "defects": [{"type": "rect", "box": [0.45, 0.45, 0.05, 0.05]}]})
    store.save_review("r1", 2, 12, {"piece": [0.4, 0.4, 0.2, 0.25]})  # defect brick, defect not visible on this frame
    dest = tmp_path / "x.zip"
    counts = store.export_zip(["r1"], dest, 64, reviewed_only=True)
    assert counts == {"train_good": 2, "test_good": 0, "test_defect": 1, "test_defect_nomask": 1, "skipped": 2}
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
        assert {"r1/wrist/train/good/c0001_s00002.png", "r1/wrist/train/good/c0001_s00003.png",
                "r1/wrist/test/defect/c0002_s00011.png", "r1/wrist/ground_truth/defect/c0002_s00011.png",
                "r1/wrist/test/defect_nomask/c0002_s00012.png"} <= names
        assert "r1/wrist/train/good/c0001_s00001.png" not in names  # not reviewed


def test_train_prepare_merges_runs_and_refuses_bad_requests(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _fill(store, "r2", 1, 1, n=6)
    store.auto_annotation("r2", 1, "good", "operator")
    for shot in range(1, 7):
        store.save_review("r2", 1, shot, {"piece": list(BOX)})
    trainer = AnomalyTrainer(store, tmp_path / "work", tmp_path / "models", "/nonexistent/python")
    root = trainer.prepare(["r1", "r2"], "wrist", tmp_path / "work" / "j")
    assert len(list((root / "train/good").glob("r2_*.png"))) == 5 and len(list((root / "test/good").glob("*.png"))) == 1
    with pytest.raises(ValueError):
        trainer.start(["r2"], "wrist", "nope")
    with pytest.raises(KeyError):
        trainer.start(["zz"], "wrist", "patchcore")
    with pytest.raises(ValueError):
        trainer.start(["r2"], "wrist", "patchcore", "../evil")
    with pytest.raises(RuntimeError):
        trainer.start(["r2"], "wrist", "patchcore")  # Anomalib python missing


def test_train_job_runs_the_worker_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in worker (same CLI and outputs as anomaly_worker.py) proves the job plumbing end to end."""
    import sys

    from sortops import anomaly_train

    store = InspectionStore(tmp_path / "runs")
    _fill(store, "r2", 1, 1, n=6)
    store.auto_annotation("r2", 1, "good", "operator")
    for shot in range(1, 7):
        store.save_review("r2", 1, shot, {"piece": list(BOX)})
    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import argparse, json, pathlib\n"
        "p = argparse.ArgumentParser(); [p.add_argument(a) for a in ('--root', '--model', '--out', '--work')]\n"
        "a = p.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True)\n"
        "print('PROGRESS 50 Training', flush=True)\n"
        "(out / 'model.xml').write_text('<xml/>'); (out / 'model.bin').write_bytes(b'0')\n"
        "(out / 'metrics.json').write_text(json.dumps({'metrics': {'image_AUROC': 0.9}}))\n"
    )
    monkeypatch.setattr(anomaly_train, "WORKER", fake)
    trainer = AnomalyTrainer(store, tmp_path / "work", tmp_path / "models", sys.executable)
    job = trainer.start(["r2"], "wrist", "patchcore", "m1")
    deadline = time.monotonic() + 30
    while (trainer.status() or {}).get("status") in ("preparing", "training") and time.monotonic() < deadline:
        time.sleep(0.05)
    status = trainer.status()
    assert status and status["status"] == "done", status
    assert status["metrics"] == {"image_AUROC": 0.9} and (tmp_path / "models/wrist/m1/model.xml").exists()
    assert job["out"].endswith("wrist/m1") and not list((tmp_path / "work").glob("*-wrist"))  # work dir cleaned
    with pytest.raises(ValueError):
        trainer.start(["r2"], "wrist", "patchcore", "m1")  # never overwrites a model


def test_review_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SORTOPS_DATA", str(tmp_path))
    store = InspectionStore(tmp_path / "runs")
    _fill(store, "r9", 1, 1)
    store.auto_annotation("r9", 1, "good", "operator")
    with TestClient(create_app()) as client:
        queue = client.get("/api/inspection/runs/r9/review").json()
        assert queue["total"] == 3
        assert client.post("/api/inspection/runs/r9/cycles/1/shots/2/review", json={"piece": list(BOX)}).status_code == 200
        assert client.post("/api/inspection/runs/r9/cycles/1/shots/2/review", json={"piece": None}).status_code == 422
        assert client.post("/api/inspection/runs/r9/cycles/1/shots/99/review", json={"piece": list(BOX)}).status_code == 404
        assert client.get("/api/inspection/runs/zz/review").status_code == 404
        assert client.get("/api/inspection/runs/r9/review").json()["reviewed"] == 1
        assert client.get("/api/inspection/runs/r9/dataset.zip?only=all").status_code == 200
        assert client.get("/api/inspection/runs/r9/dataset.zip?only=x").status_code == 422
        assert client.get("/api/anomaly/train").json() == {"job": None}
        assert client.post("/api/anomaly/train", json={"runs": ["r9"], "camera": "wrist", "model": "nope"}).status_code == 422
        assert client.post("/api/anomaly/train/cancel").status_code == 409
