"""Inspection data kept per run: readable anomaly maps, operator annotations, masks and the Anomalib export."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from sortops.cell import Cell
from sortops.config import Settings
from sortops.pipeline.annotation import InspectionStore, defect_mask, square_cut, validate_annotation
from sortops.pipeline.heatmap import render, units
from sortops.safety import Mode

TERMS = {"terms": {"mean_median": 1.0, "peak_median": 3.0, "peak_spread": 1.0, "peak_weight": 0.8,
                   "peak_fraction": 0.01, "peak_from_raw": True, "mean_spread": 0.2}, "threshold": 1.0, "spread": 1.0}


def _crop() -> np.ndarray:
    crop = np.full((256, 256, 3), 200, np.uint8)
    crop[60:200, 60:200] = (30, 80, 220)
    return crop


# Anomaly maps --------------------------------------------------------------------------------------------


def test_calibrated_map_marks_only_the_hot_spot_as_a_defect_region() -> None:
    raw = np.full((64, 64), 1.0, np.float32)  # a typical good background everywhere
    raw[40:48, 10:18] = 6.0  # one patch far hotter than the worst patch of a good brick (3 + 1)
    image, info = render(_crop(), raw, None, TERMS)
    assert image.shape == (256, 256, 3)
    assert info.scale == "terms" and info.calibrated
    assert info.max_units == pytest.approx((6 - 1) / (3 + 1 - 1), abs=0.02)  # (hot - good mean) / (worst good patch - good mean)
    assert len(info.regions) == 1
    x, y, w, h = info.regions[0]
    assert 0.12 < x < 0.2 and 0.6 < y < 0.68 and w < 0.2 and h < 0.2
    assert 0.15 < info.peak[0] < 0.3 and 0.6 < info.peak[1] < 0.78


def test_a_clean_brick_has_no_region_and_stays_below_the_contour() -> None:
    raw = np.full((64, 64), 1.0, np.float32)
    raw[30:34, 30:34] = 2.5  # a bit warm, still below the worst good patch
    _, info = render(_crop(), raw, None, TERMS)
    assert info.max_units < 1 and info.regions == []


def test_uncalibrated_maps_fall_back_to_the_export_and_say_so() -> None:
    heat = np.zeros((32, 32), np.float32)
    heat[5:9, 5:9] = 0.9
    u, scale, calibrated = units(None, heat, None)
    assert scale == "export" and not calibrated and u.max() == pytest.approx(1.8)
    _, info = render(_crop(), None, heat, None)
    assert not info.calibrated and len(info.regions) == 1


# Annotations and masks -----------------------------------------------------------------------------------


def test_annotation_is_validated_and_clamped() -> None:
    clean = validate_annotation({
        "verdict": "bad",
        "cameras": {"wrist": {"shot": 3, "piece": [0.9, 0.1, 0.5, 0.5],
                              "defects": [{"type": "rect", "box": [0.1, 0.1, 0.2, 0.2]},
                                          {"type": "stroke", "points": [[0.5, 0.5], [1.4, 0.6]], "width": 0.05}]}},
    })
    assert clean["cameras"]["wrist"]["piece"] == [0.9, 0.1, 0.1, 0.5]  # cut at the frame edge
    assert clean["cameras"]["wrist"]["defects"][1]["points"][1] == [1.0, 0.6]
    for bad in ({"verdict": "maybe"}, {"verdict": "bad", "cameras": {"../x": {"shot": 1}}},
                {"verdict": "bad", "cameras": {"wrist": {"defects": [{"type": "circle"}]}}},
                {"verdict": "bad", "cameras": {"wrist": {"piece": [0, 0, 0, 1]}}}):
        with pytest.raises(ValueError):
            validate_annotation(bad)


def test_masks_follow_the_shapes_and_the_square_cut() -> None:
    mask = defect_mask((480, 640), [{"type": "rect", "box": [0.5, 0.5, 0.1, 0.1]},
                                    {"type": "stroke", "points": [[0.1, 0.1], [0.2, 0.1]], "width": 0.02}])
    assert mask[264, 352] == 255 and mask[48, 96] == 255 and mask[400, 100] == 0
    cut = square_cut(mask, [0.5, 0.5, 0.1, 0.1], 128)
    # the 64 x 48 px box grows to a 64 px square: the defect fills its 48 middle rows
    assert cut.shape == (128, 128) and cut[64, 64] == 255 and cut[2, 64] == 0 and 0.7 < (cut > 0).mean() < 0.8


# Store and export ----------------------------------------------------------------------------------------


def _fill(store: InspectionStore, run: str, cycle: int, start: int) -> None:
    frame = np.full((480, 640, 3), 120, np.uint8)
    frame[200:300, 280:380] = (220, 60, 40)
    for i in range(3):
        store.save_shot(run, cycle, start + i, "wrist", "present", frame, frame[200:300, 280:380], (0.4375, 0.4167, 0.15625, 0.2083),
                        {"usable": i > 0, "sharpness": 50.0 + i}, 0.1 * i)


def test_store_keeps_frames_annotations_and_exports_anomalib_layout(tmp_path: Path) -> None:
    store = InspectionStore(tmp_path / "runs")
    _fill(store, "r1", 1, 1)
    _fill(store, "r1", 2, 10)
    _fill(store, "r1", 3, 20)
    assert store.best_shots("r1", 1)["wrist"]["shot"] == 3  # usable and sharpest
    store.auto_annotation("r1", 1, "good", "model")
    store.save_annotation("r1", 2, {"verdict": "bad", "cameras": {"wrist": {
        "shot": 11, "piece": [0.4, 0.4, 0.2, 0.25], "defects": [{"type": "rect", "box": [0.45, 0.45, 0.05, 0.05]}]}}})
    store.save_annotation("r1", 3, {"verdict": "no_piece", "cameras": {}})
    runs = store.runs()
    assert runs[0]["run"] == "r1" and runs[0]["good"] == 1 and runs[0]["bad"] == 1 and runs[0]["no_piece"] == 1
    assert runs[0]["frames"] == 9 and runs[0]["bytes"] > 0
    with pytest.raises(KeyError):
        store.file("r1", 1, "../annotation.json")
    for bad_run in ("../..", "a/b", ""):
        with pytest.raises(KeyError):
            store.cycle_dir(bad_run, 1)

    dest = tmp_path / "out.zip"
    counts = store.export_zip(["r1"], dest, 128)
    assert counts == {"train_good": 1, "test_good": 1, "test_defect": 1, "test_defect_nomask": 0, "skipped": 1}
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
        assert {"r1/wrist/test/good/c0001_s00003.png", "r1/wrist/train/good/c0001_s00002.png",
                "r1/wrist/test/defect/c0002_s00011.png", "r1/wrist/ground_truth/defect/c0002_s00011.png",
                "r1/wrist/full/c0002_s00011.jpg", "annotations.jsonl", "README.txt"} <= names
        mask = cv2.imdecode(np.frombuffer(archive.read("r1/wrist/ground_truth/defect/c0002_s00011.png"), np.uint8), 0)
        image = cv2.imdecode(np.frombuffer(archive.read("r1/wrist/test/defect/c0002_s00011.png"), np.uint8), 1)
        assert mask.shape == (128, 128) and image.shape == (128, 128, 3) and 0 < (mask > 0).mean() < 0.5
        assert len(archive.read("annotations.jsonl").splitlines()) == 2


# The cell ------------------------------------------------------------------------------------------------


def _run_one(cell: Cell, operator_answer: tuple[str, dict] | None) -> None:
    cell.set_mode(Mode.LIVE)
    cell.start_cycles(loop=False)
    deadline = time.monotonic() + 60
    while cell.cycle_running() and time.monotonic() < deadline:
        if operator_answer and cell.awaiting_verdict():
            cell.give_verdict(*operator_answer)
            operator_answer = None
        time.sleep(0.1)
    assert not cell.cycle_running()


def test_a_late_verdict_for_another_brick_is_refused(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "decision": {"verdict_source": "operator"}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=False)
        deadline = time.monotonic() + 60
        while not cell.awaiting_verdict() and time.monotonic() < deadline:
            time.sleep(0.05)
        waiting = cell.snapshot()["inspection"]
        assert cell.awaiting_verdict() and waiting
        stale = {"cameras": {"wrist": {"piece": [0.1, 0.1, 0.2, 0.2], "defects": []}}, "note": "previous brick"}
        for run, cycle in ((waiting["run"], waiting["cycle"] - 1), ("other-run", waiting["cycle"])):
            try:
                cell.give_verdict("good", stale, run, cycle)
            except RuntimeError as error:
                assert "brick waiting" in str(error)
            else:
                raise AssertionError("a verdict for another brick was accepted")
        assert cell.awaiting_verdict()  # still waiting: the stale answer was not consumed
        cell.give_verdict("bad", None, waiting["run"], waiting["cycle"])
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.05)
        record = cell.cycles.cycles()[0]
        assert record.outcome == "bad"
        assert cell.inspection_store.cycle(record.run, record.id)["annotation"].get("note") != "previous brick"
    finally:
        cell.shutdown()


def test_operator_drawing_is_saved_with_the_cycle_and_exported(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "decision": {"verdict_source": "operator"}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        drawing = {"cameras": {"wrist": {"piece": [0.3, 0.3, 0.4, 0.4],
                                         "defects": [{"type": "rect", "box": [0.4, 0.4, 0.05, 0.05]}]},
                               "front": {"piece": [0.3, 0.1, 0.3, 0.4], "defects": []}}, "note": "marker dot"}
        _run_one(cell, ("bad", drawing))
        record = cell.cycles.cycles()[0]
        assert record.outcome == "bad" and record.truth == "bad"
        detail = cell.inspection_store.cycle(record.run, record.id)
        assert detail["annotation"]["source"] == "operator" and detail["annotation"]["note"] == "marker dot"
        assert detail["annotation"]["cameras"]["wrist"]["defects"][0]["box"] == [0.4, 0.4, 0.05, 0.05]
        assert {s["camera"] for s in detail["shots"]} == {"front", "wrist"}
        assert all((cell.inspection_store.cycle_dir(record.run, record.id) / s["frame"]).exists() for s in detail["shots"])
        counts = cell.inspection_store.export_zip([record.run], tmp_path / "x.zip")
        # wrist has the drawn defect (mask); front had nothing drawn: kept apart, without an empty mask
        assert counts["test_defect"] == 1 and counts["test_defect_nomask"] == 1
    finally:
        cell.shutdown()


def test_model_mode_saves_rendered_maps_and_an_automatic_annotation(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        _run_one(cell, None)
        record = cell.cycles.cycles()[0]
        assert record.outcome in {"good", "bad"}
        detail = cell.inspection_store.cycle(record.run, record.id)
        assert detail["annotation"]["source"] == "model" and detail["annotation"]["verdict"] == record.outcome
        heats = [s for s in detail["shots"] if s.get("heat")]
        assert heats and all("max_units" in s["heat_info"] for s in heats)
        assert all(cam["piece"] for cam in detail["annotation"]["cameras"].values())
    finally:
        cell.shutdown()


def test_api_lists_serves_and_corrects_inspection_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from sortops.server import create_app

    monkeypatch.setenv("SORTOPS_DATA", str(tmp_path))
    store = InspectionStore(tmp_path / "runs")
    _fill(store, "r9", 1, 1)
    store.auto_annotation("r9", 1, "good", "model")
    with TestClient(create_app()) as client:
        assert client.get("/api/inspection/runs").json()[0]["run"] == "r9"
        cycle = client.get("/api/inspection/runs/r9/cycles/1").json()
        assert cycle["best"]["wrist"]["shot"] == 3
        image = client.get(f"/api/inspection/runs/r9/cycles/1/files/{cycle['best']['wrist']['frame']}")
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        # path tricks never reach a file outside the cycle's frames (the client may normalize "..": SPA fallback)
        for trick in ("..%2Fannotation.json", ".hidden", "annotation.json%00.jpg"):
            response = client.get(f"/api/inspection/runs/r9/cycles/1/files/{trick}")
            assert response.status_code == 404 or "verdict" not in response.text
        response = client.get("/api/inspection/runs/..%2F..%2Fetc/cycles/1")
        assert response.status_code == 404 or "shots" not in response.text
        fixed = client.post("/api/inspection/runs/r9/cycles/1/annotation", json={
            "verdict": "bad", "cameras": {"wrist": {"shot": 3, "piece": [0.4, 0.4, 0.2, 0.2],
                                                    "defects": [{"type": "rect", "box": [0.45, 0.45, 0.05, 0.05]}]}}})
        assert fixed.status_code == 200 and fixed.json()["source"] == "operator"
        assert client.post("/api/inspection/runs/r9/cycles/1/annotation", json={"verdict": "nope"}).status_code == 422
        archive = client.get("/api/inspection/runs/all/dataset.zip")
        assert archive.status_code == 200 and archive.headers["content-type"] == "application/zip"
        assert client.get("/api/inspection/runs/zz/dataset.zip").status_code == 404
        # a verdict must name its brick (422 without it); 409: nothing is waiting for a verdict
        assert client.post("/api/verdict", json={"verdict": "good"}).status_code == 422
        assert client.post("/api/verdict", json={"verdict": "good", "run": "r9", "cycle": 1}).status_code == 409
    assert not list((tmp_path / "exports").glob("*.zip"))  # streamed then removed
