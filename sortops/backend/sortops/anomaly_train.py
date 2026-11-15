"""Data -> Train: an Anomalib model trained on the REVIEWED frames of chosen runs, for one inspection camera.

The reviewed export (``InspectionStore.export_zip(reviewed_only=True)``) is unpacked into one Folder dataset (runs merged,
file names prefixed by the run), then ``anomaly_worker.py`` runs in Anomalib's venv as a subprocess. The OpenVINO model
goes to ``<anomaly_model_roots[0]>/<camera>/<name>/`` with its ``metrics.json``: SortOps lists it as a new anomaly
model; nothing is activated and no existing model is touched. One training at a time.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sortops.pipeline.annotation import InspectionStore

WORKER = Path(__file__).with_name("anomaly_worker.py")
MODELS = ("patchcore", "padim", "efficient_ad", "fre", "dinomaly")
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass
class TrainJob:
    id: str
    runs: list[str]
    camera: str
    model: str
    name: str
    out: str
    status: str = "preparing"  # preparing | training | done | failed | canceled
    progress: int = 0
    message: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    images: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    log_tail: list[str] = field(default_factory=list)


class AnomalyTrainer:
    def __init__(self, store: InspectionStore, work_dir: Path, models_root: Path, python: str) -> None:
        self.store = store
        self.work_dir = work_dir
        self.models_root = models_root
        self.python = python
        self._lock = threading.Lock()
        self._job: TrainJob | None = None
        self._process: subprocess.Popen[str] | None = None

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._job is None else asdict(self._job)

    def start(self, runs: list[str], camera: str, model: str, name: str | None = None) -> dict[str, Any]:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if not runs:
            raise ValueError("choose at least one run")
        if not re.fullmatch(r"[a-z0-9_-]+", camera):
            raise ValueError("bad camera")
        known = {item["run"] for item in self.store.runs()}
        missing = [r for r in runs if r not in known]
        if missing:
            raise KeyError(f"no inspection data for {missing}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = name or f"{model}-{stamp}"
        if not NAME_OK.match(name):
            raise ValueError("name: letters, digits, '.', '_' or '-' (64 max)")
        out = self.models_root / camera / name
        if out.exists():
            raise ValueError(f"a model named {name} already exists for {camera}")
        if not Path(self.python).exists():
            raise RuntimeError(f"Anomalib python not found: {self.python} (hardware.anomalib_python)")
        with self._lock:
            if self._job is not None and self._job.status in ("preparing", "training"):
                raise RuntimeError("a training is already running")
            self._job = TrainJob(id=stamp, runs=list(runs), camera=camera, model=model, name=name, out=str(out))
            job = self._job
        threading.Thread(target=self._run, args=(job,), name="anomaly-train", daemon=True).start()
        return asdict(job)

    def cancel(self) -> None:
        with self._lock:
            process, job = self._process, self._job
            if job is None or job.status not in ("preparing", "training"):
                raise RuntimeError("no training is running")
            job.status = "canceled"
        if process is not None:
            process.terminate()

    def _update(self, job: TrainJob, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(job, key, value)

    def _run(self, job: TrainJob) -> None:
        work = self.work_dir / f"{job.id}-{job.camera}"
        try:
            root = self.prepare(job.runs, job.camera, work)
            images = {d: len(list((root / d).glob("*.png"))) for d in ("train/good", "test/good", "test/defect")}
            self._update(job, images=images)
            if images["train/good"] < 5:
                raise RuntimeError(f"only {images['train/good']} reviewed good frames for {job.camera}: review at least 5")
            if job.status == "canceled":
                return
            self._update(job, status="training", progress=5, message="Starting Anomalib")
            command = [self.python, str(WORKER), "--root", str(root), "--model", job.model, "--out", job.out,
                       "--work", str(work / "anomalib")]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            with self._lock:
                self._process = process
            tail: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-40:]
                if line.startswith("PROGRESS "):
                    _, value, *message = line.split(" ", 2)
                    self._update(job, progress=int(value), message=message[0] if message else "", log_tail=tail)
                else:
                    self._update(job, log_tail=tail)
            code = process.wait()
            if job.status == "canceled":
                shutil.rmtree(job.out, ignore_errors=True)
                self._update(job, message="Canceled")
                return
            if code != 0:
                shutil.rmtree(job.out, ignore_errors=True)
                raise RuntimeError(f"Anomalib exited with code {code}: {tail[-1] if tail else ''}")
            report = json.loads((Path(job.out) / "metrics.json").read_text())
            self._update(job, status="done", progress=100, message="Model ready: choose it in Models",
                         metrics=report.get("metrics", {}))
        except Exception as error:  # noqa: BLE001 - reported on the job
            if job.status != "canceled":
                self._update(job, status="failed", message=str(error))
        finally:
            with self._lock:
                self._process = None
            self._update(job, ended=time.time())
            shutil.rmtree(work, ignore_errors=True)

    def prepare(self, runs: list[str], camera: str, work: Path) -> Path:
        """Reviewed export of ``runs`` for ``camera`` unpacked as one Folder dataset; returns its root."""
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        archive = work / "export.zip"
        self.store.export_zip(runs, archive, reviewed_only=True)
        root = work / "dataset"
        with zipfile.ZipFile(archive) as zf:
            for entry in zf.namelist():
                parts = entry.split("/")
                # <run>/<camera>/<split>/<kind>/<name>.png
                if len(parts) != 5 or parts[1] != camera or parts[2] == "full" or not parts[4].endswith(".png"):
                    continue
                run, _, split, kind, name = parts
                if f"{split}/{kind}" not in ("train/good", "test/good", "test/defect", "ground_truth/defect"):
                    continue
                target = root / split / kind / f"{run}_{name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(entry))
        archive.unlink()
        for d in ("train/good", "test/good", "test/defect", "ground_truth/defect"):
            (root / d).mkdir(parents=True, exist_ok=True)
        return root
