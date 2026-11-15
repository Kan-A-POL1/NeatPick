"""Evidence for the jury: inference latency per model and device, platform telemetry (power, temperatures,
CPU / iGPU / NPU load), cycle records with their stage timings and energy, and the confusion matrix built from
operator labels. Cycle records are appended to ``<data>/runs/<run>.jsonl``.
"""

from __future__ import annotations

from typing import Any

import csv
import io
import json
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
from loguru import logger


class LatencyBook:
    """Rolling latency samples (ms) per key, e.g. ``pick_present`` or ``anomaly``, tagged with their device."""

    def __init__(self, size: int = 400) -> None:
        self._lock = threading.Lock()
        self._size = size
        self._samples: dict[str, deque[float]] = {}
        self._device: dict[str, str] = {}
        self._count: dict[str, int] = {}

    def record(self, key: str, ms: float, device: str) -> None:
        with self._lock:
            if key not in self._samples or self._device.get(key) != device:
                self._samples[key] = deque(maxlen=self._size)
            self._samples[key].append(ms)
            self._device[key] = device
            self._count[key] = self._count.get(key, 0) + 1

    def summary(self) -> dict[str, dict[str, float | int | str]]:
        with self._lock:
            out: dict[str, dict[str, float | int | str]] = {}
            for key, samples in self._samples.items():
                values = np.asarray(samples, np.float64)
                out[key] = {
                    "device": self._device[key],
                    "count": self._count[key],
                    "last": round(float(values[-1]), 2),
                    "p50": round(float(np.percentile(values, 50)), 2),
                    "p95": round(float(np.percentile(values, 95)), 2),
                    "mean": round(float(values.mean()), 2),
                }
            return out

    def recent(self, key: str, n: int = 60) -> list[float]:
        with self._lock:
            return [round(v, 2) for v in list(self._samples.get(key, []))[-n:]]


@dataclass
class SystemSample:
    t: float
    cpu_pct: float
    ram_pct: float
    package_w: float | None
    package_temp_c: float | None
    npu_pct: float | None
    gpu_pct: float | None


class SystemMonitor:
    """1 Hz platform telemetry from sysfs: RAPL package energy, x86_pkg_temp, NPU busy time, Xe GT idle residency."""

    RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
    NPU_BUSY = Path("/sys/class/accel/accel0/device/npu_busy_time_us")

    def __init__(self, history: int = 300) -> None:
        self.history: deque[SystemSample] = deque(maxlen=history)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._energy_j = 0.0
        self._prev: dict[str, float] = {}
        self._rapl_ok = self._ensure_readable(self.RAPL)
        self._temp_path = self._find_thermal("x86_pkg_temp")
        self._gpu_idle = next(iter(sorted(Path("/sys/class/drm").glob("card*/device/tile0/gt0/gtidle/idle_residency_ms"))), None)
        psutil.cpu_percent(None)

    @staticmethod
    def _ensure_readable(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            path.read_text()
            return True
        except PermissionError:
            # RAPL is root-only since Platypus; sudo is passwordless on the robot PC (see CLAUDE.md).
            result = subprocess.run(["sudo", "-n", "chmod", "a+r", str(path)], capture_output=True, check=False)
            if result.returncode != 0:
                logger.warning("RAPL energy unreadable (power metrics off): {}", result.stderr.decode().strip())
                return False
            return True

    @staticmethod
    def _find_thermal(kind: str) -> Path | None:
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                if (zone / "type").read_text().strip() == kind:
                    return zone / "temp"
            except OSError:
                continue
        return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sysmon", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def energy_j(self) -> float:
        """Package energy accumulated since start (J); a cycle's energy is the difference of two reads."""
        with self._lock:
            return self._energy_j

    def _delta(self, key: str, value: float, now: float) -> tuple[float, float] | None:
        previous = self._prev.get(key)
        previous_t = self._prev.get(f"{key}:t")
        self._prev[key], self._prev[f"{key}:t"] = value, now
        if previous is None or previous_t is None or now <= previous_t:
            return None
        return value - previous, now - previous_t

    def _sample(self) -> SystemSample:
        now = time.monotonic()
        power = None
        if self._rapl_ok:
            try:
                energy = float(self.RAPL.read_text())
                delta = self._delta("rapl", energy, now)
                if delta is not None:
                    d_uj, dt = delta
                    if d_uj < 0:
                        d_uj += float(self.RAPL_MAX.read_text())
                    with self._lock:
                        self._energy_j += d_uj / 1e6
                    power = d_uj / 1e6 / dt
            except OSError:
                power = None
        temp = None
        if self._temp_path is not None:
            try:
                temp = float(self._temp_path.read_text()) / 1000.0
            except OSError:
                temp = None
        npu = None
        if self.NPU_BUSY.exists():
            try:
                delta = self._delta("npu", float(self.NPU_BUSY.read_text()), now)
                if delta is not None:
                    npu = max(0.0, min(100.0, delta[0] / 1e6 / delta[1] * 100.0))
            except OSError:
                npu = None
        gpu = None
        if self._gpu_idle is not None:
            try:
                delta = self._delta("gpu", float(self._gpu_idle.read_text()), now)
                if delta is not None:
                    gpu = max(0.0, min(100.0, 100.0 - delta[0] / 1000.0 / delta[1] * 100.0))
            except OSError:
                gpu = None
        return SystemSample(
            t=time.time(),
            cpu_pct=psutil.cpu_percent(None),
            ram_pct=psutil.virtual_memory().percent,
            package_w=None if power is None else round(power, 2),
            package_temp_c=temp,
            npu_pct=None if npu is None else round(npu, 1),
            gpu_pct=None if gpu is None else round(gpu, 1),
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.history.append(self._sample())
            except Exception:  # noqa: BLE001 - telemetry must never stop the cell
                logger.exception("system sample failed")
            self._stop.wait(1.0)

    def latest(self) -> SystemSample | None:
        return self.history[-1] if self.history else None

    def series(self, n: int = 120) -> list[dict[str, float | None]]:
        return [asdict(sample) for sample in list(self.history)[-n:]]


@dataclass
class CycleRecord:
    id: int
    run: str
    started: float
    ended: float | None = None
    outcome: str = "running"
    """``good`` / ``bad`` (placed in that bin), ``no_piece``, ``aborted``, ``failed``."""
    decision_reason: str = ""
    score: float | None = None
    scores: list[float] = field(default_factory=list)
    camera_scores: dict[str, list[float]] = field(default_factory=dict)
    """Usable-frame scores per inspection camera."""
    camera_decision: dict[str, float] = field(default_factory=dict)
    opinions: dict[str, Any] = field(default_factory=dict)
    """Verdict of each method (anomalib, algorithm, manual) on this brick, whichever decided."""
    """Aggregated score per camera that the fusion combined."""
    frames_seen: int = 0
    frames_usable: int = 0
    recaptures: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    """Seconds spent in each stage (pick_present, inspect, decide, place_good…)."""
    decisions: list[dict[str, Any]] = field(default_factory=list)
    """Every transition taken during the cycle and why: {t (s into the cycle), stage, next, text, ok}."""
    pick_attempts: int = 0
    energy_j: float | None = None
    inferences: int = 0
    truth: str | None = None
    """Operator label: ``good`` or ``bad``."""
    error: str | None = None

    @property
    def duration(self) -> float | None:
        return None if self.ended is None else self.ended - self.started


class CycleBook:
    """Cycle history of the current run, persisted line by line, with accuracy derived from operator labels."""

    def __init__(self, runs_dir: Path) -> None:
        self._dir = runs_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self._cycles: list[CycleRecord] = []

    def new_run(self) -> str:
        with self._lock:
            self.run_id = time.strftime("%Y%m%d-%H%M%S")
            self._cycles = []
            return self.run_id

    def open(self) -> CycleRecord:
        with self._lock:
            record = CycleRecord(id=len(self._cycles) + 1, run=self.run_id, started=time.time())
            self._cycles.append(record)
            return record

    def close(self, record: CycleRecord) -> None:
        with self._lock:
            record.ended = time.time()
            self._append(record)

    def label(self, cycle_id: int, truth: str | None) -> CycleRecord:
        if truth not in {"good", "bad", None}:
            raise ValueError("truth must be 'good', 'bad' or null")
        with self._lock:
            for record in self._cycles:
                if record.id == cycle_id:
                    record.truth = truth
                    self._append(record)
                    return record
        raise KeyError(cycle_id)

    def _append(self, record: CycleRecord) -> None:
        line = json.dumps({**asdict(record), "duration": record.duration})
        with (self._dir / f"{record.run}.jsonl").open("a") as handle:
            handle.write(line + "\n")

    def cycles(self) -> list[CycleRecord]:
        with self._lock:
            return list(self._cycles)

    def stats(self) -> dict[str, object]:
        cycles = [c for c in self.cycles() if c.ended is not None]
        sorted_ = [c for c in cycles if c.outcome in {"good", "bad"}]
        durations = [c.duration for c in sorted_ if c.duration is not None]
        energies = [c.energy_j for c in sorted_ if c.energy_j is not None]
        labelled = [c for c in sorted_ if c.truth is not None]
        tp = sum(1 for c in labelled if c.outcome == "bad" and c.truth == "bad")
        tn = sum(1 for c in labelled if c.outcome == "good" and c.truth == "good")
        fp = sum(1 for c in labelled if c.outcome == "bad" and c.truth == "good")
        fn = sum(1 for c in labelled if c.outcome == "good" and c.truth == "bad")
        span = (sorted_[-1].ended or 0) - sorted_[0].started if len(sorted_) > 1 else None
        return {
            "run": self.run_id,
            "cycles": len(cycles),
            "sorted": len(sorted_),
            "good": sum(1 for c in sorted_ if c.outcome == "good"),
            "bad": sum(1 for c in sorted_ if c.outcome == "bad"),
            "failed": sum(1 for c in cycles if c.outcome in {"failed", "aborted"}),
            "recaptured": sum(1 for c in sorted_ if c.recaptures > 0),
            "cycle_s_mean": round(float(np.mean(durations)), 2) if durations else None,
            "cycle_s_p95": round(float(np.percentile(durations, 95)), 2) if durations else None,
            "throughput_per_min": round(len(sorted_) / span * 60.0, 2) if span else None,
            "energy_j_mean": round(float(np.mean(energies)), 1) if energies else None,
            "labelled": len(labelled),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": round((tp + tn) / len(labelled), 3) if labelled else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }

    def csv(self) -> str:
        buffer = io.StringIO()
        columns = [
            "run", "id", "started", "duration", "outcome", "truth", "score", "frames_seen", "frames_usable",
            "recaptures", "energy_j", "inferences", "decision_reason", "stages", "pick_attempts", "decisions", "error",
        ]
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for record in self.cycles():
            row = {
                **asdict(record),
                "duration": record.duration,
                "stages": json.dumps(record.stages),
                "decisions": " | ".join(f"{d['stage']}→{d['next']}: {d['text']}" for d in record.decisions),
            }
            writer.writerow({key: row.get(key) for key in columns})
        return buffer.getvalue()
