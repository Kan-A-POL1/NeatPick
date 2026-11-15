"""Inspection keeps and scores one frame per camera (the most usable one) instead of every frame of the window."""

from __future__ import annotations

import time
from pathlib import Path

from sortops.cell import Cell
from sortops.config import Settings
from sortops.safety import Mode


def _one_cycle(tmp_path: Path, best: bool) -> tuple[Cell, list[str]]:
    settings = Settings(
        tmp_path,
        {"safety": {"watchdog_enabled": False}, "run": {"max_cycles": 1}, "inspection": {"best_frame_only": best, "anomalib_cameras": ["front", "wrist"]}},
    )
    cell = Cell(settings, tmp_path)
    shots: list[str] = []
    original = cell.record_shot

    def counting(record, phase, camera, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        shots.append(camera)
        return original(record, phase, camera, *args, **kwargs)

    cell.record_shot = counting  # type: ignore[method-assign]
    cell.start()
    cell.set_mode(Mode.LIVE)
    cell.start_cycles(loop=False)
    deadline = time.monotonic() + 80
    while cell.cycle_running() and time.monotonic() < deadline:
        time.sleep(0.2)
    return cell, shots


def test_one_scored_and_kept_frame_per_camera(tmp_path: Path) -> None:
    cell, shots = _one_cycle(tmp_path, best=True)
    try:
        (cycle,) = cell.cycles.cycles()
        assert cycle.outcome in {"good", "bad"}, (cycle.outcome, cycle.decision_reason, cycle.error)
        assert sorted(shots) == ["front", "wrist"]  # one kept image per camera, no duplicates
        assert {cam: len(v) for cam, v in cycle.camera_scores.items()} == {"front": 1, "wrist": 1}
        assert cycle.frames_seen > 2  # many frames were rated, only the best of each camera was analyzed
    finally:
        cell.shutdown()


def test_every_frame_mode_still_scores_the_whole_window(tmp_path: Path) -> None:
    cell, shots = _one_cycle(tmp_path, best=False)
    try:
        (cycle,) = cell.cycles.cycles()
        assert cycle.outcome in {"good", "bad"}, (cycle.outcome, cycle.decision_reason, cycle.error)
        assert len(shots) > 2
        assert all(len(v) >= 2 for v in cycle.camera_scores.values())
    finally:
        cell.shutdown()
