"""Release reflex: a place segment ends as soon as the gripper is opened wide over the bowl."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sortops.cell import Cell
from sortops.config import Settings
from sortops.safety import Mode


def _one_cycle(tmp_path: Path, release: dict) -> Cell:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "release": release,
                                   "decision": {"verdict_source": "operator"}})
    cell = Cell(settings, tmp_path)
    cell.start()
    cell.set_mode(Mode.LIVE)
    cell.start_cycles(loop=False)
    deadline = time.monotonic() + 60
    answered = False
    while cell.cycle_running() and time.monotonic() < deadline:
        if not answered and cell.awaiting_verdict():
            waiting = cell.snapshot()["inspection"]
            cell.give_verdict("good", None, waiting["run"], waiting["cycle"])
            answered = True
        time.sleep(0.05)
    assert not cell.cycle_running()
    return cell


def test_release_opens_wide_and_ends_the_place_segment(tmp_path: Path) -> None:
    cell = _one_cycle(tmp_path, {"enabled": True, "trigger": 10.0, "open": 100.0, "hold_s": 0.2})
    try:
        record = cell.cycles.cycles()[0]
        assert record.outcome == "good"
        messages = [e["msg"] for e in cell.events]
        assert any(m.startswith("place_good: release at") for m in messages), messages
        # (the arm then goes home after every cycle, so its final pose says nothing about the release)
        assert "release" in record.stages
        # the scripted place takes 4.2 s plus settling: the reflex cuts it right after the opening
        assert record.stages["place_good"] < 4.0
    finally:
        cell.shutdown()


def test_release_disabled_plays_the_whole_policy(tmp_path: Path) -> None:
    cell = _one_cycle(tmp_path, {"enabled": False})
    try:
        messages = [e["msg"] for e in cell.events]
        assert not any("release at" in m for m in messages)
        assert cell.cycles.cycles()[0].stages["place_good"] >= 4.0
    finally:
        cell.shutdown()


@pytest.mark.parametrize("patch", [{"trigger": 0.0}, {"open": 150.0}, {"hold_s": -1.0}, {"trigger": 50.0, "open": 20.0},
                                   {"trigger": 55.0, "open": 60.0}])
def test_release_settings_are_bounded(tmp_path: Path, patch: dict) -> None:
    settings = Settings(tmp_path, {})
    with pytest.raises(ValueError):
        settings.update({"release": patch})
