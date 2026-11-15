"""Step 1 → inspection: grasp read from the gripper, adaptive admission zone, no brick in time → home and retry."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from sortops.cell import Cell
from sortops.config import Settings
from sortops.pipeline.grasp import GraspDetector, admission_radius
from sortops.safety import Mode

GRASP = {"open_below": 15.0, "min_closed_pos": 2.0, "min_load": 10.0, "hold_s": 0.3, "lost_s": 0.5}


def _feed(detector: GraspDetector, position: float, load: float, start: float, seconds: float) -> float:
    t = start
    while t < start + seconds:
        detector.update(position, load, t)
        t += 1 / 30
    return t


def test_grasp_needs_a_closed_squeezing_gripper_for_hold_s() -> None:
    detector = GraspDetector(GRASP)
    t = _feed(detector, 3.0, 18.0, 0.0, 0.2)
    assert not detector.held  # not confirmed yet
    _feed(detector, 3.0, 18.0, t, 0.2)
    assert detector.held and "brick held" in detector.last.reason


@pytest.mark.parametrize(
    ("position", "load", "why"),
    [(1.0, 4.0, "closed on nothing"), (1.0, 18.0, "closed on nothing"), (3.0, 4.0, "no grip force"), (40.0, 2.0, "open")],
)
def test_no_grasp_when_empty_weak_or_open(position: float, load: float, why: str) -> None:
    detector = GraspDetector(GRASP)
    _feed(detector, position, load, 0.0, 1.0)
    assert not detector.held and why in detector.last.reason


def test_grasp_lost_when_the_brick_drops_or_the_gripper_opens() -> None:
    detector = GraspDetector(GRASP)
    t = _feed(detector, 3.0, 18.0, 0.0, 0.5)
    t = _feed(detector, 1.0, 3.0, t, 0.3)
    assert detector.held  # a short dip is not a drop
    _feed(detector, 1.0, 3.0, t, 0.4)
    assert not detector.held and "fully closed" in (detector.lost_reason or "")
    detector = GraspDetector(GRASP)
    t = _feed(detector, 3.0, 18.0, 0.0, 0.5)
    detector.update(30.0, 0.0, t)
    assert not detector.held and detector.lost_reason == "gripper opened"


def test_admission_zone_grows_by_half_every_ten_admissible_frames() -> None:
    cfg = {"pose_tolerance_m": 0.05, "zone_step_frames": 10, "zone_growth": 0.5, "zone_max_growth": 2.0}
    assert admission_radius(cfg, 0) == pytest.approx((0.10, 0.0))
    assert admission_radius(cfg, 9) == pytest.approx((0.10, 0.0))
    assert admission_radius(cfg, 10) == pytest.approx((0.15, 0.5))
    assert admission_radius(cfg, 20) == pytest.approx((0.20, 1.0))
    assert admission_radius(cfg, 500) == pytest.approx((0.30, 2.0))  # capped at +200 %


def _run(tmp_path: Path, overrides: dict[str, Any], *, supply: bool = True, seconds: float = 60) -> tuple[Cell, list[dict]]:
    base = {"safety": {"watchdog_enabled": False}, "run": {"max_cycles": 1}, "decision": {"verdict_source": "cv"}}
    for key, value in overrides.items():
        base[key] = {**base.get(key, {}), **value}
    cell = Cell(Settings(tmp_path, base), tmp_path)
    assert cell.world is not None
    cell.world.supply = supply
    cell.start()
    cell.set_mode(Mode.LIVE)
    cell.start_cycles(loop=False)
    seen: list[dict] = []
    deadline = time.monotonic() + seconds
    while cell.cycle_running() and time.monotonic() < deadline:
        transition = cell.snapshot()["transition"]
        if transition is not None:
            seen.append(transition)
        time.sleep(0.1)
    assert not cell.cycle_running()
    return cell, seen


def test_no_brick_in_time_goes_home_waits_and_retries(tmp_path: Path) -> None:
    cell, seen = _run(
        tmp_path,
        {"segments": {"pick_present": {"timeout_s": 2.0}}, "run": {"pick_attempts": 2}, "pick": {"retry_wait_s": 0.5}},
        supply=False,
    )
    try:
        (record,) = cell.cycles.cycles()
        assert record.outcome == "no_piece" and record.pick_attempts == 2
        assert [(d["stage"], d["next"]) for d in record.decisions] == [("pick_present", "home"), ("pick_present", "end")]
        assert "no brick held after 2 s" in record.decisions[0]["text"] and "wait 0.5 s" in record.decisions[0]["text"]
        assert record.stages["home"] > 0 and record.frames_seen == 0  # nothing sampled without a grasp
        gate = next(g for g in seen[-1]["gates"] if g["key"] == "grasp")
        assert not gate["ok"] and {t["label"] for t in seen[-1]["timers"]} >= {"No brick → home"}
    finally:
        cell.shutdown()


def test_brick_presented_off_pose_is_admitted_once_the_zone_grows(tmp_path: Path) -> None:
    # Presented ≈ 12 cm from the taught pose: outside the 10 cm base radius, inside 15 cm (+50 % after 10 frames).
    cell, seen = _run(tmp_path, {"inspection": {"pose_xyz": [0.33, 0.13, 0.215], "period_s": 0.2, "window_s": 1.0}})
    try:
        (record,) = cell.cycles.cycles()
        assert record.outcome in {"good", "bad"}, (record.outcome, record.error, record.decisions)
        step1 = record.decisions[0]
        assert (step1["stage"], step1["next"], step1["ok"]) == ("pick_present", "inspect", True)
        assert "held (load" in step1["text"] and "+50 % after" in step1["text"]
        assert record.decisions[-1]["stage"] == "inspect" and record.decisions[-1]["next"].startswith("place_")
        assert any(not g["ok"] for t in seen for g in t["gates"] if g["key"] == "zone")  # first refused…
        assert any(t["zone"]["growth"] >= 0.5 for t in seen)  # …then the zone widened
    finally:
        cell.shutdown()


def test_held_but_never_admitted_goes_on_after_present_max_s(tmp_path: Path) -> None:
    cell, _ = _run(
        tmp_path,
        {"inspection": {"pose_xyz": [0.0, 0.4, 0.4]}, "pick": {"present_max_s": 3.0, "zone_max_growth": 0.0}},
    )
    try:
        (record,) = cell.cycles.cycles()
        step1 = record.decisions[0]
        assert (step1["next"], step1["ok"]) == ("inspect", False) and "inspecting what we have" in step1["text"]
    finally:
        cell.shutdown()


def test_grasp_settings_are_validated(tmp_path: Path) -> None:
    settings = Settings(tmp_path)
    with pytest.raises(ValueError, match="min_closed_pos"):
        settings.update({"pick": {"min_closed_pos": 20.0, "open_below": 15.0}})
    with pytest.raises(ValueError, match="pick.min_load"):
        settings.update({"pick": {"min_load": 0.0}})


def test_no_new_pick_when_home_is_not_reached(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {"safety": {"watchdog_enabled": False}, "run": {"max_cycles": 1, "pick_attempts": 3},
         "segments": {"pick_present": {"timeout_s": 1.5}}, "decision": {"verdict_source": "cv"}},
    )
    cell = Cell(settings, tmp_path)
    assert cell.world is not None
    cell.world.supply = False
    picks: list[str] = []
    segment = cell._segment

    def counting(role, *args, **kwargs):
        picks.append(role)
        return segment(role, *args, **kwargs)

    cell._segment = counting  # type: ignore[method-assign]
    cell._home_motion = lambda: False  # type: ignore[method-assign]
    cell.start()
    cell.set_mode(Mode.LIVE)
    cell.start_cycles(loop=False)
    deadline = time.monotonic() + 30
    while cell.cycle_running() and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        (record,) = cell.cycles.cycles()
        assert picks == ["pick_present"]  # no second pick from an unknown pose
        assert record.outcome == "failed" and record.error == "home not reached before a pick retry"
        assert (record.decisions[-1]["stage"], record.decisions[-1]["next"]) == ("home", "end")
    finally:
        cell.shutdown()


def test_present_mode_retry_forgets_the_failed_attempt(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {"safety": {"watchdog_enabled": False}, "run": {"max_cycles": 1, "pick_attempts": 2},
         "segments": {"pick_present": {"timeout_s": 1.5}}, "present": {"enabled": True},
         "pick": {"retry_wait_s": 0.0}},
    )
    cell = Cell(settings, tmp_path)
    assert cell.world is not None
    cell.world.supply = False
    resets: list[int] = []
    import sortops.cell as cell_module

    original = cell_module._Inspection.reset_pick

    def spy(self):
        resets.append(len(self.record.decisions))
        return original(self)

    cell_module._Inspection.reset_pick = spy  # type: ignore[method-assign]
    try:
        cell.start()
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=False)
        deadline = time.monotonic() + 40
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        (record,) = cell.cycles.cycles()
        assert record.outcome == "no_piece" and record.pick_attempts == 2
        assert resets == [1]  # once, between the two attempts
    finally:
        cell_module._Inspection.reset_pick = original  # type: ignore[method-assign]
        cell.shutdown()


def test_opening_force_is_not_a_grip() -> None:
    # An opening gripper reads a large negative load (≈ -24 live on the arm): never a grasp, even inside 2-15.
    detector = GraspDetector(GRASP)
    _feed(detector, 8.0, -24.0, 0.0, 1.0)
    assert not detector.held and "no grip force" in detector.last.reason
