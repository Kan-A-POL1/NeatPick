"""Is a brick held in the gripper? Read from the gripper motor, not from the policy's intent.

Measured in the Studio datasets (SO-101 follower, normalized gripper 0..100):
holding a Duplo brick → position ≈ 3 (command 0), load ≈ 17-20; closed on nothing → position ≈ 1, load ≈ 2-5;
opening → load negative (down to ≈ -30, also live on the arm). So "held" = closed (below ``open_below``), not fully
(≥ ``min_closed_pos``), and squeezing (signed load ≥ ``min_load``: positive = closing force) for ``hold_s``. Once held, the brick counts as lost after ``lost_s`` without squeezing
(dropped) or once the gripper opens again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GRASP_DEFAULTS: dict[str, Any] = {
    "open_below": 15.0,
    "min_closed_pos": 2.0,
    "min_load": 10.0,
    "hold_s": 0.3,
    "lost_s": 0.5,
}


@dataclass
class GraspReading:
    held: bool
    position: float
    load: float
    closed: bool
    """Below ``open_below`` and at least ``min_closed_pos``: jaws shut on something."""
    squeezing: bool
    """load ≥ ``min_load`` (signed: an opening gripper reads a negative load)."""
    held_for: float
    """Seconds since the grasp was confirmed (0 when not held)."""
    reason: str


class GraspDetector:
    """Feed it the gripper position and load on every control tick (``update``); ``held`` says if a brick is held."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = {**GRASP_DEFAULTS, **cfg}
        self.held = False
        self.held_since: float | None = None
        self._candidate_since: float | None = None
        self._weak_since: float | None = None
        self.peak_load = 0.0
        self.lost_reason: str | None = None
        self.last = GraspReading(False, 0.0, 0.0, False, False, 0.0, "no reading yet")

    def update(self, position: float, load: float, now: float) -> GraspReading:
        cfg = self.cfg
        closed = float(cfg["min_closed_pos"]) <= position < float(cfg["open_below"])
        squeezing = load >= float(cfg["min_load"])
        if self.held:
            if position >= float(cfg["open_below"]):
                self._lose("gripper opened")
            elif not (closed and squeezing):
                self._weak_since = self._weak_since or now
                if now - self._weak_since >= float(cfg["lost_s"]):
                    self._lose("brick lost: gripper fully closed" if position < float(cfg["min_closed_pos"]) else "brick lost: no grip force")
            else:
                self._weak_since = None
        if not self.held:
            if closed and squeezing:
                self._candidate_since = self._candidate_since or now
                if now - self._candidate_since >= float(cfg["hold_s"]):
                    self.held, self.held_since, self._weak_since = True, now, None
            else:
                self._candidate_since = None
        if self.held:
            self.peak_load = max(self.peak_load, load)
        reason = self._reason(position, load, closed, squeezing)
        held_for = 0.0 if self.held_since is None or not self.held else now - self.held_since
        self.last = GraspReading(self.held, position, load, closed, squeezing, round(held_for, 2), reason)
        return self.last

    def _lose(self, why: str) -> None:
        self.held, self.held_since, self._candidate_since, self._weak_since = False, None, None, None
        self.lost_reason = why

    def _reason(self, position: float, load: float, closed: bool, squeezing: bool) -> str:
        cfg = self.cfg
        if self.held:
            return f"brick held: gripper {position:.0f}, load {load:.0f} ≥ {cfg['min_load']:.0f}"
        if position >= float(cfg["open_below"]):
            return f"gripper open ({position:.0f} ≥ {cfg['open_below']:.0f})"
        if position < float(cfg["min_closed_pos"]):
            return f"closed on nothing ({position:.1f} < {cfg['min_closed_pos']:.0f})"
        if not squeezing:
            return f"closed, no grip force (load {load:.0f} < {cfg['min_load']:.0f})"
        return f"squeezing, confirming ({cfg['hold_s']:.1f} s)"


def admission_radius(cfg: dict[str, Any], admissible: int) -> tuple[float, float]:
    """Radius (m) of the inspection admission zone after ``admissible`` brick-seen frames, and its growth factor.

    Base = 2 × ``pose_tolerance_m``; every ``zone_step_frames`` admissible frames add ``zone_growth`` (0.5 = +50 %),
    up to ``zone_max_growth``: the arm presenting a brick a bit off the taught pose is admitted after a while.
    """
    base = 2.0 * float(cfg["pose_tolerance_m"])
    step = max(1, int(cfg["zone_step_frames"]))
    growth = min(float(cfg["zone_max_growth"]), float(cfg["zone_growth"]) * (admissible // step))
    return base * (1.0 + growth), growth
