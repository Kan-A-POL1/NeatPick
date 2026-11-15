"""Inspection: is a frame worth scoring (usability), and what do the usable scores say (decision policy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from sortops.pipeline.brick_crop import BrickCrop


@dataclass
class Usability:
    score: float
    """0..1, the weakest of the factors below."""
    usable: bool
    sharpness: float
    exposure: float
    brick_fraction: float
    pose_error_m: float
    speed: float
    reason: str


def assess(
    crop: BrickCrop,
    ee_xyz: np.ndarray,
    speed: float,
    cfg: dict[str, Any],
) -> Usability:
    """Rate one inspection frame from its brick crop: sharp, well exposed, a brick in view, arm at the pose and slow."""
    gray = cv2.cvtColor(crop.region, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    brick = crop.brick_fraction if crop.found else 0.0
    pose_error = float(np.linalg.norm(np.asarray(ee_xyz, np.float64) - np.asarray(cfg["pose_xyz"], np.float64)))

    factors = {
        "blurred": min(1.0, sharpness / max(1e-6, cfg["min_sharpness"])),
        "exposure": 1.0 - min(1.0, max(0.0, abs(mean - 128.0) - 70.0) / 50.0),
        "no brick": min(1.0, brick / max(1e-6, cfg["min_brick_fraction"])),
        "off pose": max(0.0, 1.0 - max(0.0, pose_error - cfg["pose_tolerance_m"]) / cfg["pose_tolerance_m"]),
        "moving": max(0.0, 1.0 - max(0.0, speed - cfg["max_speed_for_capture"]) / cfg["max_speed_for_capture"]),
    }
    weakest = min(factors, key=lambda key: factors[key])
    score = float(factors[weakest])
    usable = score >= cfg["usability_threshold"]
    return Usability(
        score=round(score, 3),
        usable=usable,
        sharpness=round(sharpness, 1),
        exposure=round(mean, 1),
        brick_fraction=round(brick, 4),
        pose_error_m=round(pose_error, 4),
        speed=round(speed, 2),
        reason="ok" if usable else weakest,
    )


class Verdict(StrEnum):
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    NO_PIECE = "no_piece"


@dataclass
class Decision:
    verdict: Verdict
    score: float | None
    reason: str
    camera_scores: dict[str, float] = field(default_factory=dict)
    """Aggregated score of each inspection camera that had enough usable frames (what the fusion combined)."""
    no_brick: bool = False
    """BAD because the arm presented but no brick was recognized (counts toward the end of the batch)."""


FUSIONS = ("max", "mean", "min", "weighted")


def fuse(camera_scores: dict[str, float], fusion: str, weights: dict[str, float]) -> float:
    """One score from the per-camera scores.

    ``max``: defect as soon as one camera sees it (no defect slips through, more false rejects).
    ``mean``: average. ``min``: defect only if every camera sees it. ``weighted``: weighted average (``weights``).
    """
    values = list(camera_scores.values())
    if fusion == "max":
        return max(values)
    if fusion == "min":
        return min(values)
    if fusion == "mean":
        return float(np.mean(values))
    if fusion == "weighted":
        total = sum(max(0.0, weights.get(cam, 0.0)) for cam in camera_scores)
        if total <= 0:
            return float(np.mean(values))
        return float(sum(max(0.0, weights.get(cam, 0.0)) * v for cam, v in camera_scores.items()) / total)
    raise ValueError(f"Unknown fusion {fusion!r}")


def aggregate(scores: list[float], method: str) -> float:
    values = np.asarray(scores, np.float64)
    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method.startswith("p") and method[1:].isdigit():
        return float(np.percentile(values, int(method[1:])))
    raise ValueError(f"Unknown aggregate {method!r}")


def decide(
    scores: list[float], frames_seen: int, no_brick_frames: int, cfg: dict[str, Any], method: str, min_frames: int
) -> Decision:
    """Single-camera decision (one camera, fusion irrelevant)."""
    return decide_cameras({"camera": scores}, frames_seen, no_brick_frames, cfg, method, min_frames)


def decide_cameras(
    scores: dict[str, list[float]],
    frames_seen: int,
    no_brick_frames: int,
    cfg: dict[str, Any],
    method: str,
    min_frames: int,
) -> Decision:
    """Per-camera usable scores → aggregate per camera (``method``) → fusion (``cfg['fusion']``) → good / bad / uncertain.

    The arm presented (frames were taken) but no brick was recognized in ``no_brick_defect_ratio`` of them or more:
    an unknown or hidden piece, never a good one → defect bin. Otherwise a camera takes
    part only with ``min_frames`` usable frames; with none taking part the verdict is uncertain.
    """
    if frames_seen > 0 and no_brick_frames / frames_seen >= cfg["no_brick_defect_ratio"]:
        return Decision(
            Verdict.BAD, None, f"no brick recognized in {no_brick_frames}/{frames_seen} frames → defect bin", no_brick=True
        )
    if not any(scores.values()):
        return Decision(Verdict.UNCERTAIN, None, f"0 usable frames of {frames_seen}")
    per_camera = {cam: aggregate(values, method) for cam, values in scores.items() if len(values) >= min_frames}
    if not per_camera:
        counts = ", ".join(f"{cam} {len(values)}" for cam, values in scores.items())
        best = max((aggregate(v, method) for v in scores.values() if v), default=None)
        return Decision(Verdict.UNCERTAIN, best, f"too few usable frames ({counts}), need {min_frames}")
    fusion = cfg.get("fusion", "max")
    value = fuse(per_camera, fusion, cfg.get("weights", {}))
    detail = " · ".join(f"{cam} {v:.2f}" for cam, v in per_camera.items())
    label = f"{fusion}({detail})" if len(per_camera) > 1 else f"{method} {detail}"
    rounded = {cam: round(v, 4) for cam, v in per_camera.items()}
    if value >= cfg["bad_above"]:
        return Decision(Verdict.BAD, value, f"{label} = {value:.2f} ≥ {cfg['bad_above']:.2f}", rounded)
    if value <= cfg["good_below"]:
        return Decision(Verdict.GOOD, value, f"{label} = {value:.2f} ≤ {cfg['good_below']:.2f}", rounded)
    return Decision(Verdict.UNCERTAIN, value, f"{label} = {value:.2f} in gray zone", rounded)
