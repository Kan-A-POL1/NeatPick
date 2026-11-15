"""Readable anomaly maps: one fixed scale for every frame, the defect line drawn, the peak marked, regions boxed.

An export's own map is normalized per frame, so every frame looks "hot" somewhere and two frames cannot be compared.
Here the map is put in **good-brick units** from the calibration on known-good bricks, so the colours mean the same
thing on every frame:

- ``u = 0``: a typical good brick (median good mean of the raw map: background of a good brick);
- ``u = 1``: the hottest patch of an unusually clean-but-normal brick (good peak median + its spread, i.e. the 95th
  percentile of good bricks' worst patch). Anything above is hotter than the worst patch of 95 % of good bricks:
  that is the **defect contour**;
- ``u >= 2``: saturated.

Fallbacks: calibrated on the raw score only (no map terms) → 0 at the map's median, 1 at the raw-score defect line;
not calibrated at all → the export's normalized map, 1 at 0.5 (its own threshold), flagged ``calibrated: False``.

The connected areas above the contour become **regions** (normalized boxes on the crop), offered to the operator as a
pre-annotation of the defects; ``peak`` is where the map is hottest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

MIN_REGION_FRACTION = 0.002
"""Areas above the contour smaller than this share of the map are noise, not regions."""


@dataclass
class HeatInfo:
    scale: str
    """``terms`` (peak/mean calibration), ``raw`` (raw-score calibration) or ``export`` (uncalibrated)."""
    calibrated: bool
    max_units: float
    """Hottest point in good-brick units (1 = defect contour)."""
    peak: tuple[float, float]
    """Normalized (x, y) of the hottest point on the crop."""
    regions: list[list[float]] = field(default_factory=list)
    """Normalized (x, y, w, h) boxes on the crop of the areas above the contour, largest first."""


def units(raw_map: np.ndarray | None, heat01: np.ndarray | None, calibration: dict[str, Any] | None) -> tuple[np.ndarray, str, bool]:
    """The map in good-brick units (see module doc) and which scale was used."""
    if raw_map is not None and calibration and calibration.get("terms"):
        t = calibration["terms"]
        base = float(t["mean_median"])
        line = float(t["peak_median"]) + float(t["peak_spread"])
        return (np.asarray(raw_map, np.float32) - base) / max(line - base, 1e-6), "terms", True
    if raw_map is not None and calibration and "threshold" in calibration:
        values = np.asarray(raw_map, np.float32)
        base = float(np.median(values))
        line = float(calibration["threshold"])
        if line > base:
            return (values - base) / (line - base), "raw", True
    if heat01 is not None:
        return np.asarray(heat01, np.float32) / 0.5, "export", False
    if raw_map is not None:
        values = np.asarray(raw_map, np.float32)
        low, high = float(np.percentile(values, 50)), float(values.max())
        return (values - low) / max(high - low, 1e-6), "export", False
    raise ValueError("no anomaly map")


def analyse(u: np.ndarray) -> tuple[float, tuple[float, float], list[list[float]]]:
    """(max units, peak position, regions above 1) of a map in good-brick units."""
    h, w = u.shape[:2]
    y, x = np.unravel_index(int(np.argmax(u)), u.shape)
    mask = (u >= 1.0).astype(np.uint8)
    regions: list[tuple[float, list[float]]] = []
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if area >= MIN_REGION_FRACTION * h * w:
            regions.append((float(area), [bx / w, by / h, bw / w, bh / h]))
    regions.sort(key=lambda item: -item[0])
    return float(u.max()), ((x + 0.5) / w, (y + 0.5) / h), [[round(v, 4) for v in box] for _, box in regions]


def render(
    crop_rgb: np.ndarray,
    raw_map: np.ndarray | None,
    heat01: np.ndarray | None,
    calibration: dict[str, Any] | None,
) -> tuple[np.ndarray, HeatInfo]:
    """RGB overlay at the crop's resolution + what it shows.

    Below u = 0.25 the crop is left untouched (a good brick stays readable); above, the colour (blue → red, saturated
    at u = 2) is blended in proportion to u; the u = 1 contour is drawn in white with a dark outline; the peak gets a
    ring. A corner badge gives the scale and the max in units.
    """
    height, width = crop_rgb.shape[:2]
    source = raw_map if raw_map is not None else heat01
    if source is None:
        raise ValueError("no anomaly map")
    u, scale, calibrated_ = units(raw_map, heat01, calibration)
    u = cv2.resize(np.nan_to_num(u, nan=0.0, posinf=4.0, neginf=0.0), (width, height), interpolation=cv2.INTER_LINEAR)
    max_units, peak, regions = analyse(u)

    level = np.clip(u / 2.0, 0.0, 1.0)
    colours = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR
    alpha = np.clip((u - 0.25) / 0.75, 0.0, 1.0)[..., None] * 0.65
    base = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    out = (base * (1 - alpha) + colours.astype(np.float32) * alpha).astype(np.uint8)

    contours, _ = cv2.findContours((u >= 1.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(1, width // 160)
    cv2.drawContours(out, contours, -1, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.drawContours(out, contours, -1, (255, 255, 255), thick, cv2.LINE_AA)
    px, py = int(peak[0] * width), int(peak[1] * height)
    radius = max(6, width // 24)
    cv2.circle(out, (px, py), radius, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.circle(out, (px, py), radius, (0, 255, 255) if max_units >= 1 else (255, 255, 255), thick, cv2.LINE_AA)

    label = f"{max_units:.1f}x" + ("" if calibrated_ else " (uncal.)")
    font = max(0.35, width / 640)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    cv2.rectangle(out, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    cv2.putText(out, label, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, font,
                (80, 80, 255) if max_units >= 1 else (220, 220, 220), 1, cv2.LINE_AA)
    info = HeatInfo(scale=scale, calibrated=calibrated_, max_units=round(max_units, 3),
                    peak=(round(peak[0], 4), round(peak[1], 4)), regions=regions)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info
