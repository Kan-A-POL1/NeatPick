"""Where is the brick in an inspection frame, and the square crop of it the anomaly model sees.

This is THE crop of the project: SortOps runs it on every inspection frame before scoring, and the Anomalib capture
app (``anomaly/capture_app.py``) imports this very module and reads the same ``inspection`` settings
(``<sortops data>/config.json``) to cut its training photos. A model is only as good as the match between the photos it
learned from and the frames it scores, so never re-implement this elsewhere: import it.

Method (``crop_mode`` "auto"): inside the search zone ``roi``, keep the saturated pixels whose hue is not excluded
(the red gripper fingers and the orange plate), take the largest blob, and cut a square around it with a margin,
resized to ``crop_size``. White, red and orange bricks are not found (the demo bricks are blue,
yellow, green…). ``crop_mode`` "fixed" cuts the search zone itself. Without a brick, the auto crop falls back to the
fixed one and says so (``found`` False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

CROP_DEFAULTS: dict[str, Any] = {
    # "auto": square around the brick found in the search zone; "fixed": the search zone itself.
    "crop_mode": "auto",
    # Side of the square image handed to the anomaly model (Anomalib's default training size).
    "crop_size": 256,
    # Space kept around the brick, as a fraction of its larger side, on each side.
    "crop_margin": 0.1,
    # OpenCV hue ranges (0..180) that are never a brick: red gripper fingers (≈170-180 and 0-8), orange plate (8-22).
    "crop_exclude_hues": [[0.0, 22.0], [160.0, 180.0]],
    "crop_min_saturation": 90.0,
    "crop_min_value": 60.0,
    # Smallest blob taken for a brick, as a fraction of the search zone (0.1 %: a 1x2 brick seen from the front camera
    # covers ~0.25 % of the frame).
    "crop_min_area": 0.001,
}

CROP_BOUNDS: dict[str, tuple[float, float]] = {
    "crop_size": (32, 1024),
    "crop_margin": (0, 1),
    "crop_min_saturation": (0, 255),
    "crop_min_value": (0, 255),
    "crop_min_area": (0, 1),
}


def validate_crop(cfg: dict[str, Any]) -> None:
    """Refuse crop settings the crop cannot run with (SortOps ``validate`` calls it on the inspection section)."""
    if cfg["crop_mode"] not in {"auto", "fixed"}:
        raise ValueError("inspection.crop_mode must be 'auto' or 'fixed'")
    if not isinstance(cfg["crop_size"], int) or isinstance(cfg["crop_size"], bool):
        raise ValueError("inspection.crop_size must be a whole number of pixels")
    for key, (low, high) in CROP_BOUNDS.items():
        if not low <= cfg[key] <= high:
            raise ValueError(f"inspection.{key} must be within [{low}, {high}], got {cfg[key]}")
    for band in cfg["crop_exclude_hues"]:
        if (
            not isinstance(band, list)
            or len(band) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in band)
            or not 0 <= band[0] <= band[1] <= 180
        ):
            raise ValueError("inspection.crop_exclude_hues must be [low, high] pairs with 0 ≤ low ≤ high ≤ 180")


@dataclass
class BrickCrop:
    image: np.ndarray
    """RGB, ``crop_size`` x ``crop_size``: what the anomaly model scores."""
    region: np.ndarray
    """The same pixels before resizing (sharpness and exposure are measured here, at the camera's resolution)."""
    box: tuple[float, float, float, float]
    """Normalized (x, y, w, h) of the crop in the frame, for drawing it on a preview."""
    found: bool
    """A brick blob was found (auto mode) or brick-coloured pixels fill ``crop_min_area`` of the zone (fixed mode)."""
    brick_fraction: float
    """Brick pixels / search-zone pixels."""


def _zone(shape: tuple[int, ...], roi: list[float]) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of a normalized ROI; the full frame when it is degenerate (under 8 px)."""
    height, width = shape[:2]
    x, y, w, h = roi
    x0, y0 = int(max(0.0, x) * width), int(max(0.0, y) * height)
    x1, y1 = int(min(1.0, x + w) * width), int(min(1.0, y + h) * height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, width, height
    return x0, y0, x1, y1


def brick_mask(image_rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Pixels that may belong to a brick: saturated, bright enough, hue outside every excluded band."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    mask = (hsv[..., 1] >= cfg["crop_min_saturation"]) & (hsv[..., 2] >= cfg["crop_min_value"])
    for low, high in cfg["crop_exclude_hues"]:
        mask &= ~((hue >= low) & (hue <= high))
    return mask.astype(np.uint8)


def _square(cx: float, cy: float, side: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Square of ``side`` px centred on (cx, cy), shrunk to fit the frame, then slid inside it."""
    side = int(round(min(side, width, height)))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def crop_brick(image_rgb: np.ndarray, cfg: dict[str, Any]) -> BrickCrop:
    """Crop the brick out of an RGB frame with the ``inspection`` settings (``roi`` + ``crop_*`` keys)."""
    height, width = image_rgb.shape[:2]
    zx0, zy0, zx1, zy1 = _zone(image_rgb.shape, cfg["roi"])
    zone = image_rgb[zy0:zy1, zx0:zx1]
    zone_area = float(zone.shape[0] * zone.shape[1])
    mask = brick_mask(zone, cfg)
    # Opening drops speckles (stud highlights, noise) so they never win over the brick.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_pixels = cfg["crop_min_area"] * zone_area
    size = int(cfg["crop_size"])

    # The largest blob alone, as when the training photos were cut: merging nearby blobs (for bricks split by the
    # fingers) re-cut 13 of the 33 wrist training photos differently.
    blob = None
    if count > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[best, cv2.CC_STAT_AREA] >= max(1.0, min_pixels):
            blob = stats[best]

    if cfg["crop_mode"] == "auto" and blob is not None:
        bx, by, bw, bh, area = (int(v) for v in blob)
        side = max(bw, bh) * (1.0 + 2.0 * cfg["crop_margin"])
        x0, y0, x1, y1 = _square(zx0 + bx + bw / 2.0, zy0 + by + bh / 2.0, side, width, height)
        found, fraction = True, area / zone_area
    else:
        x0, y0, x1, y1 = zx0, zy0, zx1, zy1
        fraction = float(mask.sum()) / zone_area
        found = blob is not None if cfg["crop_mode"] == "auto" else fraction >= cfg["crop_min_area"]

    region = image_rgb[y0:y1, x0:x1]
    interpolation = cv2.INTER_AREA if region.shape[0] >= size and region.shape[1] >= size else cv2.INTER_LINEAR
    image = cv2.resize(region, (size, size), interpolation=interpolation)
    box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
    return BrickCrop(image=image, region=region, box=box, found=found, brick_fraction=round(fraction, 4))
