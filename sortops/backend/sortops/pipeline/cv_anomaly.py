"""Model-free brick inspection: auto-crop the brick by colour, then flag stains and dark / high-contrast marks.

The brick is the largest saturated blob that is neither the red gripper nor the orange bowl. Inside it (eroded, so
its outline does not count), a pixel is a *stain* when its chroma is far from the brick's median colour (another
colour on the brick: marker, dirt, sticker) and a *mark* when it is much darker than the brick (black spot, deep
scratch). Highlights (bright, unsaturated) are ignored. Score = max(stain share / stain_max, mark share / mark_max)
× 0.5, so 0.5 is the threshold, like an Anomalib score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

CV_DEFAULTS: dict[str, float] = {
    "cv_min_saturation": 90.0,  # HSV S (0-255) of brick pixels
    "cv_min_value": 50.0,
    "cv_exclude_red_orange": 1.0,  # 1: drop hues of the red gripper and the orange bowl
    "cv_erode_px": 6.0,
    "cv_stain_chroma": 28.0,  # LAB a/b distance to the brick median counting as another colour
    "cv_mark_darker": 60.0,  # LAB L this much under the brick median, and under cv_mark_dark, is a dark mark
    "cv_mark_dark": 15.0,  # (0-255 LAB L): black spots, not the studs' blue shadows
    "cv_hue_band": 14.0,  # OpenCV hue units around the brick's median hue
    "cv_stain_max": 0.01,  # stain share of the brick at the threshold (score 0.5)
    "cv_mark_max": 0.01,
    "cv_crop_size": 224.0,
    "cv_min_area": 0.01,  # brick blob share of the frame under which no brick is found
}


@dataclass
class CvResult:
    found: bool
    score: float
    stain: float
    mark: float
    box: tuple[float, float, float, float]  # normalized x, y, w, h of the tight crop
    crop: np.ndarray  # RGB, square, crop_size
    heat: np.ndarray  # float32 0..1, same size as crop: 1 on stain / mark pixels


def brick_blob(image_rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    """uint8 mask (0/255) of the brick: the largest saturated blob that is not red/orange; None if too small."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s >= cfg["cv_min_saturation"]) & (v >= cfg["cv_min_value"])
    if cfg["cv_exclude_red_orange"]:
        mask &= ~((h <= 22) | (h >= 160))  # OpenCV hue 0-180: red and orange
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blob = _largest(mask, cfg)
    if blob is None:
        return None
    # Keep the brick's own hue only: a saturated background touching it (wood, cable) would join the blob.
    hue = float(np.median(h[blob > 0]))
    near = (np.abs(h.astype(np.float32) - hue) <= cfg["cv_hue_band"]) & (s >= cfg["cv_min_saturation"])
    # Opening cuts the thin bridges to a bluish background (tape, cables) before taking the brick component.
    near = cv2.morphologyEx(near.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    blob = _largest(near, cfg)
    if blob is None:
        return None
    # Close the studs' shadows and the specular holes, without filling a concave outline with background.
    return cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _largest(mask: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    """The largest brick-shaped component: big enough, roughly square (a brick, not a wood streak) and well filled."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for index in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        x, y, w, h, area = stats[index]
        if area < cfg["cv_min_area"] * mask.size:
            return None
        if 0.5 <= w / max(h, 1) <= 2.0 and area >= 0.45 * w * h:
            return (labels == index).astype(np.uint8) * 255
    return None


def inspect(image_rgb: np.ndarray, cfg: dict[str, Any]) -> CvResult:
    size = int(cfg["cv_crop_size"])
    height, width = image_rgb.shape[:2]
    blob = brick_blob(image_rgb, cfg)
    if blob is None:
        empty = cv2.resize(image_rgb, (size, size))
        return CvResult(False, 0.0, 0.0, 0.0, (0.0, 0.0, 1.0, 1.0), empty, np.zeros((size, size), np.float32))
    x, y, w, h = cv2.boundingRect(blob)
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(width, x0 + side), min(height, y0 + side)
    inner = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=int(cfg["cv_erode_px"])) > 0
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    highlight = (hsv[..., 2] > 225) & (hsv[..., 1] < 90)
    region = inner & ~highlight
    stain_px = np.zeros(blob.shape, bool)
    mark_px = np.zeros(blob.shape, bool)
    stain = mark = 0.0
    if region.sum() > 50:
        median = np.median(lab[region], axis=0)
        chroma = np.hypot(lab[..., 1] - median[1], lab[..., 2] - median[2])
        stain_px = inner & ~highlight & (chroma > cfg["cv_stain_chroma"])
        mark_px = inner & (lab[..., 0] < median[0] - cfg["cv_mark_darker"]) & (lab[..., 0] < cfg["cv_mark_dark"])
        total = float(inner.sum())
        stain, mark = float(stain_px.sum()) / total, float(mark_px.sum()) / total
    score = 0.5 * max(stain / cfg["cv_stain_max"], mark / cfg["cv_mark_max"])
    crop = cv2.resize(image_rgb[y0:y1, x0:x1], (size, size))
    heat = cv2.resize((stain_px | mark_px)[y0:y1, x0:x1].astype(np.float32), (size, size))
    box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
    return CvResult(True, float(min(score, 1.0)), stain, mark, box, crop, heat)
