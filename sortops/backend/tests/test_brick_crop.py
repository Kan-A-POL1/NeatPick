"""The shared brick crop: finds the brick, ignores the gripper and plate hues, stays inside the frame, falls back."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sortops.config import DEFAULTS, Settings, validate
from sortops.pipeline.brick_crop import crop_brick

INSPECTION = DEFAULTS["inspection"]
BLUE, RED, ORANGE = (30, 110, 220), (210, 40, 50), (240, 120, 30)


def _scene(brick: tuple[int, int, int, int] | None = (300, 220, 90, 70), color: tuple[int, int, int] = BLUE) -> np.ndarray:
    """Grey frame with a big red gripper finger and an orange plate; optionally a brick (x, y, w, h)."""
    image = np.full((480, 640, 3), 120, np.uint8)
    cv2.rectangle(image, (40, 40), (260, 420), RED, -1)  # the finger: larger than the brick
    cv2.ellipse(image, (520, 380), (110, 30), 0, 0, 360, ORANGE, -1)
    if brick is not None:
        x, y, w, h = brick
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return image


def test_finds_the_brick_not_the_bigger_gripper() -> None:
    crop = crop_brick(_scene(), INSPECTION)
    assert crop.found
    assert crop.image.shape == (256, 256, 3)
    x, y, w, h = crop.box
    # Square around the brick centre (345, 255) with the margin, in a 640x480 frame.
    assert abs((x + w / 2) * 640 - 345) <= 2 and abs((y + h / 2) * 480 - 255) <= 2
    assert w * 640 == pytest.approx(90 * 1.2, abs=2) and h * 480 == pytest.approx(90 * 1.2, abs=2)
    # The crop is mostly brick: no red finger inside.
    hsv = cv2.cvtColor(crop.image, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] > 160) | (hsv[..., 0] < 8)) & (hsv[..., 1] > 90)
    assert red.mean() < 0.01


def test_no_brick_falls_back_to_the_search_zone() -> None:
    crop = crop_brick(_scene(brick=None), INSPECTION)
    assert not crop.found
    assert crop.brick_fraction == 0.0
    assert crop.box == (0.0, 0.0, 1.0, 1.0)
    assert crop.image.shape == (256, 256, 3)


def test_red_brick_is_not_taken_for_a_brick() -> None:
    assert not crop_brick(_scene(color=RED), INSPECTION).found


def test_brick_at_the_edge_gives_a_full_square_inside_the_frame() -> None:
    crop = crop_brick(_scene(brick=(560, 400, 80, 80)), INSPECTION)
    x, y, w, h = crop.box
    assert crop.found
    assert x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9
    assert w * 640 == pytest.approx(h * 480, abs=1)


def test_search_zone_excludes_a_brick_outside_it() -> None:
    cfg = {**INSPECTION, "roi": [0.7, 0.0, 0.3, 0.4]}
    assert not crop_brick(_scene(), cfg).found
    cfg = {**INSPECTION, "roi": [0.4, 0.3, 0.4, 0.4]}
    assert crop_brick(_scene(), cfg).found


def test_fixed_mode_crops_the_zone_itself() -> None:
    cfg = {**INSPECTION, "crop_mode": "fixed", "roi": [0.25, 0.25, 0.5, 0.5]}
    crop = crop_brick(_scene(), cfg)
    assert crop.box == (0.25, 0.25, 0.5, 0.5)
    assert crop.found and crop.brick_fraction > 0.05


def test_degenerate_zone_means_the_full_frame() -> None:
    cfg = {**INSPECTION, "roi": [0.5, 0.5, 0.0, 0.0]}
    assert crop_brick(_scene(), cfg).found


def test_specks_below_min_area_are_not_a_brick() -> None:
    assert not crop_brick(_scene(brick=(300, 220, 8, 8)), INSPECTION).found


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("crop_mode", "smart"),
        ("crop_size", 8),
        ("crop_size", 64.5),
        ("crop_margin", 2.0),
        ("crop_exclude_hues", [[30.0, 10.0]]),
        ("crop_exclude_hues", [[0.0, 200.0]]),
        ("crop_exclude_hues", [5.0]),
    ],
)
def test_bad_crop_settings_are_refused(key: str, value: object) -> None:
    values = {section: dict(content) for section, content in DEFAULTS.items()}
    values["inspection"][key] = value
    with pytest.raises(ValueError, match=key if key != "crop_exclude_hues" else "crop_exclude_hues"):
        validate(values)


def test_defaults_are_valid() -> None:
    validate(DEFAULTS)


def test_invalid_persisted_crop_settings_refuse_to_load(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"inspection": {"crop_mode": "smart"}}))
    with pytest.raises(ValueError, match="crop_mode"):
        Settings(tmp_path)


def test_older_config_without_crop_keys_loads_with_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"inspection": {"roi": [0.1, 0.1, 0.8, 0.8]}}))
    inspection = Settings(tmp_path).section("inspection")
    assert inspection["crop_mode"] == "auto" and inspection["roi"] == [0.1, 0.1, 0.8, 0.8]


def test_small_far_brick_is_found() -> None:
    # A 1x2 brick seen from the front camera: ~0.25 % of the frame (was under the old 0.4 % minimum).
    crop = crop_brick(_scene(brick=(400, 300, 36, 22)), INSPECTION)
    assert crop.found
    x, y, w, h = crop.box
    assert abs((x + w / 2) * 640 - 418) <= 2 and abs((y + h / 2) * 480 - 311) <= 2


def test_a_separate_coloured_object_far_away_is_not_merged() -> None:
    image = _scene(brick=(300, 220, 90, 70))
    cv2.rectangle(image, (560, 20), (620, 60), BLUE, -1)  # another blue thing, far from the brick
    x, y, w, h = crop_brick(image, INSPECTION).box
    assert (x + w) * 640 < 560


def test_a_differently_coloured_object_next_to_the_brick_is_not_merged() -> None:
    # The yellow plate right under a blue brick (wrist camera): the crop stays on the brick, as the capture app's did.
    image = _scene(brick=(300, 200, 90, 70))
    cv2.ellipse(image, (345, 300), (60, 22), 0, 0, 360, (230, 200, 40), -1)  # smaller than the brick, touching it
    x, y, w, h = crop_brick(image, INSPECTION).box
    assert abs((x + w / 2) * 640 - 345) <= 2 and abs((y + h / 2) * 480 - 235) <= 2
    assert w * 640 == pytest.approx(90 * 1.2, abs=2)
