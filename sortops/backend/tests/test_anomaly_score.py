"""Localized peak + whole-map mean, combined quadratically, on SortOps' calibrated 0..1 scale."""

from __future__ import annotations

import numpy as np
import pytest

from sortops.config import DEFAULTS, validate
from sortops.models.runtime import AnomalyOutput, calibration_for, score_output
from sortops.pipeline.anomaly_score import combined, map_terms, term_references

RNG = np.random.default_rng(0)
WEIGHT, FRACTION = DEFAULTS["inspection"]["score_peak_weight"], DEFAULTS["inspection"]["score_peak_fraction"]


def _good() -> np.ndarray:
    return RNG.normal(10.0, 1.0, (64, 64)).astype(np.float32)


def _local(height: float) -> np.ndarray:
    raw = _good()
    raw[30:33, 30:34] += height  # a marker dot: 12 px of 4096 (0.3 %)
    return raw


def _diffuse(offset: float) -> np.ndarray:
    return _good() + offset  # the whole brick slightly off (lighting, colour)


def _out(raw_map: np.ndarray | None, raw: float | None = None, exported: float = 0.5) -> AnomalyOutput:
    return AnomalyOutput(exported, raw, np.zeros((2, 2), np.float32), raw_map)


@pytest.fixture(scope="module")
def calibration() -> dict:
    return calibration_for([_out(_good()) for _ in range(60)], WEIGHT, FRACTION)


def test_map_terms_reads_the_hottest_share_and_the_mean() -> None:
    raw = np.zeros((10, 10))
    raw[0, 0] = 100.0
    peak, mean = map_terms(raw, 0.01)
    assert peak == 100.0 and mean == 1.0
    assert map_terms(raw, 0.02)[0] == 50.0


def test_good_bricks_score_low_and_a_localized_defect_crosses_the_line(calibration: dict) -> None:
    goods = [score_output(_out(_good()), calibration) for _ in range(40)]
    assert np.median(goods) < 0.3 and max(goods) < 0.5
    assert score_output(_out(_local(8.0)), calibration) >= 0.5


def test_localized_defects_weigh_more_than_diffuse_ones_without_eclipsing_them(calibration: dict) -> None:
    terms = calibration["terms"]
    local_peak, local_mean = map_terms(_local(8.0), FRACTION)
    # A diffuse drift with the same total excess over the map as the dot.
    drift = 8.0 * 12 / 4096
    diffuse_peak, diffuse_mean = map_terms(_diffuse(drift * 30), FRACTION)
    local = combined(local_peak, local_mean, terms, WEIGHT)
    diffuse = combined(diffuse_peak, diffuse_mean, terms, WEIGHT)
    assert local > diffuse > 1.0, (local, diffuse)  # the diffuse one still stands out of the good bricks (p95 = 1)
    # A peak-only score would ignore the mean term entirely; the combination does not.
    assert combined(terms["peak_median"], terms["mean_median"] + 3 * terms["mean_spread"], terms, WEIGHT) > 1.0


def test_term_references_never_divide_by_zero() -> None:
    ref = term_references([5.0] * 10, [2.0] * 10)
    assert ref["peak_spread"] > 0 and ref["mean_spread"] > 0


def test_older_calibrations_and_models_without_a_raw_map_keep_working(calibration: dict) -> None:
    legacy = calibration_for([_out(None, raw=float(v)) for v in RNG.normal(40, 2, 30)], WEIGHT, FRACTION)
    assert "terms" not in legacy
    assert score_output(_out(None, raw=80.0), legacy) == 1.0
    # Calibrated on maps, but this model gives none: the export's own score, never a mixed scale.
    assert score_output(_out(None, raw=80.0, exported=0.3), calibration) == 0.3
    assert score_output(_out(_good(), exported=0.3), None) == 0.3


@pytest.mark.parametrize(("key", "value"), [("score_peak_weight", 1.5), ("score_peak_fraction", 0.0)])
def test_bad_score_settings_are_refused(key: str, value: float) -> None:
    values = {section: dict(content) for section, content in DEFAULTS.items()}
    values["inspection"][key] = value
    with pytest.raises(ValueError, match=key):
        validate(values)


def test_peak_is_the_models_raw_score_when_it_gives_one() -> None:
    goods = [_out(_good(), raw=float(v)) for v in RNG.normal(70, 1, 40)]
    calibration = calibration_for(goods, WEIGHT, FRACTION)
    assert calibration["terms"]["peak_from_raw"] is True
    assert abs(calibration["terms"]["peak_median"] - 70) < 1
    # A high raw score (the model's most anomalous patch) on an otherwise normal map crosses the line.
    assert score_output(_out(_good(), raw=80.0), calibration) >= 0.5
    assert score_output(_out(_good(), raw=70.0), calibration) < 0.5
