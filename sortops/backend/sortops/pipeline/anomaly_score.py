"""One anomaly score per brick crop from the model's raw anomaly map: a localized peak and a diffuse mean, combined.

Our defects are small and local (a marker dot, a chip), so a localized peak must weigh
more than a slightly-off whole brick, without ignoring the latter. Two terms:

- peak: the model's raw image score (PatchCore: its most anomalous patch, before the export blurs the map with a
  33 px Gaussian); without it, the mean of the hottest ``score_peak_fraction`` of the raw map (blurred, so weaker);
- mean: mean of the whole raw map: diffuse anomalies.

On the wrist PatchCore export the top 1 % of the blurred map moved half as much as the raw score on a tape patch,
hence the raw score as the peak.

Each is put in "good brick" units, ``(x - median_good) / (p95_good - median_good)`` (0 = a typical good brick,
1 = as unusual as the 95th-percentile good brick), from the calibration on known-good crops, then combined
quadratically: ``sqrt(w * peak+^2 + (1 - w) * mean+^2)`` with ``w = score_peak_weight`` (0.8). The square favours
whichever term stands out most (in practice the localized peak) while a diffuse anomaly still raises the score.

SortOps (``models.runtime.OpenVinoAnomaly``) and ``anomaly/quick_eval.py`` both use this module: never re-implement.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCORE_DEFAULTS: dict[str, Any] = {
    # Weight of the localized peak against the whole-map mean in the quadratic combination (team: localized first).
    "score_peak_weight": 0.8,
    # Share of the hottest anomaly-map pixels averaged into the peak term when the model gives no raw image score
    # (1 % of a 256x256 crop = 655 px).
    "score_peak_fraction": 0.01,
}


def validate_score(cfg: dict[str, Any]) -> None:
    if not 0.0 <= cfg["score_peak_weight"] <= 1.0:
        raise ValueError("inspection.score_peak_weight must be within [0, 1]")
    if not 0.0 < cfg["score_peak_fraction"] <= 0.5:
        raise ValueError("inspection.score_peak_fraction must be within ]0, 0.5]")


def map_terms(raw_map: np.ndarray, peak_fraction: float, raw_score: float | None = None) -> tuple[float, float]:
    """(peak, mean) of one crop: peak = ``raw_score`` when the model gives one, else the mean of the map's hottest
    ``peak_fraction`` pixels; mean = mean of the whole raw map."""
    values = np.asarray(raw_map, np.float64).ravel()
    if values.size == 0:
        raise ValueError("empty anomaly map")
    if raw_score is not None:
        return float(raw_score), float(values.mean())
    k = max(1, int(math.ceil(values.size * peak_fraction)))
    peak = float(np.partition(values, values.size - k)[values.size - k :].mean())
    return peak, float(values.mean())


def term_references(peaks: list[float], means: list[float]) -> dict[str, float]:
    """Median and 95th percentile of each term over known-good crops (the unit of the combination)."""
    ref: dict[str, float] = {}
    for name, values in (("peak", peaks), ("mean", means)):
        array = np.asarray(values, np.float64)
        median = float(np.median(array))
        high = float(np.percentile(array, 95))
        ref[f"{name}_median"] = median
        # A flat good distribution must not make every tiny deviation huge.
        ref[f"{name}_spread"] = max(high - median, 0.05 * abs(median), 1e-6)
    return ref


def combined(peak: float, mean: float, ref: dict[str, float], peak_weight: float) -> float:
    """Quadratic combination of the two terms in good-brick units (0 for anything at or below a typical good brick)."""
    excess_peak = max(0.0, (peak - ref["peak_median"]) / ref["peak_spread"])
    excess_mean = max(0.0, (mean - ref["mean_median"]) / ref["mean_spread"])
    return math.sqrt(peak_weight * excess_peak**2 + (1.0 - peak_weight) * excess_mean**2)
