"""Pick a model based on simple data characteristics."""

from typing import Union

import numpy as np
import pandas as pd

from .linear_trend import LinearTrendModel
from .moving_average import MovingAverageModel

Model = Union[MovingAverageModel, LinearTrendModel]


def detect_trend_strength(series: pd.Series) -> float:
    """Return the absolute Pearson correlation between value and time index.

    Values close to 0 mean no linear trend; values close to 1 mean strong trend.
    """
    if len(series) < 3:
        return 0.0
    y = series.values.astype(float)
    if np.std(y) == 0:
        return 0.0
    t = np.arange(len(y), dtype=float)
    corr = np.corrcoef(t, y)[0, 1]
    return float(abs(corr)) if not np.isnan(corr) else 0.0


def select_model(series: pd.Series, trend_threshold: float = 0.7) -> Model:
    """Select Linear when the series has a strong trend, otherwise Moving Average."""
    strength = detect_trend_strength(series)
    if strength >= trend_threshold:
        return LinearTrendModel()
    return MovingAverageModel(window=30)
