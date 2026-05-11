"""Basic tests for the forecasting models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import LinearTrendModel, MovingAverageModel, detect_trend_strength, select_model


@pytest.fixture
def flat_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    values = 100 + rng.normal(0, 5, size=60)
    return pd.Series(values, index=idx)


@pytest.fixture
def trending_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.arange(60) * 2.0 + 10
    return pd.Series(values, index=idx)


def test_moving_average_constant_forecast(flat_series: pd.Series) -> None:
    model = MovingAverageModel(window=14).fit(flat_series)
    forecast = model.predict(horizon=7)
    assert len(forecast) == 7
    assert forecast.nunique() == 1
    expected = flat_series.iloc[-14:].mean()
    assert forecast.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_moving_average_clamps_negative_forecast() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series([-5] * 10, index=idx)
    forecast = MovingAverageModel(window=5).fit(series).predict(horizon=3)
    assert (forecast >= 0).all()


def test_linear_trend_follows_trend(trending_series: pd.Series) -> None:
    model = LinearTrendModel().fit(trending_series)
    forecast = model.predict(horizon=5)
    # First future value should be slightly above the last observed value.
    assert forecast.iloc[0] > trending_series.iloc[-1]
    # Values should keep increasing.
    diffs = np.diff(forecast.values)
    assert (diffs > 0).all()


def test_detect_trend_strength(flat_series: pd.Series, trending_series: pd.Series) -> None:
    assert detect_trend_strength(flat_series) < 0.3
    assert detect_trend_strength(trending_series) > 0.95


def test_select_model(flat_series: pd.Series, trending_series: pd.Series) -> None:
    assert isinstance(select_model(flat_series), MovingAverageModel)
    assert isinstance(select_model(trending_series), LinearTrendModel)


def test_fit_requires_two_points() -> None:
    one_point = pd.Series([5.0], index=pd.DatetimeIndex(["2024-01-01"]))
    with pytest.raises(ValueError):
        MovingAverageModel().fit(one_point)
    with pytest.raises(ValueError):
        LinearTrendModel().fit(one_point)


def test_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        MovingAverageModel().predict(horizon=3)
    with pytest.raises(RuntimeError):
        LinearTrendModel().predict(horizon=3)
