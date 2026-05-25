"""Basic tests for the forecasting models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import LinearTrendModel, MovingAverageModel, select_model


@pytest.fixture
def flat_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    values = [100 + np.sin(i) for i in range(60)]
    return pd.Series(values, index=idx)


@pytest.fixture
def trending_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.arange(60) * 2.0 + 10
    return pd.Series(values, index=idx)


def test_moving_average_forecast(flat_series: pd.Series) -> None:
    model = MovingAverageModel(window=14).fit(flat_series)
    forecast = model.predict(horizon=7)
    assert len(forecast) == 7
    assert all(f > 0 for f in forecast)


def test_linear_trend_forecast(trending_series: pd.Series) -> None:
    model = LinearTrendModel().fit(trending_series)
    forecast = model.predict(horizon=5)
    assert len(forecast) == 5
    assert forecast.iloc[0] > trending_series.iloc[-1]


def test_select_model(flat_series: pd.Series, trending_series: pd.Series) -> None:
    assert isinstance(select_model(flat_series), MovingAverageModel)
    assert isinstance(select_model(trending_series), LinearTrendModel)
