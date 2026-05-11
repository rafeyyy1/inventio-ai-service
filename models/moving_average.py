"""Moving Average forecasting model."""

from typing import Optional

import numpy as np
import pandas as pd


class MovingAverageModel:
    """Forecast as the mean of the last `window` observations.

    The forecast is a constant value (the trailing mean), repeated for the
    requested horizon. Suitable for series without strong trend or seasonality.
    """

    def __init__(self, window: int = 30):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.history: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "MovingAverageModel":
        if len(series) < 2:
            raise ValueError("series must have at least 2 observations")
        self.history = series.copy()
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self.history is None:
            raise RuntimeError("fit() must be called before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        effective_window = min(self.window, len(self.history))
        value = float(self.history.iloc[-effective_window:].mean())
        value = max(value, 0.0)  # demand cannot be negative

        last_date = self.history.index[-1]
        index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.Series([value] * horizon, index=index, name="forecast")
