"""Linear regression for trend forecasting."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearTrendModel:
    def __init__(self):
        self._model: Optional[LinearRegression] = None
        self.history: Optional[pd.Series] = None

    def fit(self, series: pd.Series):
        if len(series) < 2:
            raise ValueError("series must have at least 2 observations")
        self.history = series.copy()
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        self._model = LinearRegression().fit(X, y)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None or self.history is None:
            raise RuntimeError("fit() must be called before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        n = len(self.history)
        X_future = np.arange(n, n + horizon).reshape(-1, 1)
        values = self._model.predict(X_future)
        values = np.maximum(values, 0.0)

        last_date = self.history.index[-1]
        index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.Series(values, index=index, name="forecast")
