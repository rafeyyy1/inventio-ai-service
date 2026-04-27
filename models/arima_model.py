"""
ARIMA Model — Inventio AI Service
Model advanced untuk forecasting stok.

ARIMA lebih kompleks dari MA:
- Bisa menangkap autokorelasi (pola berulang)
- Bisa menangkap trend dan differencing
- Params: p (AR), d (diff), q (MA)
- Lebih lambat dari MA tapi bisa lebih akurat untuk pola kompleks
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

# statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.

    Args:
        order: tuple (p, d, q) — default (5,1,2) yang umum bagus
        name: nama model untuk logging
    """

    def __init__(self, order: tuple = (5, 1, 2), name: str = "arima"):
        self.order = order
        self.name = name
        self.model: Optional[ARIMA] = None
        self.result: Optional[ARIMAResultsWrapper] = None
        self.history: Optional[pd.Series] = None

    def _check_stationarity(self, data: pd.Series) -> bool:
        """
        Uji stasionaritas dengan ADF test.
        Jika p-value < 0.05 → data stasioner.
        """
        try:
            result = adfuller(data.dropna())
            return result[1] < 0.05
        except Exception:
            return False

    def fit(self, data: pd.Series) -> "ARIMAModel":
        """
        Fit ARIMA model ke data.

        Args:
            data: Series time series (index = datetime, values = sales)
        """
        self.history = data.copy()

        # Jika data pendek, gunakan parameter sederhana
        if len(data) < 50:
            order = (2, 1, 1)
        else:
            order = self.order

        # Fit ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ARIMA(data.values, order=order)
            self.result = self.model.fit()

        return self

    def predict(self, horizon: int = 7) -> pd.Series:
        """
        Forecast horizon hari ke depan.

        Args:
            horizon: jumlah hari yang di-prediksi

        Returns:
            Series dengan index tanggal dan nilai forecast
        """
        if self.result is None:
            raise ValueError("Model belum di-fit. Panggil fit() dulu.")

        # Forecast
        forecast = self.result.forecast(steps=horizon)

        # Buat tanggal forecast
        last_date = self.history.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        # Pastikan forecast tidak negatif (stok tidak bisa negatif)
        forecast = np.maximum(forecast, 0)

        return pd.Series(data=forecast, index=forecast_dates, name="forecast")

    def predict_in_sample(self) -> np.ndarray:
        """Get in-sample fitted values (untuk evaluasi)."""
        if self.result is None:
            raise ValueError("Model belum di-fit.")
        return self.result.fittedvalues

    def get_aic(self) -> float:
        """Get AIC score — lower is better (untuk model selection)."""
        if self.result is None:
            raise ValueError("Model belum di-fit.")
        return float(self.result.aic)

    def summary(self) -> str:
        """Return model summary sebagai string."""
        if self.result is None:
            return "Model belum di-fit."
        return str(self.result.summary())

    def get_params(self) -> dict:
        """Return parameter model."""
        return {
            "order": self.order,
            "name": self.name,
            "aic": self.get_aic() if self.result else None,
        }


# Type alias untuk hasil ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMAResults
    ARIMAResultsWrapper = ARIMAResults
except ImportError:
    ARIMAResultsWrapper = object
