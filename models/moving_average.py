"""
Moving Average Model — Inventio AI Service
Model utama untuk forecasting stok.

Kenapa Moving Average:
- Simpel, cepat, ringan (bisa jalan di edge/low-resource)
- Cocok untuk data tanpa pola musiman yang kompleks
- Sudah teruji baik untuk dataset ini (MAE ~19, MAPE ~18%)
- Mudah di-interpretasi oleh stakeholder non-teknis
"""

import numpy as np
import pandas as pd
from typing import Optional


class MovingAverageModel:
    """
    Moving Average forecasting model.

    Args:
        window: jumlah hari untuk rata-rata bergerak (default=30)
        name: nama model untuk logging
    """

    def __init__(self, window: int = 30, name: str = "moving_average"):
        self.window = window
        self.name = name
        self.history: Optional[pd.Series] = None

    def fit(self, data: pd.Series) -> "MovingAverageModel":
        """
        Simpan history data untuk forecasting.
        Moving Average tidak perlu "fit" seperti ML biasa,
        cukup simpan data terakhir.
        """
        self.history = data.copy()
        return self

    def predict(self, horizon: int = 7) -> pd.Series:
        """
        Forecast horizon hari ke depan.
        Forecast = rata-rata dari window terakhir.

        Args:
            horizon: jumlah hari yang di-prediksi

        Returns:
            Series dengan index tanggal dan nilai forecast
        """
        if self.history is None:
            raise ValueError("Model belum di-fit. Panggil fit() dulu.")

        # Hitung MA dari data terakhir
        ma_value = self.history.iloc[-self.window:].mean()

        # Buat tanggal forecast
        last_date = self.history.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        # Return series dengan forecast
        forecast = pd.Series(
            data=[ma_value] * horizon,
            index=forecast_dates,
            name="forecast"
        )
        return forecast

    def predict_next(self) -> float:
        """Predict 1 hari ke depan (untuk real-time prediction)."""
        if self.history is None:
            raise ValueError("Model belum di-fit.")
        return float(self.history.iloc[-self.window:].mean())

    def get_params(self) -> dict:
        """Return parameter model."""
        return {"window": self.window, "name": self.name}

    def update(self, new_value: float, new_date: pd.Timestamp = None) -> None:
        """
        Update model dengan data baru (online learning).
        Berguna untuk real-time prediction.
        """
        if new_date is None:
            new_date = (self.history.index[-1] + pd.Timedelta(days=1)) if len(self.history) > 0 else pd.Timestamp.today()

        new_row = pd.Series(data=[new_value], index=[new_date])
        self.history = pd.concat([self.history, new_row])
