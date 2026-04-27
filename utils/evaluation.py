"""
Evaluation Metrics — Inventio AI Service
Menghitung error untuk evaluasi model.
"""

import numpy as np
import pandas as pd
from typing import Union


def mae(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Absolute Error (MAE).
    Rata-rata selisih absolut antara prediksi dan actual.

    MAE dipilih karena:
    - Mudah diinterpretasi (satuan sama dengan data asli)
    - Tidak sensitif terhadap outlier seperti MSE
    - Dalam konteks stok, MAE = "rata-rata kesalahan unit"
      → "Prediksi meleset ~19 unit dari actual"
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Error dalam bentuk persentase.

    - 18% MAPE = prediksi rata-rata meleset 18% dari nilai actual
    - Mudah dipahami stakeholder (dalam %)
    - Problem: undefined jika ada 0 di y_true
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Hindari division by zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> dict:
    """
    Evaluasi lengkap: MAE + MAPE.
    """
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 2),
    }


def compare_models(results: dict) -> dict:
    """
    Bandingkan beberapa model berdasarkan MAE.
    Returns dict dengan ranking dan model terbaik.
    """
    sorted_models = sorted(results.items(), key=lambda x: x[1]["mae"])
    best_model = sorted_models[0][0]

    comparison = {
        "rankings": [
            {"rank": i + 1, "model": name, "mae": res["mae"], "mape": res["mape"]}
            for i, (name, res) in enumerate(sorted_models)
        ],
        "best_model": best_model,
        "best_mae": results[best_model]["mae"],
    }
    return comparison
