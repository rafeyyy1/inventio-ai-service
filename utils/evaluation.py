"""Forecast accuracy metrics."""

from typing import Iterable

import numpy as np


def _arr(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y, yhat = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y, yhat = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Weighted Absolute Percentage Error, in percent.

    WAPE = sum(|y - yhat|) / sum(|y|) * 100. More stable than MAPE when some
    y values are zero or near zero.
    """
    y, yhat = _arr(y_true), _arr(y_pred)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - yhat)) / denom * 100)


def evaluate(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    return {
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
        "wape": round(wape(y_true, y_pred), 2),
    }
