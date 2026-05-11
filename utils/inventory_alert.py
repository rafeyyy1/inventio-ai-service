"""Turn a forecast into a simple restock recommendation.

The recommendation compares the recent demand (last 7 days of history) against
the upcoming demand (first 7 days of forecast). If demand is rising, the user
should restock; if falling, the user has overstock.
"""

from typing import Literal, Optional

import pandas as pd

Trend = Literal["up", "stable", "down"]
Status = Literal["restock", "ok", "overstock"]


def _trend_from_change(pct_change: float, threshold: float = 5.0) -> Trend:
    if pct_change > threshold:
        return "up"
    if pct_change < -threshold:
        return "down"
    return "stable"


def build_alert(
    history: pd.Series,
    forecast: pd.Series,
    current_stock: Optional[float] = None,
    window: int = 7,
) -> dict:
    """Build an inventory alert from history + forecast.

    Args:
        history: past observed demand (daily).
        forecast: predicted demand (daily).
        current_stock: current stock level. If None, days_until_stockout will be null.
        window: how many days to average for the comparison (default 7).
    """
    recent_avg = float(history.iloc[-window:].mean()) if len(history) else 0.0
    future_avg = float(forecast.iloc[:window].mean()) if len(forecast) else 0.0

    if recent_avg == 0:
        pct_change = 0.0 if future_avg == 0 else 100.0
    else:
        pct_change = (future_avg - recent_avg) / recent_avg * 100

    trend = _trend_from_change(pct_change)
    status: Status = {"up": "restock", "down": "overstock", "stable": "ok"}[trend]

    days_until_stockout = None
    if current_stock is not None and future_avg > 0:
        days_until_stockout = int(current_stock / future_avg)

    return {
        "status": status,
        "trend": trend,
        "pct_change": round(pct_change, 2),
        "recent_avg": round(recent_avg, 2),
        "future_avg": round(future_avg, 2),
        "current_stock": current_stock,
        "days_until_stockout": days_until_stockout,
    }
