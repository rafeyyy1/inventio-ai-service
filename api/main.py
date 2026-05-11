"""FastAPI application for the Inventio forecasting service.

Single endpoint: POST /api/predict (batch forecasting per item).
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ForecastDataPoint,
    ForecastParameters,
    ForecastRequest,
    ForecastResponse,
    ForecastType,
    ItemForecastRequest,
    ItemForecastResult,
    PeriodType,
)
from models import select_model

app = FastAPI(
    title="Inventio AI Service",
    description="Batch forecasting service for the Inventio inventory system.",
    version="1.0.0",
)

# CORS: read allowed origins from env, fall back to localhost for dev.
_allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 string ending in Z or with offset."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _to_series(item: ItemForecastRequest) -> pd.Series:
    """Convert request historical data to a daily Series."""
    points = sorted(item.historicalData, key=lambda p: p.date)
    index = pd.DatetimeIndex([_parse_iso(p.date) for p in points])
    values = [p.value for p in points]
    series = pd.Series(values, index=index, name="value", dtype=float)
    # Drop tzinfo so asfreq works predictably, then make daily.
    series.index = series.index.tz_convert(None) if series.index.tz is not None else series.index
    return series.asfreq("D").ffill()


def _aggregate_to_period(daily_forecast: pd.Series, period: PeriodType, horizon: int, historical_length: int = None) -> List[tuple[datetime, float]]:
    """Convert a daily forecast series into the requested period grain.

    For DAILY: return the first `horizon` daily values.
    For MONTHLY: aggregate using variable bucket sizes:
      - Period 1: 1 day
      - Period 2: historical_length days (to match historical window)
      - Period 3+: 30 days each
    Returns list of (date, value) tuples where date represents the period month.
    """
    if period == PeriodType.DAILY:
        rows = list(daily_forecast.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    if period == PeriodType.WEEKLY:
        bucketed = daily_forecast.resample("W").sum()
        rows = list(bucketed.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]
    
    # MONTHLY: custom aggregation with variable bucket sizes
    # Each bucket gets assigned to a calendar month for dating purposes
    result = []
    current_idx = 0
    
    # Start from the next month after historical data ends
    forecast_start = daily_forecast.index[0]
    # Move to next month boundary
    period_date = forecast_start.replace(day=1)
    if period_date <= forecast_start:
        # Add one month
        if period_date.month == 12:
            period_date = period_date.replace(year=period_date.year + 1, month=1)
        else:
            period_date = period_date.replace(month=period_date.month + 1)
    
    for period_num in range(horizon):
        if period_num == 0:
            bucket_size = 1
        elif period_num == 1 and historical_length:
            bucket_size = historical_length
        else:
            bucket_size = 30
        
        # Extract bucket_size days from forecast
        if current_idx < len(daily_forecast):
            bucket = daily_forecast.iloc[current_idx:current_idx + bucket_size]
            if len(bucket) > 0:
                period_value = bucket.sum()
                result.append((period_date.to_pydatetime(), float(period_value)))
                
                # Move to next month for next iteration
                if period_date.month == 12:
                    period_date = period_date.replace(year=period_date.year + 1, month=1)
                else:
                    period_date = period_date.replace(month=period_date.month + 1)
                
                current_idx += bucket_size
    
    return result


def _format_date(dt: datetime, period: PeriodType) -> str:
    if period == PeriodType.MONTHLY:
        return dt.strftime("%Y-%m-01Z")
    if period == PeriodType.WEEKLY:
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}Z"
    return dt.strftime("%Y-%m-%dZ")


def _periods_needed(period: PeriodType, horizon: int) -> int:
    """How many daily steps to forecast to cover `horizon` periods."""
    if period == PeriodType.DAILY:
        return horizon
    if period == PeriodType.WEEKLY:
        return horizon * 7 + 7  # extra buffer for week alignment
    return horizon * 31 + 31  # extra buffer for month alignment


def _unit_for(forecast_type: ForecastType) -> str:
    return "units" if forecast_type == ForecastType.STOCK_DEMAND else "IDR"


def _forecast_one(
    item: ItemForecastRequest,
    forecast_type: ForecastType,
    params: ForecastParameters,
) -> ItemForecastResult:
    try:
        series = _to_series(item)
        if len(series) < 2:
            return ItemForecastResult(
                itemId=item.itemId,
                status="error",
                error="At least 2 historical data points are required",
            )

        model = select_model(series)
        model.fit(series)
        daily_horizon = _periods_needed(params.period, params.horizon)
        daily_forecast = model.predict(horizon=daily_horizon)

        # Pass historical length for proper monthly aggregation
        historical_length = len(item.historicalData)
        rows = _aggregate_to_period(daily_forecast, params.period, params.horizon, historical_length)
        unit = _unit_for(forecast_type)
        points = [
            ForecastDataPoint(
                forecastDate=_format_date(dt, params.period),
                forecastValue=round(val, 2),
                unit=unit,
            )
            for dt, val in rows
        ]

        return ItemForecastResult(
            itemId=item.itemId,
            status="success",
            forecast=points,
            modelUsed=type(model).__name__,
        )
    except Exception as exc:  # noqa: BLE001 - we want to surface any failure per-item
        return ItemForecastResult(
            itemId=item.itemId,
            status="error",
            error=str(exc),
        )


@app.post("/api/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    """Batch forecast for one or more items.

    See `docs/api_contract.md` for the full request/response shape.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="At least one item is required")

    job_id = f"forecast-job-{uuid.uuid4().hex[:8]}"
    results = [
        _forecast_one(item, request.forecastType, request.forecastParameters)
        for item in request.items
    ]
    return ForecastResponse(jobId=job_id, results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
