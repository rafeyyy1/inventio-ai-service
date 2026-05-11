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


def _aggregate_to_period(daily_forecast: pd.Series, period: PeriodType, horizon: int) -> List[tuple[datetime, float]]:
    """Convert a daily forecast series into the requested period grain.

    For DAILY: return the first `horizon` daily values.
    For WEEKLY/MONTHLY: sum daily values into buckets then take the first `horizon` buckets.
    """
    if period == PeriodType.DAILY:
        rows = list(daily_forecast.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    rule = "W" if period == PeriodType.WEEKLY else "MS"
    bucketed = daily_forecast.resample(rule).sum()
    rows = list(bucketed.items())[:horizon]
    return [(idx.to_pydatetime(), float(val)) for idx, val in rows]


def _periods_needed(period: PeriodType, horizon: int) -> int:
    """How many daily steps to forecast to cover `horizon` periods."""
    if period == PeriodType.DAILY:
        return horizon
    if period == PeriodType.WEEKLY:
        return horizon * 7 + 7  # extra buffer for week alignment
    return horizon * 31 + 31  # extra buffer for month alignment


def _format_date(dt: datetime, period: PeriodType) -> str:
    if period == PeriodType.MONTHLY:
        return dt.strftime("%Y-%m-01Z")
    if period == PeriodType.WEEKLY:
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}Z"
    return dt.strftime("%Y-%m-%dZ")


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

        rows = _aggregate_to_period(daily_forecast, params.period, params.horizon)
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


@app.get("/health")
def health() -> dict:
    """Liveness probe used by container orchestrators."""
    return {"status": "ok", "service": "inventio-ai-service"}


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
