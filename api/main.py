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

_allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _to_series(item: ItemForecastRequest) -> pd.Series:
    points = sorted(item.historicalData, key=lambda p: p.date)
    index = pd.DatetimeIndex([_parse_iso(p.date) for p in points])
    values = [p.value for p in points]
    series = pd.Series(values, index=index, name="value", dtype=float)
    series.index = series.index.tz_convert(None) if series.index.tz is not None else series.index
    series = series.groupby(level=0).sum()
    return series.asfreq("D").ffill()


def _aggregate_daily_to_period(daily_forecast: pd.Series, period: PeriodType, horizon: int) -> List[tuple[datetime, float]]:
    """Aggregate daily forecast to monthly/weekly/daily based on period and horizon."""
    if period == PeriodType.DAILY:
        rows = list(daily_forecast.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    if period == PeriodType.WEEKLY:
        # Resample to weekly and take horizon weeks
        weekly = daily_forecast.resample("W").mean()
        rows = list(weekly.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    # MONTHLY
    # Resample to monthly and take horizon months
    monthly = daily_forecast.resample("MS").mean()
    rows = list(monthly.items())[:horizon]
    return [(idx.to_pydatetime(), float(val)) for idx, val in rows]


def _periods_needed(period: PeriodType, horizon: int) -> int:
    """Calculate how many daily forecast points are needed."""
    if period == PeriodType.DAILY:
        return horizon
    if period == PeriodType.WEEKLY:
        # ~7 days per week, add buffer
        return horizon * 7 + 7
    # MONTHLY
    # ~30 days per month, add buffer
    return horizon * 30 + 30


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
        daily_series = _to_series(item)
        if len(daily_series) < 2:
            return ItemForecastResult(
                itemId=item.itemId,
                status="error",
                error="At least 2 historical data points are required",
            )

        # Always work with daily series for model selection and fitting
        model = select_model(daily_series)
        model.fit(daily_series)
        
        # Calculate how many daily points to forecast
        daily_horizon = _periods_needed(params.period, params.horizon)
        
        # Get daily forecast
        daily_forecast = model.predict(horizon=daily_horizon)
        
        # Aggregate to requested period
        rows = _aggregate_daily_to_period(daily_forecast, params.period, params.horizon)
        
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
    except Exception as exc:
        return ItemForecastResult(
            itemId=item.itemId,
            status="error",
            error=str(exc),
        )


@app.get("/")
def root() -> dict:
    """Root endpoint - redirect info to /docs"""
    return {
        "service": "Inventio AI Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "inventio-ai-service"}


@app.post("/api/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
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
