"""Inventio AI forecasting API."""

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
    description="Forecasting service for inventory management.",
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


def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _to_series(item):
    points = sorted(item.historicalData, key=lambda p: p.date)
    index = pd.DatetimeIndex([_parse_iso(p.date) for p in points])
    values = [p.value for p in points]
    series = pd.Series(values, index=index, name="value", dtype=float)
    series.index = series.index.tz_convert(None) if series.index.tz is not None else series.index
    series = series.groupby(level=0).sum()
    return series.asfreq("D").ffill()


def _aggregate_daily_to_period(daily_forecast, period, horizon, historical_day=None, last_historical_date=None):
    if period == PeriodType.DAILY:
        rows = list(daily_forecast.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    if period == PeriodType.WEEKLY:
        weekly = daily_forecast.resample("W").mean()
        rows = list(weekly.items())[:horizon]
        return [(idx.to_pydatetime(), float(val)) for idx, val in rows]

    # For MONTHLY: aggregate by month, but adjust day to match historical pattern
    monthly = daily_forecast.resample("MS").mean()
    
    # Filter out any rows that fall on or before the last historical date (BEFORE slicing)
    if last_historical_date is not None:
        filtered_rows = [(idx, val) for idx, val in monthly.items() if idx > last_historical_date]
    else:
        filtered_rows = list(monthly.items())
    
    # Now take only horizon items from filtered rows
    rows = filtered_rows[:horizon]
    
    # If historical data had a specific day (e.g., 15th), adjust forecast dates
    if historical_day is not None and historical_day > 1:
        adjusted_rows = []
        for idx, val in rows:
            # Get the first day of month, then add (historical_day - 1) days
            adjusted_date = idx.replace(day=1) + pd.Timedelta(days=historical_day - 1)
            adjusted_rows.append((adjusted_date.to_pydatetime(), float(val)))
        return adjusted_rows
    
    return [(idx.to_pydatetime(), float(val)) for idx, val in rows]


def _periods_needed(period, horizon):
    if period == PeriodType.DAILY:
        return horizon
    if period == PeriodType.WEEKLY:
        return horizon * 7 + 7
    return horizon * 30 + 30


def _format_date(dt, period):
    if period == PeriodType.MONTHLY:
        # Keep the day as-is from the datetime object (don't force to 01)
        return dt.strftime("%Y-%m-%dZ")
    if period == PeriodType.WEEKLY:
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}Z"
    return dt.strftime("%Y-%m-%dZ")


def _unit_for(forecast_type):
    return "units" if forecast_type == ForecastType.STOCK_DEMAND else "IDR"


def _forecast_one(item, forecast_type, params):
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
        
        # Extract day-of-month and last date from last historical date (for date alignment)
        historical_day = daily_series.index[-1].day
        last_historical_date = daily_series.index[-1]
        
        # Aggregate to requested period
        rows = _aggregate_daily_to_period(daily_forecast, params.period, params.horizon, historical_day, last_historical_date)
        
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
