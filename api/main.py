"""
FastAPI Main — Inventio AI Service
Batch Forecasting API untuk sistem Inventio — Inventory Management.
"""

import sys
import uuid
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Local imports
from models import MovingAverageModel
from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastParameters,
    ItemForecastRequest,
    ItemForecastResult,
    ForecastDataPoint,
    PeriodType,
    ForecastType,
)

# ==================== APP SETUP ====================
app = FastAPI(
    title="Inventio AI Service",
    description="Batch forecasting API untuk sistem Inventio — UMKM Inventory Management",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HELPER FUNCTIONS ====================

def historical_to_series(item: ItemForecastRequest) -> "pd.Series":
    """Convert historical data dari request ke pandas Series."""
    import pandas as pd

    data = [(p.date, p.value) for p in item.historicalData]
    data.sort(key=lambda x: x[0])

    dates = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d, _ in data]
    values = [v for _, v in data]

    series = pd.Series(data=values, index=dates, name="value")
    series.index.name = "date"
    return series


def get_unit(forecast_type: ForecastType) -> str:
    """Get unit berdasarkan tipe forecast."""
    return "units" if forecast_type == ForecastType.STOCK_DEMAND else "IDR"


def generate_forecast_dates(last_date: datetime, horizon: int, period: PeriodType) -> list:
    """Generate list tanggal forecast berdasarkan period."""
    from dateutil.relativedelta import relativedelta

    dates = []
    for i in range(horizon):
        if period == PeriodType.DAILY:
            current = last_date + relativedelta(days=i + 1)
            date_str = current.strftime("%Y-%m-%dZ")
        elif period == PeriodType.WEEKLY:
            current = last_date + relativedelta(weeks=i + 1)
            iso = current.isocalendar()
            date_str = f"{iso[0]}-W{iso[1]:02d}Z"
        else:
            current = last_date + relativedelta(months=i + 1)
            date_str = current.strftime("%Y-%m-01Z")

        dates.append(date_str)

    return dates


def forecast_item(
    item: ItemForecastRequest,
    forecast_type: ForecastType,
    params: ForecastParameters,
) -> ItemForecastResult:
    """Forecast untuk satu item. Returns ItemForecastResult dengan hasil atau error."""
    try:
        series = historical_to_series(item)

        if len(series) < 2:
            return ItemForecastResult(
                itemId=item.itemId,
                status="error",
                error="Minimal 2 data point historis diperlukan",
            )

        model = MovingAverageModel(window=min(len(series), 30), name="moving_average")
        model.fit(series)
        forecast_series = model.predict(horizon=params.horizon)

        last_date = series.index[-1]
        unit = get_unit(forecast_type)
        forecast_dates = generate_forecast_dates(last_date, params.horizon, params.period)

        forecast_points = []
        for i, (idx, val) in enumerate(forecast_series.items()):
            forecast_points.append(
                ForecastDataPoint(
                    forecastDate=forecast_dates[i] if i < len(forecast_dates) else idx.strftime("%Y-%m-%dZ"),
                    forecastValue=round(float(val), 2),
                    unit=unit,
                )
            )

        return ItemForecastResult(
            itemId=item.itemId,
            status="success",
            forecast=forecast_points,
        )

    except Exception as e:
        return ItemForecastResult(
            itemId=item.itemId,
            status="error",
            error=str(e),
        )


# ==================== ENDPOINTS ====================

@app.post("/api/predict", response_model=ForecastResponse)
async def predict_forecast(request: ForecastRequest):
    """
    Batch forecasting endpoint.

    POST /predict

    Request Body:
    {
        "forecastType": "stock_demand" | "sales_revenue",
        "forecastParameters": {
            "period": "daily" | "weekly" | "monthly",
            "horizon": 3
        },
        "items": [
            {
                "itemId": "uuid-produk-A001",
                "historicalData": [
                    { "date": "2026-03-15T00:00:00Z", "value": 20 },
                    { "date": "2026-04-15T00:00:00Z", "value": 25 }
                ]
            }
        ]
    }

    Returns:
    {
        "jobId": "forecast-job-123",
        "results": [
            {
                "itemId": "uuid-produk-A001",
                "status": "success",
                "forecast": [
                    { "forecastDate": "2026-05-01Z", "forecastValue": 28, "unit": "units" }
                ]
            }
        ]
    }
    """
    try:
        job_id = f"forecast-job-{uuid.uuid4().hex[:8]}"

        results = []
        for item in request.items:
            result = forecast_item(item, request.forecastType, request.forecastParameters)
            results.append(result)

        return ForecastResponse(jobId=job_id, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing forecast: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
