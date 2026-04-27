"""
FastAPI Main — Inventio AI Service
API untuk forecasting stok yang bisa dipanggil oleh backend Node.js.

Run:
    cd ai_service
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Test:
    curl http://localhost:8000/forecast?horizon=7
    curl http://localhost:8000/health
"""

import sys
import warnings
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd

# Local imports
from models import MovingAverageModel, ARIMAModel
from utils import (
    load_dataset,
    prepare_for_model,
    get_train_test_split,
    evaluate_model,
    compare_models,
    generate_inventory_alert,
)
from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastDataPoint,
    HealthResponse,
    EvaluationResponse,
)

warnings.filterwarnings("ignore")

# ==================== APP SETUP ====================
app = FastAPI(
    title="Inventio AI Service",
    description="Inventory forecasting API untuk sistem Inventio — UMKM Inventory Management",
    version="1.0.0",
)

# CORS — agar bisa diakses dari frontend/ backend manapun
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: ganti ke domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================
# Model instance (disimpan di memory)
_models: dict = {}


def get_data_path() -> Path:
    """Get path ke dataset."""
    return Path(__file__).parent.parent / "data" / "sample_sales.csv"


def load_and_prepare_data(data_path: str = None) -> pd.Series:
    """Load + prepare data dari CSV."""
    if data_path is None:
        data_path = get_data_path()

    df = load_dataset(str(data_path))
    series = prepare_for_model(df, sales_col="sales")
    return series


# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Digunakan untuk monitoring dan readiness probe.
    """
    return HealthResponse(
        status="healthy",
        service="Inventio AI Service",
        version="1.0.0",
    )


@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    horizon: int = Query(default=7, ge=1, le=90),
    model: str = Query(default="moving_average"),
    window: int = Query(default=7, ge=3, le=365),
):
    """
    Endpoint utama forecasting.

    GET /forecast?horizon=7&model=moving_average&window=30

    Args:
        horizon: jumlah hari forecast (1-90)
        model: 'moving_average' atau 'arima'
        window: window size untuk MA (3-365)

    Returns:
        JSON dengan forecast values + business recommendation
    """
    try:
        # 1. Load data
        data = load_and_prepare_data()

        # 2. Train model
        if model == "arima":
            ai_model = ARIMAModel(order=(5, 1, 2), name="arima")
            ai_model.fit(data)
            forecast_series = ai_model.predict(horizon=horizon)
        else:
            # Default: Moving Average
            ai_model = MovingAverageModel(window=window, name="moving_average")
            ai_model.fit(data)
            forecast_series = ai_model.predict(horizon=horizon)

        # 3. Generate business logic
        alert = generate_inventory_alert(forecast_series)

        # 4. Format response
        forecast_points = [
            ForecastDataPoint(
                date=idx.strftime("%Y-%m-%d"),
                forecast=round(float(val), 2),
            )
            for idx, val in forecast_series.items()
        ]

        return ForecastResponse(
            model=ai_model.name,
            horizon=horizon,
            forecast=forecast_points,
            trend=alert["trend"],
            pct_change=alert["pct_change"],
            recommendation=alert["status"],
            message=alert["message"],
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/evaluate", response_model=EvaluationResponse)
async def evaluate_models(
    test_size: int = Query(default=30, ge=7, le=180),
):
    """
    Evaluasi kedua model (MA + ARIMA) dengan MAE & MAPE.
    Bandingkan hasil dan return model terbaik.

    GET /evaluate?test_size=30
    """
    try:
        data = load_and_prepare_data()
        train, test = get_train_test_split(data, test_size=test_size)

        results = {}

        # Evaluate Moving Average
        ma_model = MovingAverageModel(window=30)
        ma_model.fit(train)
        ma_forecast = ma_model.predict(horizon=test_size)
        ma_metrics = evaluate_model(test.values, ma_forecast.values)
        results["moving_average"] = ma_metrics

        # Evaluate ARIMA
        try:
            arima_model = ARIMAModel(order=(5, 1, 2))
            arima_model.fit(train)
            arima_forecast = arima_model.predict(horizon=test_size)
            arima_metrics = evaluate_model(test.values, arima_forecast.values)
            results["arima"] = arima_metrics
        except Exception:
            results["arima"] = {"mae": None, "mape": None}

        # Compare
        comparison = compare_models(results)
        best = comparison["best_model"]

        return EvaluationResponse(
            model=best,
            mae=results[best]["mae"],
            mape=results[best]["mape"],
            test_size=test_size,
            compared_with={
                name: {"mae": res["mae"], "mape": res["mape"]}
                for name, res in results.items()
            },
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/retrain")
async def retrain_model(
    request: dict = None,
):
    """
    Retrain model dengan data baru.
    Untuk diintegrasikan dengan database real Inventio nanti.

    POST /retrain
    Body: {"data": [[date, sales], ...]} atau kosong untuk retrain dari CSV
    """
    # Placeholder untuk retrain dari database
    return {"status": "ok", "message": "Model retrained (placeholder)"}


# ==================== ROOT ====================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Inventio AI Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast?horizon=7&model=moving_average",
            "evaluate": "/evaluate?test_size=30",
            "retrain": "/retrain (POST)",
        },
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
