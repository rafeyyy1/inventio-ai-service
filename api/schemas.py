"""
Pydantic Schemas — Inventio AI Service
Definisi request/response untuk FastAPI.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ForecastRequest(BaseModel):
    """Request body untuk /forecast endpoint."""
    horizon: int = Field(default=7, ge=1, le=90, description="Jumlah hari forecast")
    model: str = Field(default="moving_average", description="Model: 'moving_average' atau 'arima'")
    window: int = Field(default=30, ge=3, le=365, description="Window size untuk MA")


class ForecastDataPoint(BaseModel):
    """Satu titik data forecast."""
    date: str
    forecast: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None


class ForecastResponse(BaseModel):
    """
    Response utama API forecasting.
    Include model, forecast values, dan business recommendation.
    """
    model: str
    horizon: int
    forecast: List[ForecastDataPoint]
    trend: str = Field(description="'up' | 'stable' | 'down'")
    pct_change: float = Field(description="Persentase perubahan tren")
    recommendation: str = Field(description="'restock' | 'aman' | 'overstock'")
    message: str = Field(description="Pesan dalam Bahasa Indonesia")
    metrics: Optional[dict] = Field(default=None, description="MAE/MAPE jika ada test data")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


class TrainRequest(BaseModel):
    """Request untuk retrain model."""
    data_path: Optional[str] = None
    model: str = Field(default="moving_average")


class EvaluationResponse(BaseModel):
    """Response evaluasi model."""
    model: str
    mae: float
    mape: float
    test_size: int
    compared_with: Optional[dict] = None
