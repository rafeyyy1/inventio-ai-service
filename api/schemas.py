"""
Pydantic Schemas — Inventio AI Service
Definisi request/response untuk batch forecasting API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ==================== ENUMS ====================

class ForecastType(str, Enum):
    STOCK_DEMAND = "stock_demand"
    SALES_REVENUE = "sales_revenue"


class PeriodType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ==================== REQUEST SCHEMAS ====================

class HistoricalDataPoint(BaseModel):
    """Satu titik data historis."""
    date: str = Field(description="ISO 8601 date string, contoh: 2026-03-15T00:00:00Z")
    value: float = Field(description="Nilai historical (unit untuk stock_demand, IDR untuk sales_revenue)")


class ItemForecastRequest(BaseModel):
    """Request untuk satu item/produk."""
    itemId: str = Field(description="UUID produk")
    historicalData: List[HistoricalDataPoint] = Field(min_length=2, description="Minimal 2 data point historis")


class ForecastParameters(BaseModel):
    """Parameter untuk forecasting."""
    period: PeriodType = Field(default=PeriodType.MONTHLY, description="Period: daily | weekly | monthly")
    horizon: int = Field(default=3, ge=1, le=12, description="Jumlah periode ke depan yang di-prediksi")


class ForecastRequest(BaseModel):
    """Request utama untuk batch forecasting."""
    forecastType: ForecastType = Field(description="Tipe forecast: stock_demand atau sales_revenue")
    forecastParameters: ForecastParameters = Field(description="Parameter forecast")
    items: List[ItemForecastRequest] = Field(min_length=1, description="Array item yang akan di-forecast")


# ==================== RESPONSE SCHEMAS ====================

class ForecastDataPoint(BaseModel):
    """Satu titik hasil forecast."""
    forecastDate: str = Field(description="Tanggal forecast, contoh: 2026-05-01Z")
    forecastValue: float = Field(description="Nilai forecast")
    unit: str = Field(description="Satuan: units atau IDR")


class ItemForecastResult(BaseModel):
    """Hasil forecast untuk satu item."""
    itemId: str = Field(description="UUID produk yang di-request")
    status: str = Field(description="Status: success atau error")
    forecast: List[ForecastDataPoint] = Field(default_factory=list, description="Array hasil forecast")
    error: Optional[str] = Field(default=None, description="Pesan error jika status = error")


class ForecastResponse(BaseModel):
    """Response utama batch forecasting."""
    jobId: str = Field(description="ID unik job untuk tracking")
    results: List[ItemForecastResult] = Field(description="Array hasil per item")
