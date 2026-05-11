"""Request and response schemas for POST /api/predict.

Matches the contract in `docs/Spesifikasi_API_Forecasting_AI.pdf`.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastType(str, Enum):
    STOCK_DEMAND = "stock_demand"
    SALES_REVENUE = "sales_revenue"


class PeriodType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class HistoricalDataPoint(BaseModel):
    date: str = Field(description="ISO 8601 date, e.g. 2026-03-15T00:00:00Z")
    value: float = Field(description="Units sold or revenue in IDR")


class ItemForecastRequest(BaseModel):
    itemId: str = Field(description="Product UUID")
    historicalData: List[HistoricalDataPoint] = Field(
        min_length=2, description="At least 2 historical points required"
    )


class ForecastParameters(BaseModel):
    period: PeriodType = Field(default=PeriodType.MONTHLY)
    horizon: int = Field(default=3, ge=1, le=12, description="Number of future periods")


class ForecastRequest(BaseModel):
    forecastType: ForecastType
    forecastParameters: ForecastParameters
    items: List[ItemForecastRequest] = Field(min_length=1)


class ForecastDataPoint(BaseModel):
    forecastDate: str
    forecastValue: float
    unit: str


class ItemForecastResult(BaseModel):
    itemId: str
    status: str = Field(description="success or error")
    forecast: List[ForecastDataPoint] = Field(default_factory=list)
    error: Optional[str] = None
    modelUsed: Optional[str] = None


class ForecastResponse(BaseModel):
    jobId: str
    results: List[ItemForecastResult]
