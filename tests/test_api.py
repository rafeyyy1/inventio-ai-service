"""Tests for the FastAPI endpoint."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app

client = TestClient(app)


def _historical(n: int = 60) -> list[dict]:
    """Generate n daily points ending 2024-03-01."""
    from datetime import datetime, timedelta
    end = datetime(2024, 3, 1)
    return [
        {"date": (end - timedelta(days=n - i - 1)).strftime("%Y-%m-%dT00:00:00Z"),
         "value": float(20 + i)}
        for i in range(n)
    ]


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_success() -> None:
    body = {
        "forecastType": "stock_demand",
        "forecastParameters": {"period": "monthly", "horizon": 3},
        "items": [
            {"itemId": "prod-001", "historicalData": _historical(60)},
        ],
    }
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["jobId"].startswith("forecast-job-")
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["itemId"] == "prod-001"
    assert result["status"] == "success"
    assert len(result["forecast"]) == 3
    for point in result["forecast"]:
        assert point["unit"] == "units"
        assert point["forecastValue"] > 0


def test_predict_sales_revenue_unit() -> None:
    body = {
        "forecastType": "sales_revenue",
        "forecastParameters": {"period": "daily", "horizon": 5},
        "items": [{"itemId": "p1", "historicalData": _historical(30)}],
    }
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200
    assert all(p["unit"] == "IDR" for p in r.json()["results"][0]["forecast"])


def test_predict_error() -> None:
    body = {
        "forecastType": "stock_demand",
        "forecastParameters": {"period": "daily", "horizon": 3},
        "items": [
            {"itemId": "x", "historicalData": [{"date": "2024-01-01T00:00:00Z", "value": 10}]},
        ],
    }
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422
