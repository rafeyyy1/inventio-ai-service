"""
Business Logic — Inventio AI Service
Menambahkan logika bisnis: restock, aman, overstock.
"""

import numpy as np
import pandas as pd


def analyze_trend(forecast_series: pd.Series) -> dict:
    """
    Analisa tren dari data forecast.
    Bandingkan rata-rata 7 hari terakhir vs 7 hari ke depan.

    Return:
        trend: "up" | "stable" | "down"
        pct_change: persentase perubahan
    """
    if len(forecast_series) < 14:
        # Jika data kurang dari 14 hari, gunakan semua data
        recent_avg = forecast_series.iloc[:len(forecast_series)//2].mean()
        future_avg = forecast_series.iloc[len(forecast_series)//2:].mean()
    else:
        recent_avg = forecast_series.iloc[:7].mean()
        future_avg = forecast_series.iloc[-7:].mean()

    pct_change = ((future_avg - recent_avg) / (recent_avg + 1e-9)) * 100

    if pct_change > 5:
        trend = "up"
    elif pct_change < -5:
        trend = "down"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "pct_change": round(float(pct_change), 2),
        "recent_avg": round(float(recent_avg), 2),
        "future_avg": round(float(future_avg), 2),
    }


def get_recommendation(trend: str, avg_forecast: float) -> str:
    """
    Business logic utama:
    - Trend naik → restock (stok akan habis)
    - Trend stabil → aman (stok cukup)
    - Trend turun → overstock (stok menumpuk)

    Args:
        trend: "up" | "stable" | "down"
        avg_forecast: rata-rata forecast untuk konteks

    Returns:
        "restock" | "aman" | "overstock"
    """
    if trend == "up":
        return "restock"
    elif trend == "down":
        return "overstock"
    else:
        return "aman"


def generate_inventory_alert(
    forecast_series: pd.Series,
    current_stock: float = None,
    safety_stock: float = 50
) -> dict:
    """
    Generate alert lengkap untuk sistem inventaris.

    Args:
        forecast_series: data forecast
        current_stock: stok saat ini (opsional, default dari data terakhir)
        safety_stock: batas minimum stok pengaman (default 50 unit)

    Returns:
        {
            "status": "restock" | "aman" | "overstock",
            "trend": "up" | "stable" | "down",
            "message": "Pesan dalam bahasa Indonesia untuk user",
            "days_until_stockout": int | None (null jika trend turun/stabil)
        }
    """
    if current_stock is None:
        # Ambil dari data forecast (asumsikan forecast = demand)
        current_stock = float(forecast_series.iloc[-1]) if len(forecast_series) > 0 else safety_stock

    trend_info = analyze_trend(forecast_series)
    trend = trend_info["trend"]
    avg_forecast = trend_info["future_avg"]
    recommendation = get_recommendation(trend, avg_forecast)

    # Hitung days until stockout hanya jika trend naik
    days_until_stockout = None
    if trend == "up" and avg_forecast > 0:
        days_until_stockout = int(current_stock / avg_forecast)

    # Generate pesan
    messages = {
        "restock": f"[WARNING] Stok perlu ditambah! Permintaan naik {trend_info['pct_change']:.1f}% "
                   f"(~{avg_forecast:.0f} unit/hari). Estimasi stok habis dalam "
                   f"{days_until_stockout} hari jika tidak di-restock.",
        "aman": f"[OK] Stok dalam kondisi aman. Permintaan stabil "
                f"(~{avg_forecast:.0f} unit/hari). Tidak perlu action.",
        "overstock": f"[INFO] Permintaan turun {abs(trend_info['pct_change']):.1f}%. "
                     f"Kurangi pembelian stok baru untuk menghindari penumpukan.",
    }

    return {
        "status": recommendation,
        "trend": trend,
        "pct_change": trend_info["pct_change"],
        "recent_avg": trend_info["recent_avg"],
        "future_avg": avg_forecast,
        "current_stock": current_stock,
        "days_until_stockout": days_until_stockout,
        "safety_stock": safety_stock,
        "message": messages[recommendation],
    }
