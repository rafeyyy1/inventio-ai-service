"""
Data Preparation Module — Inventio AI Service
Menyiapkan data time series untuk forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load dataset dari file CSV.
    Jika path tidak diberikan, gunakan default dataset Kaggle.
    """
    if path is None:
        path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
    else:
        path = Path(path)

    df = pd.read_csv(path)
    return df


def parse_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Parsing kolom date ke datetime index.
    Ini penting karena:
    1. Forecasting memerlukan urutan waktu yang benar
    2. Resampling/agregasi time series butuh datetime type
    3. Model (ARIMA, Prophet) memerlukan datetime index
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def aggregate_daily(df: pd.DataFrame, sales_col: str = "sales") -> pd.Series:
    """
    Agregasi data per hari.
    Jika ada duplikasi tanggal, dijumlahkan (sum).
    """
    # df sudah punya datetime index dari parse_datetime
    if df.index.name == "date" and sales_col in df.columns:
        daily = df[sales_col].resample("D").sum()
    else:
        daily = df.resample("D").sum()

    # Fill missing days dengan 0
    daily = daily.fillna(0)
    return daily


def prepare_for_model(df: pd.DataFrame, sales_col: str = "sales") -> pd.Series:
    """
    Pipeline lengkap persiapan data:
    1. Load
    2. Parse datetime
    3. Agregasi daily
    4. Return clean series

    Kenapa penting untuk forecasting:
    - Model time series (ARIMA, MA) butuh data berurutan per waktu
    - Missing days harus di-handle (fill/NaN)
    - Outlier bisa merusak prediksi
    """
    df = parse_datetime(df)
    daily = aggregate_daily(df, sales_col)
    return daily


def get_train_test_split(
    series: pd.Series,
    test_size: int = 30
) -> tuple[pd.Series, pd.Series]:
    """
    Split data jadi train & test untuk evaluasi.
    test_size = jumlah hari terakhir sebagai test set.
    """
    train = series[:-test_size]
    test = series[-test_size:]
    return train, test


def resample_weekly(series: pd.Series) -> pd.Series:
    """Resample ke mingguan untuk analisis tren."""
    return series.resample("W").sum()


def resample_monthly(series: pd.Series) -> pd.Series:
    """Resample ke bulanan untuk laporan."""
    return series.resample("ME").sum()
