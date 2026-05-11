"""Load and reshape the retail-store-inventory dataset for forecasting."""

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "retail_store_inventory.csv"


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the retail-store-inventory CSV as a long-format dataframe.

    Expected columns: Date, Store ID, Product ID, Units Sold, Price, ...
    """
    path = Path(path) if path else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from Kaggle and place it in the data/ folder."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)


def get_series(
    df: pd.DataFrame,
    store_id: str,
    product_id: str,
    value_col: str = "Units Sold",
) -> pd.Series:
    """Extract a single (store, product) time series as a daily-frequency Series."""
    mask = (df["Store ID"] == store_id) & (df["Product ID"] == product_id)
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(f"No rows for store={store_id} product={product_id}")
    series = sub.set_index("Date")[value_col].astype(float)
    return series.asfreq("D").ffill()


def train_test_split(series: pd.Series, test_size: int = 30) -> tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(series):
        raise ValueError(f"test_size must be in (0, {len(series)})")
    return series.iloc[:-test_size], series.iloc[-test_size:]
