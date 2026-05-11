"""Build a sample POST /api/predict request from the dataset.

Picks N random (store, product) pairs, takes the last K days of history, and
prints a JSON body that can be piped into curl.

Usage:
    python -m scripts.generate_sample_request --items 3 --history 90 > sample_request.json
    curl -X POST http://localhost:8000/api/predict \\
         -H "Content-Type: application/json" \\
         -d @sample_request.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import get_series, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample request for /api/predict")
    parser.add_argument("--items", type=int, default=3, help="How many products to include")
    parser.add_argument("--history", type=int, default=90, help="Days of history per item")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon")
    parser.add_argument("--period", choices=["daily", "weekly", "monthly"], default="monthly")
    parser.add_argument("--type", choices=["stock_demand", "sales_revenue"], default="stock_demand")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(args.dataset)
    combos = df.groupby(["Store ID", "Product ID"]).first().reset_index()
    combos = combos.sample(n=args.items, random_state=args.seed)

    value_col = "Units Sold" if args.type == "stock_demand" else "Price"

    items = []
    for _, row in combos.iterrows():
        series = get_series(df, row["Store ID"], row["Product ID"], value_col=value_col)
        if args.type == "sales_revenue":
            # revenue = price * units sold
            units = get_series(df, row["Store ID"], row["Product ID"], value_col="Units Sold")
            series = (series * units).reindex(units.index).ffill()
        recent = series.iloc[-args.history:]
        items.append({
            "itemId": f"{row['Store ID']}-{row['Product ID']}",
            "historicalData": [
                {"date": idx.strftime("%Y-%m-%dT00:00:00Z"), "value": float(val)}
                for idx, val in recent.items()
            ],
        })

    body = {
        "forecastType": args.type,
        "forecastParameters": {"period": args.period, "horizon": args.horizon},
        "items": items,
    }
    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
