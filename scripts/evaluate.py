"""Evaluate forecasting models on the retail_store_inventory dataset.

Usage:
    python -m scripts.evaluate --samples 30 --test-size 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import LinearTrendModel, MovingAverageModel  # noqa: E402
from utils import evaluate, get_series, load_dataset, train_test_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecasting models")
    parser.add_argument("--samples", type=int, default=30, help="Number of (store, product) pairs to test")
    parser.add_argument("--test-size", type=int, default=30, help="Days held out as test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to dataset CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(args.dataset)
    combos = df.groupby(["Store ID", "Product ID"]).first().reset_index()
    combos = combos.sample(n=min(args.samples, len(combos)), random_state=args.seed)

    models = {
        "MovingAverage(w=30)": lambda: MovingAverageModel(window=30),
        "MovingAverage(w=14)": lambda: MovingAverageModel(window=14),
        "MovingAverage(w=7)":  lambda: MovingAverageModel(window=7),
        "LinearTrend":         lambda: LinearTrendModel(),
    }

    aggregated = {name: {"mae": [], "rmse": [], "wape": []} for name in models}

    for _, row in combos.iterrows():
        series = get_series(df, row["Store ID"], row["Product ID"])
        if len(series) <= args.test_size:
            continue
        train, test = train_test_split(series, test_size=args.test_size)
        for name, factory in models.items():
            model = factory()
            model.fit(train)
            forecast = model.predict(horizon=args.test_size)
            scores = evaluate(test.values, forecast.values)
            for k, v in scores.items():
                aggregated[name][k].append(v)

    print(f"\nEvaluated on {len(combos)} (store, product) pairs, "
          f"test size = {args.test_size} days\n")
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'WAPE%':>8}")
    print("  " + "-" * 50)
    ranked = sorted(aggregated.items(), key=lambda kv: np.mean(kv[1]["mae"]) if kv[1]["mae"] else 1e9)
    for name, scores in ranked:
        if not scores["mae"]:
            print(f"  {name:<22} (no data)")
            continue
        print(
            f"  {name:<22} "
            f"{np.mean(scores['mae']):>8.2f} "
            f"{np.mean(scores['rmse']):>8.2f} "
            f"{np.mean(scores['wape']):>7.2f}%"
        )
    print(
        "\nNotes:\n"
        "  MAE  = mean absolute error in units\n"
        "  RMSE = root mean squared error (penalises big misses)\n"
        "  WAPE = weighted absolute % error (stable when actuals contain zeros)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
