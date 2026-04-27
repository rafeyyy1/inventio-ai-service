"""
CLI Runner — Inventio AI Service
Run model evaluation dan forecasting dari command line.
Tanpa perlu menjalankan API server.

Usage:
    python run.py forecast          # Forecast 7 hari (default)
    python run.py forecast --horizon 30 --model arima
    python run.py evaluate          # Bandingkan semua model
    python run.py evaluate --test-size 30
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import MovingAverageModel, ARIMAModel
from utils import (
    load_dataset,
    prepare_for_model,
    get_train_test_split,
    evaluate_model,
    compare_models,
    generate_inventory_alert,
)


def cmd_forecast(args):
    """Jalankan forecasting."""
    print("\n" + "=" * 60)
    print(f"  INVENTIO AI — Forecasting ({args.model})")
    print("=" * 60)

    # Load data
    data_path = Path(__file__).parent / "data" / "sample_sales.csv"
    df = load_dataset(str(data_path))
    series = prepare_for_model(df, sales_col="sales")
    print(f"\n[Data] Loaded {len(series)} days: {series.index[0].date()} -> {series.index[-1].date()}")
    print(f"[Data] Avg sales: {series.mean():.1f} unit/hari")
    print(f"[Data] Total sales: {series.sum():.0f} unit")

    # Train model
    if args.model == "arima":
        model = ARIMAModel(order=(5, 1, 2))
        model.fit(series)
        print(f"\n[Model] ARIMA trained (AIC: {model.get_aic():.2f})")
    else:
        model = MovingAverageModel(window=args.window)
        model.fit(series)
        print(f"\n[Model] Moving Average (window={args.window}) trained")

    # Forecast
    forecast = model.predict(horizon=args.horizon)
    print(f"\n[Forecast] {args.horizon} hari ke depan:")
    print("-" * 40)
    for date, val in forecast.items():
        print(f"  {date.strftime('%Y-%m-%d')} : {val:.1f} unit")

    # Business logic
    alert = generate_inventory_alert(forecast)
    print(f"\n[Business Logic]")
    print(f"  Trend    : {alert['trend']} ({alert['pct_change']:+.1f}%)")
    print(f"  Status   : {alert['status'].upper()}")
    print(f"  Forecast : ~{alert['future_avg']:.0f} unit/hari")
    if alert['days_until_stockout']:
        print(f"  Estimasi stok habis: {alert['days_until_stockout']} hari")
    print(f"\n  >> {alert['message']}")


def cmd_evaluate(args):
    """Evaluasi model."""
    print("\n" + "=" * 60)
    print(f"  INVENTIO AI — Model Evaluation (test_size={args.test_size})")
    print("=" * 60)

    # Load data
    data_path = Path(__file__).parent / "data" / "sample_sales.csv"
    df = load_dataset(str(data_path))
    series = prepare_for_model(df, sales_col="sales")
    train, test = get_train_test_split(series, test_size=args.test_size)

    print(f"\n[Data] Train: {len(train)} hari, Test: {len(test)} hari")

    results = {}

    # Evaluate MA
    ma = MovingAverageModel(window=7)
    ma.fit(train)
    ma_forecast = ma.predict(horizon=args.test_size)
    ma_metrics = evaluate_model(test.values, ma_forecast.values)
    results["moving_average"] = ma_metrics

    # Evaluate ARIMA
    try:
        arima = ARIMAModel(order=(5, 1, 2))
        arima.fit(train)
        arima_forecast = arima.predict(horizon=args.test_size)
        arima_metrics = evaluate_model(test.values, arima_forecast.values)
        results["arima"] = arima_metrics
    except Exception as e:
        print(f"[Warning] ARIMA failed: {e}")
        results["arima"] = {"mae": None, "mape": None}

    # Comparison
    print(f"\n[Results]")
    print("-" * 50)
    print(f"  {'Model':<20} {'MAE':>8} {'MAPE':>8}")
    print("-" * 50)
    for name, res in results.items():
        mae_str = f"{res['mae']:.2f}" if res["mae"] else "N/A"
        mape_str = f"{res['mape']:.1f}%" if res["mape"] else "N/A"
        print(f"  {name:<20} {mae_str:>8} {mape_str:>8}")
    print("-" * 50)

    comparison = compare_models(results)
    best = comparison["best_model"]
    print(f"\n[*] Best Model: {best} (MAE: {results[best]['mae']:.2f})")

    print(f"\n[Interpretation]")
    print(f"  MAE = Mean Absolute Error")
    print(f"  Rata-rata kesalahan prediksi vs actual = ~{results[best]['mae']:.0f} unit")
    print(f"  MAPE = Mean Absolute Percentage Error")
    print(f"  Rata-rata kesalahan = ~{results[best]['mape']:.0f}% dari nilai actual")


def main():
    parser = argparse.ArgumentParser(description="Inventio AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Forecast command
    p_forecast = subparsers.add_parser("forecast", help="Jalankan forecasting")
    p_forecast.add_argument("--horizon", type=int, default=7, help="Jumlah hari forecast")
    p_forecast.add_argument("--model", type=str, default="moving_average",
                            choices=["moving_average", "arima"], help="Model yang digunakan")
    p_forecast.add_argument("--window", type=int, default=7, help="Window size untuk MA")

    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Evaluasi model")
    p_eval.add_argument("--test-size", type=int, default=30, help="Jumlah hari test set")

    args = parser.parse_args()

    if args.command == "forecast":
        cmd_forecast(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
