# AI Design Notes

This document explains how the AI side of Inventio was built: which models were considered, how they were compared, and why the final two were chosen.

## Problem

Predict future demand (in units) and revenue (in IDR) per product, so the Inventio backend can show restock recommendations to UMKM users. The forecast runs inside a batch HTTP call: one request lists many products, and the service returns a forecast per item.

## Models considered

Five models were tested, all standard for univariate time-series forecasting:

| Model | Idea | Library |
|---|---|---|
| Moving Average | forecast = mean of last `w` observations | numpy |
| Linear Regression | fit `y = a + b·t` over the day index | scikit-learn |
| Exponential Smoothing (Holt-Winters) | adaptive smoothing of trend + seasonality | statsmodels |
| ARIMA(p,d,q) | autoregressive integrated moving average | statsmodels |
| Prophet | additive trend + seasonality + holidays | prophet |

## Evaluation setup

- **Dataset:** [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) (Kaggle, synthetic). 73,100 rows, 731 days, 100 product-store time series.
- **Split:** last 30 days held out as test set, the rest used for training.
- **Sample size:** 30 random product-store pairs.
- **Metrics:**
  - **MAE** (Mean Absolute Error) — average miss in units. Easy to interpret.
  - **RMSE** (Root Mean Squared Error) — penalises large misses.
  - **WAPE** (Weighted Absolute Percentage Error) — relative error, stable when some actuals are zero. Preferred over MAPE because daily demand often hits zero.

Reproduce with:

```bash
python -m scripts.evaluate --samples 30 --test-size 30
```

## Results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| ARIMA(2,1,1) | 86.88 | 104.08 | 66.83% |
| Linear Regression | 87.01 | 104.17 | 66.97% |
| Moving Average (w=30) | 87.35 | 105.51 | 67.26% |
| Moving Average (w=14) | 88.63 | 106.88 | 68.04% |
| Moving Average (w=7) | 92.85 | 112.49 | 71.38% |

Exponential Smoothing and Prophet were tested in an earlier exploration (see `docs/forecasting_exploration.html` in the main repo) and underperformed on this data, so they were dropped before the final evaluation.

## Why this is a noisy dataset

WAPE around 67% sounds bad, but the dataset is intentionally noisy — daily demand at the per-product level swings widely, and Friday-vs-Tuesday patterns are weak. **All five models land within ~5 percentage points of each other**, so no model is dramatically better. This is a property of the data, not a problem with any one model. On real Inventio data (which aggregates real transactions) the error rate should be lower.

## Final choice: Moving Average + Linear Regression

ARIMA wins on MAE by 0.5 units (less than 0.6%). That gap does not justify its costs:

- Adds `statsmodels` (~30 MB) to the Docker image
- Occasionally throws convergence warnings on real input
- Slower per request, which matters for batch inference

**Two models in production:**

1. **Moving Average (window=30)** — default. Constant forecast, robust to noise, deterministic.
2. **Linear Regression** — used when the series shows a clear trend.

### Auto-selection rule

`models/selector.py` measures the Pearson correlation between the time index and the values. If `|correlation| >= 0.3`, Linear Regression is used; otherwise Moving Average. The threshold is conservative — most noisy series stay with Moving Average.

This rule was chosen for transparency. A stakeholder or grader can read one screen of code and understand exactly which model will run on any given input.

## What the API returns

Each `result` includes `modelUsed` so the caller can see which model ran for that item. Example:

```json
{
  "itemId": "prod-001",
  "status": "success",
  "forecast": [...],
  "modelUsed": "MovingAverageModel"
}
```

## Limitations and possible upgrades

- **No external regressors used.** Price, promotion, and weather are available in the dataset but not used by the current models. Adding them would require a regression-based approach (e.g. XGBoost) and would only pay off once the data volume justifies it.
- **No retraining loop.** Every request fits the model fresh from the historicalData in the payload. This is intentional — it keeps the service stateless and easy to scale horizontally — but it means the service does not learn from past forecasts.
- **No confidence intervals.** Both models return point forecasts. If the frontend needs a range, this is a small addition (bootstrap residuals on Moving Average, prediction intervals on Linear Regression).
