# Data

This folder holds the dataset used for model evaluation and demo requests.

## retail_store_inventory.csv

- **Source:** [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) on Kaggle (synthetic, MIT-style use).
- **Size:** 73,100 rows, ~6 MB.
- **Span:** 2022-01-01 to 2024-01-01 (731 days).
- **Granularity:** daily, per (Store ID, Product ID). 5 stores × 20 products = 100 series.
- **Columns:** `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Units Ordered`, `Demand Forecast`, `Price`, `Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, `Seasonality`.

## Why this dataset

It mirrors the production schema in the Inventio backend (`StockMovement`, `Product`, `Stock`): one daily observation per product per warehouse, with quantity sold and price. When the real Inventio database is populated, the same code path can read from PostgreSQL instead of this CSV.

## Replacing with real data

The service only depends on the CSV during evaluation and sample-request generation (`scripts/evaluate.py`, `scripts/generate_sample_request.py`). The API itself receives historical data from the request body, so swapping data sources doesn't require changing the API code.
