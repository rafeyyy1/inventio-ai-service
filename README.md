# Inventio AI Service

Batch forecasting microservice for the **Inventio** inventory management system. Predicts future stock demand and sales revenue per product, exposed as a single HTTP endpoint.

This service is a separate repository from the main Inventio backend. The backend (Node.js + Express) calls this service over HTTP when it needs forecasts.

## What it does

- **Input:** an array of products, each with its own daily historical demand.
- **Output:** forecast values per product for the next N periods (daily, weekly, or monthly).
- **Models:** Moving Average (default) and Linear Regression (auto-selected when the series shows a clear trend).

The contract matches `docs/Spesifikasi_API_Forecasting_AI.pdf`: one batch endpoint, one request → one job ID with per-item results.

## Project layout

```
ai_service/
├── api/
│   ├── main.py          FastAPI app and the /api/predict endpoint
│   └── schemas.py       Pydantic request/response models
├── models/
│   ├── moving_average.py
│   ├── linear_trend.py
│   └── selector.py      Picks the model based on trend strength
├── utils/
│   ├── data_prep.py     Loads and reshapes the Kaggle dataset
│   ├── evaluation.py    MAE, RMSE, WAPE
│   └── inventory_alert.py
├── scripts/
│   ├── evaluate.py                  Compare models on the dataset
│   └── generate_sample_request.py   Build a sample POST body
├── tests/               pytest suite (12 tests)
├── data/
│   └── retail_store_inventory.csv   Kaggle dataset for eval & demos
├── docs/
│   ├── AI_DESIGN.md     Why these models, with the eval numbers
│   ├── NETWORKING.md    Protocol, ports, CORS, how the backend calls it
│   ├── DEPLOYMENT.md    Step-by-step Azure Container Apps deploy
│   └── INTEGRATION.md   Sample Express + Prisma code for the backend
├── examples/
│   └── sample_request.json          Ready-to-use POST body
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Quick start

```bash
git clone <this-repo>
cd ai_service
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI.

## API

### `POST /api/predict`

Request body:

```json
{
  "forecastType": "stock_demand",
  "forecastParameters": { "period": "monthly", "horizon": 3 },
  "items": [
    {
      "itemId": "uuid-product-A001",
      "historicalData": [
        { "date": "2025-11-01T00:00:00Z", "value": 20 },
        { "date": "2025-11-02T00:00:00Z", "value": 25 }
      ]
    }
  ]
}
```

Response:

```json
{
  "jobId": "forecast-job-c5839c79",
  "results": [
    {
      "itemId": "uuid-product-A001",
      "status": "success",
      "forecast": [
        { "forecastDate": "2025-12-01Z", "forecastValue": 22.5, "unit": "units" },
        { "forecastDate": "2026-01-01Z", "forecastValue": 22.5, "unit": "units" },
        { "forecastDate": "2026-02-01Z", "forecastValue": 22.5, "unit": "units" }
      ],
      "modelUsed": "MovingAverageModel"
    }
  ]
}
```

- `forecastType`: `stock_demand` (unit: `units`) or `sales_revenue` (unit: `IDR`).
- `forecastParameters.period`: `daily` | `weekly` | `monthly`.
- `forecastParameters.horizon`: 1..12.
- Each item is forecast independently. Errors on one item don't fail the whole batch — that item gets `status: "error"` with an `error` message.

### `GET /health`

Returns `{"status": "ok"}`. Used by Docker and cloud platforms as a liveness probe.

## Try it

In one terminal:

```bash
uvicorn api.main:app --reload
```

In another:

```bash
python -m scripts.generate_sample_request --items 3 --history 90 > sample_request.json
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json | python -m json.tool
```

## Model choice

Five models were compared on the Kaggle [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) (30 random store-product pairs, 30-day holdout):

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Linear Regression | 87.01 | 104.17 | 66.97% |
| Moving Average (w=30) | 87.35 | 105.51 | 67.26% |
| Moving Average (w=14) | 88.63 | 106.88 | 68.04% |
| Moving Average (w=7) | 92.85 | 112.49 | 71.38% |
| ARIMA(2,1,1) | 86.88 | 104.08 | 66.83% |

ARIMA wins by ~0.5% MAE but adds ~30 MB of dependencies (statsmodels), occasional convergence warnings, and slower batch inference. The gain doesn't justify the cost in a real-time API, so the production build uses only Moving Average and Linear Regression. The service picks Linear when the series has a clear trend (|correlation with time| ≥ 0.3) and Moving Average otherwise.

Reproduce the table:

```bash
python -m scripts.evaluate --samples 30 --test-size 30
```

WAPE values are high (~67%) because the dataset is intentionally noisy — daily demand fluctuates a lot at the per-product level. The metric reflects the inherent difficulty of the data, not the model. Production demand from the Inventio database should be less noisy because it aggregates real transactions.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

12 tests cover the models, the selector, and the API end-to-end.

## Docker

```bash
docker build -t inventio-ai-service .
docker run -p 8000:8000 --env-file .env inventio-ai-service
```

The image is ~250 MB (slim Python base + pandas/sklearn). Health check is built in.

## Deploying to Azure

1. Push image to Azure Container Registry:
   ```bash
   az acr build --registry <your-acr> --image inventio-ai-service:latest .
   ```
2. Deploy to Azure Container Apps (recommended) or Container Instances:
   ```bash
   az containerapp create \
     --name inventio-ai-service \
     --resource-group <rg> \
     --image <your-acr>.azurecr.io/inventio-ai-service:latest \
     --target-port 8000 \
     --ingress external \
     --env-vars ALLOWED_ORIGINS="https://your-frontend.com,https://your-backend.com"
   ```
3. In the Inventio backend, set `AI_SERVICE_URL` to the public ingress URL and call `POST {AI_SERVICE_URL}/api/predict`.

## How the backend calls this service

From the Inventio Express server:

```js
const res = await fetch(`${process.env.AI_SERVICE_URL}/api/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    forecastType: "stock_demand",
    forecastParameters: { period: "monthly", horizon: 3 },
    items: products.map(p => ({
      itemId: p.id,
      historicalData: p.movements.map(m => ({
        date: m.createdAt.toISOString(),
        value: m.quantity,
      })),
    })),
  }),
});
const { jobId, results } = await res.json();
// Persist each result to the Forecast table:
//   { productId, type, forecastValue, unit, forecastDate }
```

The shape of `forecast[]` lines up directly with the `Forecast` model in `schema.prisma`. See `docs/INTEGRATION.md` for a full Express + Prisma example.

## Further reading

- `docs/AI_DESIGN.md` — model comparison and selection rationale (the AI part of Senpro grading)
- `docs/NETWORKING.md` — network topology, ports, CORS (the networking part)
- `docs/DEPLOYMENT.md` — Azure deployment walkthrough (the cloud part)
- `docs/INTEGRATION.md` — copy-paste backend code

## License

For coursework use (Praktikum Senior Project, DTETI UGM 2026).
