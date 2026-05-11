# Examples

## sample_request.json

A ready-to-use request body for `POST /api/predict`. Generated from the Kaggle dataset, contains 2 products with 60 days of history each, asking for a 3-month monthly forecast.

### Use it

Start the service:

```bash
uvicorn api.main:app --reload
```

In another terminal:

```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_request.json | python -m json.tool
```

### Regenerate it

```bash
python -m scripts.generate_sample_request \
  --items 2 \
  --history 60 \
  --horizon 3 \
  --period monthly \
  > examples/sample_request.json
```

Options:
- `--items` (default 3): how many products to include
- `--history` (default 90): days of history per product
- `--horizon` (default 3): forecast horizon (1..12)
- `--period` (default monthly): `daily | weekly | monthly`
- `--type` (default stock_demand): `stock_demand | sales_revenue`
