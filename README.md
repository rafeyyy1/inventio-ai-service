# Inventio AI Service

Microservice untuk prediksi inventory menggunakan forecasting models.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn api.main:app --reload
```

API akan berjalan di `http://localhost:8000`

## Test

```bash
pytest tests/
```
