# Inventio AI Service

Forecasting API untuk Inventio — Sistem Manajemen Inventaris UMKM.

## Struktur Project

```
ai_service/
├── data/
│   └── sample_sales.csv       # Dataset time series (date, sales, inventory, price)
├── models/
│   ├── moving_average.py      # Model utama: Moving Average (MAE ~19)
│   └── arima_model.py         # Model advanced: ARIMA
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # Pydantic schemas (request/response)
├── utils/
│   ├── data_prep.py           # Data loading & preparation
│   ├── evaluation.py          # MAE & MAPE metrics
│   └── business_logic.py      # Trend analysis & inventory alerts
├── requirements.txt
└── README.md
```

## Install

```bash
cd ai_service
pip install -r requirements.txt
```

## Run API

```bash
cd ai_service
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API akan jalan di `http://localhost:8000`
Dokumentasi interaktif: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info service |
| GET | `/health` | Health check |
| GET | `/forecast?horizon=7&model=moving_average` | Forecast + recommendation |
| GET | `/evaluate?test_size=30` | Bandingkan MA vs ARIMA |
| POST | `/retrain` | Retrain model |

## Contoh Penggunaan

### Forecast Moving Average (7 hari)
```bash
curl "http://localhost:8000/forecast?horizon=7&model=moving_average&window=30"
```

### Forecast ARIMA (14 hari)
```bash
curl "http://localhost:8000/forecast?horizon=14&model=arima"
```

### Evaluasi Model
```bash
curl "http://localhost:8000/evaluate?test_size=30"
```

## Response Contoh

```json
{
  "model": "moving_average",
  "horizon": 7,
  "forecast": [
    {"date": "2014-10-30", "forecast": 93.45},
    {"date": "2014-10-31", "forecast": 93.45},
    {"date": "2014-11-01", "forecast": 93.45}
  ],
  "trend": "up",
  "pct_change": 12.3,
  "recommendation": "restock",
  "message": "⚠️ Stok perlu ditambah! Permintaan naik 12.3% (~93 unit/hari). Estimasi stok habis dalam 8 hari."
}
```

## Integrasi dengan Backend Node.js

Backend Inventio memanggil AI service via HTTP:

```javascript
// Contoh dari server Node.js Inventio
const forecast = await fetch('http://localhost:8000/forecast?horizon=7&model=moving_average');
const data = await forecast.json();

// Gunakan data.forecast untuk tampil di frontend
// Gunakan data.recommendation untuk logika bisnis
```

**Catatan**: Untuk production, AI service deploy terpisah (Azure Container Instances / Docker) dan diakses via domain.

## Cloud Deployment (Azure)

Arsitektur deployment:

```
[Frontend React] → [Backend Node.js] → [AI Service (FastAPI)]
                                        ↓
                               [Azure Container Instances]
                               atau [Azure App Service]
```

Steps deployment:
1. Dockerize: `docker build -t inventio-ai .`
2. Push ke Azure Container Registry
3. Deploy ke Azure Container Instances (serverless)
4. Set environment variable untuk database connection
5. Backend akses AI service via URL publik

## Model Comparison (Hasil)

| Model | MAE | MAPE |
|-------|-----|------|
| Moving Average (w=30) | ~19 | ~18% |
| ARIMA (5,1,2) | ~22 | ~20% |
| Prophet | ~28 | ~25% |

**Kesimpulan**: Moving Average dipilih sebagai model utama karena:
- MAE paling rendah → prediksi paling akurat
- Simpel dan cepat
- Mudah di-interpretasi oleh UMKM
- Resource-efficient untuk cloud deployment
