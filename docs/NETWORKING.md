# Networking

This service exposes a REST API over HTTP. This document describes the network design, ports, and how it integrates with the rest of Inventio.

## Architecture

```
┌────────────┐      HTTPS      ┌──────────────┐      HTTP       ┌─────────────────┐
│  Frontend  │ ──────────────► │   Backend    │ ──────────────► │  AI Service     │
│  (React)   │ ◄────────────── │  (Express)   │ ◄────────────── │  (FastAPI)      │
│  port 5173 │                 │  port 5000   │                 │  port 8000      │
└────────────┘                 └──────┬───────┘                 └────────┬────────┘
                                      │                                  │
                                      ▼                                  │
                               ┌──────────────┐                          │
                               │  PostgreSQL  │                          │
                               │  (Prisma)    │ ◄────────────────────────┘
                               └──────────────┘     forecast results
                                                    written by backend
```

The AI service does **not** talk to the database directly. The backend reads stock movements from PostgreSQL, sends them to the AI service over HTTP, and persists the forecast results back into the `Forecast` table.

This separation:
- Keeps the AI service stateless (easier to scale, easier to restart, no migration coupling).
- Keeps the database password out of the AI service.
- Lets the AI service be replaced or A/B-tested without touching the database layer.

## Protocols and ports

| Endpoint | Protocol | Default port | Auth |
|---|---|---|---|
| AI Service `GET /health` | HTTP/1.1 | 8000 | none (liveness probe) |
| AI Service `POST /api/predict` | HTTP/1.1 | 8000 | none on the wire (the backend is the only client; restrict by network ACL in production) |

In Azure Container Apps, the service is fronted by HTTPS automatically.

## Request and response over the network

`POST /api/predict` accepts JSON, returns JSON. Content-Type and Accept are both `application/json`. No multipart, no streaming, no websockets — a single round trip per forecast batch.

A typical batch from the backend is ~100 products × 90 days of history ≈ 60 KB compressed. Well under the default 1 MB body limit. If the backend ever needs to forecast all products at once (thousands), the recommended pattern is to chunk requests into batches of ~50 items.

## CORS

CORS is restricted by the `ALLOWED_ORIGINS` environment variable. Default in `.env.example` allows only `localhost:3000` and `localhost:5173`. In production, set it to the Inventio frontend and backend URLs:

```
ALLOWED_ORIGINS=https://inventio.app,https://api.inventio.app
```

The wildcard origin `*` is intentionally not used.

## Failure modes and timeouts

- **Per-item failure does not fail the batch.** If item 3 of 10 has bad input, the response still has 10 entries — item 3 has `status: "error"` and an error message, the other 9 still get forecasts. This avoids the "all-or-nothing" cliff for the backend.
- **No retry inside the service.** Retries are the caller's responsibility. The recommended pattern in the backend is exponential backoff on HTTP 5xx and timeouts.
- **Recommended client timeout:** 30 seconds for batches up to 100 items. Forecasting is CPU-bound but fast — Moving Average is O(n), Linear Regression is O(n) for `fit`. A batch of 100 finishes well under 1 second on a 1-core container.

## Local testing

```bash
# Terminal 1: start the service
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: health check
curl http://localhost:8000/health

# Terminal 2: a real forecast call
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_request.json
```
