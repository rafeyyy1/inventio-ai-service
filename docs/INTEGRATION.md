# Backend Integration Guide

This document is written for whoever works on the Inventio Express backend. It shows how to call the AI service and how to persist the result.

## Where to call from

The recommended pattern is a single Express route that the frontend calls when the user opens the "Stock Forecast" page:

```
POST /api/v1/forecasts/run
```

The backend then queries `StockMovement` (or `TransactionItem`) for the requested products, builds an AI service request, and persists the result.

## Example (Express + Prisma)

```js
// server/services/forecastService.js
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();
const AI_SERVICE_URL = process.env.AI_SERVICE_URL; // set in .env

export async function runStockForecast({ tenantId, productIds, horizon = 3 }) {
  // 1. Pull last 90 days of movements per product
  const since = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);
  const movements = await prisma.stockMovement.findMany({
    where: {
      tenantId,
      productId: { in: productIds },
      createdAt: { gte: since },
    },
    select: { productId: true, quantity: true, createdAt: true, type: true },
  });

  // 2. Group by product, aggregate per day. Only count outbound movements
  //    (sales / stock-out) as demand.
  const byProduct = new Map();
  for (const m of movements) {
    if (m.type !== "OUT") continue; // adjust to your TransactionType code
    const day = m.createdAt.toISOString().slice(0, 10);
    if (!byProduct.has(m.productId)) byProduct.set(m.productId, new Map());
    const days = byProduct.get(m.productId);
    days.set(day, (days.get(day) ?? 0) + m.quantity);
  }

  // 3. Build request body
  const body = {
    forecastType: "stock_demand",
    forecastParameters: { period: "monthly", horizon },
    items: [...byProduct.entries()].map(([productId, days]) => ({
      itemId: productId,
      historicalData: [...days.entries()]
        .sort()
        .map(([day, value]) => ({ date: `${day}T00:00:00Z`, value })),
    })),
  };

  // 4. Call the AI service
  const res = await fetch(`${AI_SERVICE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30_000), // 30s timeout
  });

  if (!res.ok) {
    throw new Error(`AI service returned ${res.status}: ${await res.text()}`);
  }

  const { jobId, results } = await res.json();

  // 5. Persist successful forecasts
  const rows = [];
  for (const r of results) {
    if (r.status !== "success") continue;
    for (const point of r.forecast) {
      rows.push({
        tenantId,
        productId: r.itemId,
        type: "STOCK_DEMAND",
        forecastValue: point.forecastValue,
        unit: point.unit,
        forecastDate: parseForecastDate(point.forecastDate),
      });
    }
  }
  if (rows.length) await prisma.forecast.createMany({ data: rows });

  return { jobId, results };
}

function parseForecastDate(s) {
  // The AI service returns dates like "2026-05-01Z" (monthly) or
  // "2026-W18Z" (weekly) or "2026-05-01Z" (daily).
  if (/^\d{4}-W\d{2}Z$/.test(s)) {
    // Weekly: take the Monday of that ISO week
    const [year, week] = s.replace("Z", "").split("-W").map(Number);
    return mondayOfIsoWeek(year, week);
  }
  return new Date(s.replace("Z", "Z"));
}
```

## Error handling

The AI service returns 200 with per-item errors inside `results[]`:

```json
{
  "jobId": "forecast-job-c5839c79",
  "results": [
    { "itemId": "p1", "status": "success", "forecast": [...] },
    { "itemId": "p2", "status": "error", "error": "At least 2 historical data points are required", "forecast": [] }
  ]
}
```

So the backend should:
- Log items with `status: "error"` but **not** fail the user's request.
- Show the user only the items that succeeded.

A 5xx from the AI service is a service-level failure (network, restart, crash). Retry with backoff, then surface a "forecast unavailable, try again" to the user.

## Mapping to the `Forecast` table

The AI response maps cleanly to the existing Prisma model:

| Response field | Forecast column |
|---|---|
| `result.itemId` | `productId` |
| `point.forecastValue` | `forecastValue` |
| `point.unit` | `unit` |
| `point.forecastDate` | `forecastDate` (after parsing) |
| `forecastType` from request | `type` (`STOCK_DEMAND` or `SALES_REVENUE`) |

`tenantId` comes from the authenticated request, not the AI response.

## Caching

Forecasts don't change minute-to-minute. The backend can cache the last forecast per product for 24 hours, with an invalidation hook on new stock movements. This is optional — the AI service itself is fast enough to run on demand for typical UMKM batch sizes (< 200 products).
