# Deployment Guide — Azure

This service is deployed to Azure as a containerised application. Azure Container Apps is the recommended target because it auto-scales to zero (no cost when idle) and gives HTTPS for free.

## Prerequisites

- Azure subscription (the DTETI shared account is fine)
- `az` CLI installed and logged in: `az login`
- Docker installed locally (optional — `az acr build` can build remotely)

## Variables to set

```bash
export RG=inventio-rg                       # resource group name
export LOCATION=southeastasia               # or another close region
export ACR_NAME=inventioacr                 # must be globally unique
export APP_NAME=inventio-ai-service
export IMAGE_TAG=v1
```

## Step 1 — One-time setup

```bash
# Resource group
az group create --name $RG --location $LOCATION

# Container Registry (where the image lives)
az acr create --resource-group $RG --name $ACR_NAME --sku Basic

# Container Apps Environment (the runtime)
az containerapp env create \
  --name inventio-env \
  --resource-group $RG \
  --location $LOCATION
```

## Step 2 — Build and push the image

```bash
# Build remotely on Azure (no local Docker needed):
az acr build \
  --registry $ACR_NAME \
  --image $APP_NAME:$IMAGE_TAG \
  .
```

Or build locally and push:

```bash
docker build -t $ACR_NAME.azurecr.io/$APP_NAME:$IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $ACR_NAME.azurecr.io/$APP_NAME:$IMAGE_TAG
```

## Step 3 — Deploy the container app

```bash
az containerapp create \
  --name $APP_NAME \
  --resource-group $RG \
  --environment inventio-env \
  --image $ACR_NAME.azurecr.io/$APP_NAME:$IMAGE_TAG \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_NAME.azurecr.io \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1Gi \
  --env-vars ALLOWED_ORIGINS="https://inventio.app,https://api.inventio.app"
```

After ~30 seconds, get the public URL:

```bash
az containerapp show --name $APP_NAME --resource-group $RG \
  --query properties.configuration.ingress.fqdn -o tsv
```

Example output: `inventio-ai-service.bluestone-1234abcd.southeastasia.azurecontainerapps.io`

## Step 4 — Verify

```bash
URL=$(az containerapp show --name $APP_NAME --resource-group $RG \
  --query properties.configuration.ingress.fqdn -o tsv)

curl https://$URL/health
# {"status":"ok","service":"inventio-ai-service"}

curl -X POST https://$URL/api/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_request.json
```

## Step 5 — Tell the backend where the service is

In the Inventio backend `.env`:

```
AI_SERVICE_URL=https://inventio-ai-service.<region>.azurecontainerapps.io
```

## Updating the service

```bash
# Bump the tag
export IMAGE_TAG=v2

# Rebuild
az acr build --registry $ACR_NAME --image $APP_NAME:$IMAGE_TAG .

# Roll out
az containerapp update \
  --name $APP_NAME \
  --resource-group $RG \
  --image $ACR_NAME.azurecr.io/$APP_NAME:$IMAGE_TAG
```

Container Apps does a rolling restart with zero downtime.

## Cost estimate

With min-replicas=0 and max-replicas=3:

- Idle: **$0/month** (scales to zero)
- Active forecasting hour: ~$0.02 (0.5 vCPU + 1 GiB for one hour)
- Registry: ~$5/month for Basic SKU

For coursework usage, expect well under $5/month total.

## Troubleshooting

**Service won't start, logs show "ImportError":**
Check `requirements.txt` includes all dependencies. Rebuild and redeploy.

**`502 Bad Gateway` on the public URL:**
The container exposed the wrong port. `target-port` must be `8000` and the Dockerfile `CMD` must bind to `0.0.0.0:8000`.

**CORS errors when called from the browser:**
Update `ALLOWED_ORIGINS` and run `az containerapp update --env-vars ...`.

**Logs:**
```bash
az containerapp logs show --name $APP_NAME --resource-group $RG --follow
```
