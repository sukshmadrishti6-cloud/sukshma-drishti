# SukshmaDrishti — Production Deployment & Security Guide

This document provides complete instructions for deploying, configuring, securing, and operating the **SukshmaDrishti** Hybrid Classical-Quantum Machine Learning platform in production environments.

---

## 1. System Architecture Overview

SukshmaDrishti follows a hardened, multi-tier Single Source of Truth architecture:

$$\begin{aligned}
\text{Client Browser} &\longrightarrow \text{Nginx Reverse Proxy / Vite Client (Port 3000 / 80)} \\
&\longrightarrow \text{FastAPI REST Engine (Port 8000)} \\
&\longrightarrow \text{Cryptographic JWT Auth / RBAC Middleware} \\
&\longrightarrow \text{Authoritative Model Registry (\texttt{MODEL\_REGISTRY})} \\
&\longrightarrow \text{Production Artifacts (\texttt{v3.0} Classical \& Quantum Joblib)} \\
&\longrightarrow \text{Qiskit Aer Quantum Simulator Engine}
\end{aligned}$$

---

## 2. Canonical Runtime Specification

To guarantee zero deserialization mismatches or scientific divergence between model training and production inference, the application stack MUST run on the canonical environment:

| Layer | Runtime / Package Specification |
| :--- | :--- |
| **Python Runtime** | `Python 3.11.x` / `3.14.x` (64-bit Linux / Windows) |
| **Node.js Runtime** | `Node.js 20.x` LTS (Alpine Linux / Debian) |
| **API Framework** | `FastAPI == 0.110.0`, `Uvicorn[standard] == 0.28.0`, `Pydantic == 2.6.0` |
| **Auth & Security** | `PyJWT == 2.8.0` (Cryptographic HS256 / RS256 verification) |
| **Classical Machine Learning** | `scikit-learn == 1.4.0`, `xgboost == 2.0.0`, `numpy == 1.26.4`, `pandas == 2.2.0`, `scipy == 1.12.0` |
| **Quantum Stack** | `qiskit == 0.46.0`, `qiskit-machine-learning == 0.7.2`, `qiskit-aer == 0.13.3` |

---

## 3. Environment Configuration

Copy `.env.example` to `.env` in the project root. **Never commit `.env` files containing real secrets.**

```ini
# Core Backend Configuration
APP_NAME="SukshmaDrishti — Hybrid Quantum ML Platform"
APP_ENV=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
RANDOM_SEED=42

# CORS Approved Origins (Comma-separated list of exact production domains)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://sukshmadrishti.org

# Authentication / Supabase Cryptographic Integration
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_ANON_KEY=your_public_anon_key
SUPABASE_JWT_SECRET=your_32byte_cryptographic_jwt_signing_secret
```

> **Security Rule**: In production (`APP_ENV=production`), wildcard CORS (`CORS_ORIGINS=*`) is strictly forbidden and will cause backend startup validation to fail.

---

## 4. Multi-Stage Docker Container Deployment

### Local Container Build & Startup
To build and start both the backend FastAPI service and the frontend Nginx web server using Docker Compose:

```powershell
# 1. Build immutable production container images
docker compose build --no-cache

# 2. Spin up containers in detached background mode
docker compose up -d

# 3. Verify container health status
docker compose ps
```

### Container Verification
- **Frontend Nginx Web App**: Accessible at `http://localhost:3000` (or `http://localhost:80` when deployed).
- **Backend FastAPI Service**: Accessible at `http://localhost:8000/api/v1/health`.

### Docker Architecture
- **Backend Image (`Dockerfile`)**: Multi-stage build based on `python:3.11-slim-bookworm`. Drops root privileges to run as unprivileged non-root user `appuser` (UID 10001). Includes built-in Python standard library health check.
- **Frontend Image (`frontend/Dockerfile`)**: Multi-stage build based on `node:20-alpine` (Stage 1 build) and `nginx:alpine` (Stage 2 web server). Nginx proxies `/api/v1/` requests directly to `http://backend:8000/api/v1/` with 300s timeouts for long quantum simulations.

---

## 5. Security & Threat Model Mitigation

### OWASP Defensive Headers
Every response emitted by the backend automatically includes standard OWASP defensive headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'; frame-ancestors 'none'; object-src 'none'`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (production mode)

### Payload Size & Rate Limiting
- **Payload Limit**: `RequestSizeLimitMiddleware` restricts request bodies to a maximum of **1 MB** ($1,048,576$ bytes), returning `HTTP 413 Payload Too Large` for oversized requests.
- **Rate Limiting**: `RateLimiterMiddleware` enforces a sliding-window rate limit of **120 requests/minute** per IP address, returning `HTTP 429 Too Many Requests` with a `Retry-After` header.

### Cryptographic JWT Verification & RBAC
- **Token Verification**: Tokens are verified using PyJWT against `SUPABASE_JWT_SECRET` checking signature (`HS256`), expiration (`exp`), audience (`aud="authenticated"`), and issuer (`iss`).
- **Role Enforcement**: `require_role("physician")` and `require_role("admin")` protect administrative and clinical endpoints.
- **BOLA/IDOR Defense**: `verify_user_resource_access` guarantees users can only access their authorized user resources.

---

## 6. Operational Health Checks & Verification

### 1. Health & Model Readiness Endpoint
```powershell
curl -X GET http://localhost:8000/api/v1/health
```
**Expected Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "production",
  "backend_engine": "FastAPI",
  "models_available": [
    {
      "model_id": "random_forest_v3",
      "display_name": "Random Forest (v3.0)",
      "model_family": "classical",
      "status": "Ready",
      "probability_support": true,
      "explainability_support": true
    },
    ...
  ]
}
```

### 2. Live Inference Request
```powershell
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    "model_id": "svm_rbf_v3"
  }'
```

### 3. Asynchronous Quantum Experiment Run
```powershell
curl -X POST http://localhost:8000/api/v1/experiments/run \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Production 4Q QSVC Test",
    "qubit_count": 4,
    "feature_map": "ZZFeatureMap",
    "ansatz": "n/a",
    "optimizer": "n/a",
    "shots": 1024
  }'
```

---

## 7. Troubleshooting & Operational Procedures

- **Model Artifact Loading Errors**: Ensure `models/classical/` and `models/quantum/` contain valid `.joblib` artifacts alongside their corresponding `_manifest.json` metadata files.
- **Path Traversal Rejections**: Non-registry model keys containing slash (`/`), backslash (`\`), or parent directory (`..`) references will automatically be rejected with `HTTP 404 Not Found`.
- **Quantum Simulation Latency**: 4-qubit Aer simulations execute in $< 500\,\text{ms}$. If running under constrained compute, set `workers=2` in Uvicorn to ensure worker availability during CPU-bound Qiskit statevector simulation.
