# SukshmaDrishti

**Hybrid Quantum–Classical Machine Learning Platform for Early Multi-Disease Risk Detection**

[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](backend/)
[![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg)](frontend/)
[![Qiskit](https://img.shields.io/badge/QML-Qiskit%200.46-6929C4.svg)](ml/quantum/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

> Built for **SIH26139 — Smart India Hackathon**. SukshmaDrishti ("subtle vision") predicts risk for **Diabetes**, **Cardiovascular Disease**, and **Breast Cancer** from routine clinical measurements, using a rigorously benchmarked combination of **classical ML** and **quantum ML (QSVC/VQC)** models — trained on identical, leakage-free data splits and reported with full scientific honesty.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup — Step by Step](#local-setup--step-by-step)
- [Running the Application](#running-the-application)
- [Running with Docker](#running-with-docker)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Model Registry Snapshot](#model-registry-snapshot)
- [Scientific & Technical Limitations](#scientific--technical-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What This Project Does

SukshmaDrishti is a full-stack, production-styled reference platform that:

1. Accepts clinical/lab measurements (manually entered **or** auto-extracted from an uploaded medical report) for one of three diseases.
2. Runs the input through a **leakage-free preprocessing pipeline**, then through a **classical model** (Logistic Regression, Random Forest, XGBoost, SVM, ensembles, …) or a **quantum model** (QSVC / VQC built on Qiskit ZZFeatureMap circuits, run on the Aer simulator).
3. Returns a **calibrated risk score**, a **feature-level explanation** of why the model decided what it decided, and a **rule-based, non-diagnostic clinical recommendation**.
4. Persists every prediction/recommendation per authenticated user, and exposes research dashboards to compare classical vs. quantum performance, inspect the preprocessing pipeline, and browse the full model registry.

It is explicitly a **research and decision-support tool** — not a diagnostic device.

---

## Key Features

- **Multi-disease coverage** — Diabetes (Pima Indians), Cardiovascular Disease (UCI Cleveland), Breast Cancer (UCI Wisconsin/WDBC), each with its own adapter, preprocessing pipeline, and model portfolio.
- **Dual model engine** — 13 classical model types (including soft/hard voting and stacking ensembles) trained side-by-side with quantum QSVC/VQC models at 4/6/8-qubit register widths, on identical stratified 80/20 splits.
- **67 canonical, registry-verified model artifacts** (23 diabetes · 22 cardiovascular · 22 cancer), each SHA-256 integrity-verified at load time.
- **AI Medical Report Scanner** — upload a PDF or image (JPG/PNG/WEBP) lab report; the platform extracts text (native PDF text or Tesseract OCR), matches clinical parameters via regex/alias rules, normalizes units (e.g. mmol/L ↔ mg/dL), and pre-fills the assessment form for the user to verify — manual entry always remains available as a fallback.
- **Calibrated, threshold-optimized predictions** — Platt scaling + PR-AUC-driven decision thresholds instead of a naive 0.5 cutoff.
- **Model-appropriate explainability** — additive log-odds for linear models, perturbation/MDI for trees, finite-difference gradients for RBF-SVM, and angle-sensitivity projection for quantum models.
- **Deterministic recommendation engine** — disease-specific, rule-based clinical guidance with a standing Responsible-AI disclaimer. No LLM is involved in generating medical guidance.
- **Research dashboards** — Data Pipeline viewer, Classical Baselines benchmark, Quantum Lab (circuit metadata + qubit-scaling results), Hybrid Comparison matrix, and a live Experiments runner.
- **Defense-in-depth security** — JWT auth (bcrypt + HS256), OWASP security headers, sliding-window rate limiting, model-id whitelisting with path-traversal defense, SHA-256 artifact integrity checks, and row-level (BOLA-safe) user data isolation.
- **CI/CD-verified delivery** — GitHub Actions pipeline (lint → test → Docker build + health-check smoke test), 267+ automated test functions across 33 files.

---

## System Architecture

```text
React 19 + TypeScript Frontend (Vite, Tailwind CSS)
   │  Disease Forms · Report Scanner · Overview · Data-Pipeline · Quantum Lab
   │  Hybrid Comparison · History · Recommendations
   ▼  HTTP / REST  (proxied: /api/v1 → localhost:8000)
FastAPI Backend  —  /api/v1
   │  JWT Auth · Rate Limiting · OWASP Headers · CORS Allow-list · 1MB Body Limit
   ▼
Disease Adapter Routing  (Diabetes | Cardiovascular | Cancer — schema validation)
   ▼
Leakage-Free Preprocessing  (fit on train fold only, frozen for inference)
   ▼
        ┌───────────────────────┬───────────────────────┐
        ▼                       ▼
Classical ML Engine      Quantum ML Engine
(LR·SVM·RF·XGBoost·      (Mutual-info reduction → ZZFeatureMap
 ensembles)                → QSVC / VQC on Qiskit Aer)
        └───────────────────────┴───────────────────────┘
   ▼
Calibration & Threshold Optimization  (Platt scaling, PR-AUC-tuned t*)
   ▼
        ┌───────────────────────┬───────────────────────┐
        ▼                       ▼                           
Explainability Service    Recommendation Engine
        └───────────────────────┴───────────────────────┘
   ▼
Persistence  (SQLAlchemy ORM: User · PredictionRecord · RecommendationRecord)
   ▼
JSON Response → back to Frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI 0.110, Pydantic v2, Uvicorn (ASGI) |
| **Database** | SQLAlchemy 2.0 ORM, SQLite (default) / PostgreSQL-ready |
| **Auth** | PyJWT (HS256), Passlib + bcrypt |
| **Classical ML** | scikit-learn 1.4, XGBoost 2.0, NumPy, pandas, SciPy |
| **Quantum ML** | Qiskit 0.46, Qiskit Machine Learning 0.7.2, Qiskit Aer 0.13.3 |
| **Report Scanning** | pypdf (PDF text), Pillow + pytesseract (OCR), python-multipart (uploads) |
| **Frontend** | React 19, TypeScript 5.8, Vite 6, Tailwind CSS 4, lucide-react |
| **Testing** | pytest 8.0, httpx, ruff, black |
| **Deployment** | Docker (multi-stage), docker-compose, GitHub Actions CI/CD |

---

## Project Structure

```text
SukshmaDrishti/
├── backend/app/          FastAPI app: routers, services, DB, security, disease adapters
│   ├── api/v1/           auth, predict, diseases, history, recommendations, report,
│   │                     experiments, models, pipeline, baselines, comparison,
│   │                     benchmarks, evaluation, diagnostics, health
│   ├── report/           Medical report parser, matcher, normalizer, disease mapper
│   ├── diseases/         Diabetes / Cardiovascular / Cancer adapters + registry
│   ├── recommendations/  Rule-based recommendation engine + per-disease rule sets
│   ├── services/         ml_service, report_service, auth_service, prediction_service
│   ├── db/                SQLAlchemy models & session
│   └── core/              config, security, middleware, logging, errors
├── ml/                   Framework-agnostic ML/QML engine
│   ├── preprocessing/    Leakage-free preprocessing pipelines
│   ├── classical/        Classical model factory
│   ├── quantum/          Feature reduction, feature maps, QSVC, VQC, kernels
│   ├── cancer/, cardiovascular/   Per-disease pipelines
│   ├── evaluation/       Metrics, cross-validation, benchmarking
│   └── explainability/   Per-model-family explainers + dispatch service
├── frontend/src/         React 19 + TypeScript single-page app
│   ├── components/       Disease forms, Report Scanner, dashboards, auth modal
│   └── api/              Typed API client
├── scripts/              CLI: train.py, evaluate.py, experiment.py, validate_dataset.py, …
├── notebooks/            Exploratory Jupyter notebooks (01–05)
├── tests/unit/, tests/integration/   267+ pytest test functions
├── data/raw/, data/processed/        Source datasets & benchmark output
├── models/                Serialized .joblib artifacts + SHA-256 manifests
├── configs/                base.yaml, classical.yaml, quantum.yaml
├── docs/                   Architecture, methodology, phase implementation reports
├── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
├── .env.example
└── requirements.txt, pyproject.toml
```

---

## Prerequisites

Install these before you start:

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.11+ | Required for the backend, ML engine, and quantum stack |
| **Node.js** | 20.x LTS | Required for the frontend (Vite + React 19) |
| **npm** | bundled with Node | Package manager for the frontend |
| **Tesseract OCR** | any recent version | *Optional but recommended* — enables OCR on image-based medical reports (JPG/PNG/WEBP). Without it, the scanner still works for text-based PDFs and gracefully falls back to manual entry for images. |
| **Docker & Docker Compose** | recent | *Optional* — for containerized runs instead of a local Python/Node setup |
| **Git** | any | To clone the repository (skip if you already have the extracted source) |

---

## Local Setup — Step by Step

### 1. Get the code

```bash
git clone <your-repository-url> SukshmaDrishti
cd SukshmaDrishti
```

(If you're working from an extracted `.zip`, just `cd` into the extracted project folder instead.)

### 2. Create and activate a Python virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, scikit-learn, XGBoost, the full Qiskit stack (Qiskit, Qiskit Machine Learning, Qiskit Aer), and the report-scanning libraries (`pypdf`, `pytesseract`, `python-multipart`). It's a heavy install (Qiskit especially) — expect a few minutes.

### 4. (Recommended) Install the Tesseract OCR engine

`pytesseract` is a Python wrapper — it needs the actual Tesseract binary installed on your system to OCR image-based reports.

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# macOS (Homebrew)
brew install tesseract

# Windows
# Download and run the installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then ensure the install directory is added to your PATH.
```

> If you skip this step, PDF-based report scanning still works (native text extraction via `pypdf`); image-based reports will simply fall back to prompting the user for manual entry instead of erroring out.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and review the defaults — for local development the defaults work out of the box. Key variables:

| Variable | Purpose | Local Default |
|---|---|---|
| `APP_ENV` | `development` / `testing` / `production` | `development` |
| `API_HOST` / `API_PORT` | Backend bind address | `0.0.0.0` / `8000` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./sukshmadrishti.db` |
| `JWT_SECRET_KEY` | JWT signing secret — **change this for anything beyond local dev** | dev placeholder |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT expiry | `60` |
| `RANDOM_SEED` | Global reproducibility seed | `42` |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | not required locally (defaults are permissive in dev mode) |

Also set up the frontend's own env file:

```bash
cd frontend
cp .env.example .env
cd ..
```

`frontend/.env` just needs `VITE_API_URL=http://localhost:8000/api/v1` (already the default).

### 6. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend (FastAPI)

The backend imports both the `app` package (in `backend/`) and the `ml` package (at the repo root), so both directories need to be on `PYTHONPATH` when running from the project root:

```bash
# Linux / macOS
PYTHONPATH=backend:. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Windows (PowerShell)
$env:PYTHONPATH="backend;."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API base URL: **http://localhost:8000/api/v1**
- Interactive Swagger docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/api/v1/health**

### Terminal 2 — Frontend (React + Vite)

```bash
cd frontend
npm run dev
```

- App URL: **http://localhost:3000**

The Vite dev server proxies any `/api/v1/*` request straight to `http://localhost:8000`, so no extra CORS configuration is needed for local development.

---

## Running with Docker

If you'd rather not install Python/Node locally, Docker Compose builds and runs both services for you:

```bash
docker compose up --build
```

- Frontend (Nginx-served production build): **http://localhost:3000**
- Backend API: **http://localhost:8000**
- The frontend container waits for the backend's healthcheck (`/api/v1/health`) to pass before starting.

To run just the backend image standalone:

```bash
docker build -t sukshmadrishti-backend:latest .
docker run -p 8000:8000 --cpus="2.0" --memory="2048m" sukshmadrishti-backend:latest
```

> **Note:** the shipped `Dockerfile` does not install the `tesseract-ocr` system package, so image-based report OCR is unavailable inside the container by default (PDF text extraction still works). Add `RUN apt-get update && apt-get install -y tesseract-ocr` to the runtime stage of the `Dockerfile` if you need OCR support in a containerized deployment.

---

## Running Tests

```bash
# Make sure your virtual environment is activated first

# Quick syntax/compile check
python -m compileall backend ml scripts configs tests

# Lint
ruff check .

# Full test suite (unit + integration)
pytest -q
```

`pyproject.toml` already configures `pythonpath = ["backend", "."]` for pytest, so tests resolve `app` and `ml` imports automatically — no manual `PYTHONPATH` export needed when running `pytest`.

---

## API Reference

All endpoints are versioned under `/api/v1`. Full interactive documentation is always available at `/docs` once the backend is running.

| Router | Example Endpoints | Purpose |
|---|---|---|
| `health` | `GET /health` | Liveness/readiness probe |
| `diagnostics` | `GET /diagnostics` | Model registry integrity & count verification |
| `auth` | `POST /auth/register`, `POST /auth/login` | Account creation & JWT issuance |
| `diseases` | `GET /diseases`, `GET /diseases/{id}/schema` | Disease list & input feature schemas |
| `predict` | `POST /predict/{disease}` | Core inference endpoint |
| `report` | `POST /report/scan` | Upload a medical report (PDF/image) for parameter extraction |
| `history` | `GET /history` | Authenticated user's past predictions |
| `recommendations` | `GET /recommendations/{prediction_id}` | Stored clinical recommendation |
| `experiments` | `GET /experiments`, `POST /experiments` | Run/inspect quantum experiments (qubit scaling, etc.) |
| `models` | `GET /models` | Canonical model registry introspection |
| `pipeline` | `GET /pipeline/{disease}` | Preprocessing pipeline metadata per disease |
| `baselines` | `GET /baselines` | Classical baseline benchmark status/results |
| `comparison` | `GET /comparison/{disease}` | Classical vs. quantum comparison table |
| `benchmarks` | `GET /benchmarks` | Canonical benchmark records (integrity-verified) |
| `evaluation` | `GET /evaluation/{disease}/{model}` | Holdout metrics, ROC/PR curves, confusion matrix |

---

## Model Registry Snapshot

| Disease | Dataset | Total Models | Classical | Quantum |
|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes | 23 | 17 | 6 |
| Cardiovascular | UCI Cleveland | 22 | 16 | 6 |
| Breast Cancer | UCI Wisconsin (WDBC) | 22 | 16 | 6 |
| **Total** | | **67** | **49** | **18** |

Every artifact is packaged as a `.joblib` file with a companion SHA-256 manifest; a tampered or corrupted file fails to load rather than being silently served.

---

## Scientific & Technical Limitations

1. **Simulated quantum execution only.** All QSVC/VQC computation runs on Qiskit Aer's local statevector/shot simulator — the platform does not connect to physical quantum hardware and makes no hardware quantum-advantage claims.
2. **Qubit scaling trade-off.** 4-qubit quantum models currently generalize better than the 6-/8-qubit variants on this dataset — reported transparently rather than smoothed over.
3. **Cancer assessment needs pathology-grade input.** Breast cancer prediction requires 30 FNA cytology morphometric features; it cannot infer malignancy from a routine blood panel.
4. **Report Scanner is an assistive layer, not a source of truth.** Extracted values are always presented back to the user for verification before any prediction is run — nothing is auto-submitted.
5. **Not a diagnostic tool.** All outputs are statistical risk estimates for research/decision-support purposes and do not constitute medical advice or diagnosis.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` or `'ml'` | Backend started without the right `PYTHONPATH` | Run uvicorn with `PYTHONPATH=backend:.` as shown in [Running the Application](#running-the-application) |
| Frontend loads but API calls fail | Backend isn't running, or wrong port | Confirm `http://localhost:8000/api/v1/health` returns `200`, and that `frontend/.env` points to it |
| Image report upload always falls back to manual entry | Tesseract not installed / not on `PATH` | Install Tesseract (step 4 above) and restart the backend |
| `pip install` fails or hangs on the Qiskit stack | Slow network / low memory | Retry with `pip install -r requirements.txt --no-cache-dir`, or increase available memory |
| Docker frontend never becomes reachable | Backend healthcheck failing | Run `docker compose ps` and `docker compose logs backend` to inspect startup errors |
| `pytest` can't find `app`/`ml` modules | Not running from the repo root | Run `pytest` from the project root directory (where `pyproject.toml` lives) |

---

## License

MIT — see `pyproject.toml` for package metadata.

---

*SukshmaDrishti was built for SIH26139 (Smart India Hackathon) as a research and decision-support platform. It is not a certified medical device and should not be used for clinical diagnosis.*