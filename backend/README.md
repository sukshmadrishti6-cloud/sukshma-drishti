# SukshmaDrishti Backend API Service

FastAPI service providing production-grade REST API endpoints for classical & quantum model inference, explainability (XAI), health monitoring, and experiment metadata persistence.

## Architecture

```text
backend/
├── app/
│   ├── main.py          # FastAPI application entry point with security middlewares
│   ├── api/
│   │   └── v1/          # Version 1 API routers (predict, health, models)
│   ├── core/            # Configuration, logging filters, security headers, rate limiting
│   ├── schemas/         # Pydantic v2 request/response and explanation schemas
│   ├── services/        # MLService connecting to packaged ModelArtifacts
│   └── dependencies/    # FastAPI dependency injections & RBAC/BOLA authorization
└── README.md
```

## Local Development

Start the backend server using `uvicorn`:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger API documentation is accessible at `http://localhost:8000/docs`.
