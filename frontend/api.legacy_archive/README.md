# Legacy Serverless API Archive

This directory contains deprecated serverless endpoints previously used in early prototypes.
They have been isolated from production for the following security and architectural reasons:
1. They referenced `SUPABASE_SERVICE_ROLE_KEY` directly.
2. They configured wildcard CORS (`Access-Control-Allow-Origin: *`).
3. SukshmaDrishti now uses a single, authoritative FastAPI backend at `/api/v1` (`backend/app`).

The React client communicates exclusively through `frontend/src/api/client.ts` to the FastAPI backend.
Do not deploy or serve this directory in production.