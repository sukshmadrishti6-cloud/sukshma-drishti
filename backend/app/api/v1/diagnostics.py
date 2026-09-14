"""Diagnostics and Integrity verification endpoint for SukshmaDrishti Model Registry."""
import json
import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter
from app.services.ml_service import CANONICAL_MODELS

router = APIRouter()
logger = logging.getLogger(__name__)

BENCHMARK_FILE = Path("data/processed/benchmark_results.json")
OPTIMIZATION_FILE = Path("data/processed/optimization_results.json")

EXPECTED_COUNTS = {
    "diabetes": {"total": 23, "classical": 17, "quantum": 6},
    "cardiovascular": {"total": 22, "classical": 16, "quantum": 6},
    "cancer": {"total": 22, "classical": 16, "quantum": 6},
    "total": 67,
}


@router.get("")
@router.get("/")
def get_registry_diagnostics() -> dict[str, Any]:
    """Returns real-time diagnostics and model integrity verification across all 3 diseases."""
    bench_data = {}
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception:
            bench_data = {}

    eval_by_id = {}
    for r in bench_data.get("results", []):
        r_id = r.get("model_id")
        if r_id:
            eval_by_id[r_id] = r
    for m in bench_data.get("models", []):
        m_id = m.get("model_id")
        if m_id and m_id not in eval_by_id:
            eval_by_id[m_id] = m

    disease_stats: dict[str, dict[str, Any]] = {
        "diabetes": {"total_registered": 0, "classical": 0, "quantum": 0, "evaluated": 0, "on_disk": 0, "models": []},
        "cardiovascular": {"total_registered": 0, "classical": 0, "quantum": 0, "evaluated": 0, "on_disk": 0, "models": []},
        "cancer": {"total_registered": 0, "classical": 0, "quantum": 0, "evaluated": 0, "on_disk": 0, "models": []},
    }

    all_models_status = []
    for model_id, m_def in CANONICAL_MODELS.items():
        dis = m_def["disease"]
        is_q = m_def["family"] == "quantum" or m_def.get("quantum", False)
        art_path = Path(m_def["artifact_path"])
        exists_disk = art_path.exists()
        eval_item = eval_by_id.get(model_id, {})
        has_eval = bool(eval_item.get("metrics", {}).get("accuracy") is not None)

        if dis in disease_stats:
            disease_stats[dis]["total_registered"] += 1
            if is_q:
                disease_stats[dis]["quantum"] += 1
            else:
                disease_stats[dis]["classical"] += 1
            if has_eval:
                disease_stats[dis]["evaluated"] += 1
            if exists_disk:
                disease_stats[dis]["on_disk"] += 1
            disease_stats[dis]["models"].append(model_id)

        all_models_status.append({
            "model_id": model_id,
            "disease": dis,
            "family": m_def["family"],
            "display_name": m_def["display_name"],
            "exists_on_disk": exists_disk,
            "evaluated": has_eval,
            "production": m_def.get("production", False),
        })

    all_intact = (
        disease_stats["diabetes"]["total_registered"] == EXPECTED_COUNTS["diabetes"]["total"]
        and disease_stats["cardiovascular"]["total_registered"] == EXPECTED_COUNTS["cardiovascular"]["total"]
        and disease_stats["cancer"]["total_registered"] == EXPECTED_COUNTS["cancer"]["total"]
        and len(CANONICAL_MODELS) == EXPECTED_COUNTS["total"]
    )

    return {
        "status": "Healthy" if all_intact else "Degraded",
        "integrity_verified": all_intact,
        "total_canonical_models": len(CANONICAL_MODELS),
        "expected_total_models": EXPECTED_COUNTS["total"],
        "diseases": disease_stats,
        "benchmark_file": {
            "path": str(BENCHMARK_FILE),
            "exists": BENCHMARK_FILE.exists(),
            "results_count": len(bench_data.get("results", [])),
            "primary_models_count": len(bench_data.get("models", [])),
        },
        "models": all_models_status,
    }
