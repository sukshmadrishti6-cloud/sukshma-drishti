import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import APIRouter
from app.schemas.api import BenchmarkSummary
from app.services.ml_service import CANONICAL_MODELS

router = APIRouter()
logger = logging.getLogger(__name__)

BENCHMARK_FILE = Path("data/processed/benchmark_results.json")
OPTIMIZATION_FILE = Path("data/processed/optimization_results.json")


def _format_calibration(calib_obj: Any, metrics_dict: dict[str, Any]) -> dict[str, Any]:
    if isinstance(calib_obj, dict) and "status" in calib_obj:
        return calib_obj
    if "brier_score" in metrics_dict and metrics_dict["brier_score"] is not None:
        return {"status": "available", "brier_score": metrics_dict["brier_score"]}
    return {"status": "available" if "brier_score" in metrics_dict else "unsupported"}


BENCHMARK_RUN_STATE: dict[str, Any] = {
    "status": "idle",
    "benchmark_id": None,
    "message": "No benchmark run in progress.",
    "started_at": None,
    "completed_at": None,
    "error": None,
}


def _execute_benchmark_background(b_id: str, disease: str | None = None):
    global BENCHMARK_RUN_STATE
    try:
        target_dis = (disease or "all").lower().strip()
        BENCHMARK_RUN_STATE["status"] = "running"
        BENCHMARK_RUN_STATE["message"] = f"Executing additive benchmark suite for {target_dis} on 80/20 holdout split..."

        from scripts.populate_all_canonical_evaluations import populate_all_evaluations
        populate_all_evaluations()

        BENCHMARK_RUN_STATE["status"] = "completed"
        BENCHMARK_RUN_STATE["completed_at"] = datetime.now().isoformat()
        BENCHMARK_RUN_STATE["message"] = f"Official benchmark suite execution for {target_dis} completed successfully without model loss."
    except Exception as e:
        logger.error(f"Background benchmark execution failed: {e!s}", exc_info=True)
        BENCHMARK_RUN_STATE["status"] = "failed"
        BENCHMARK_RUN_STATE["error"] = str(e)
        BENCHMARK_RUN_STATE["message"] = f"Benchmark execution failed: {e!s}"


@router.post("/run")
def run_classical_baselines(disease: str | None = None) -> dict[str, Any]:
    """Triggers execution of the unified benchmark suite in the background."""
    global BENCHMARK_RUN_STATE
    if BENCHMARK_RUN_STATE["status"] == "running":
        return {
            "status": "running",
            "benchmark_id": BENCHMARK_RUN_STATE["benchmark_id"],
            "message": "A benchmark run is already in progress.",
            "started_at": BENCHMARK_RUN_STATE["started_at"],
        }

    b_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    BENCHMARK_RUN_STATE["status"] = "running"
    BENCHMARK_RUN_STATE["benchmark_id"] = b_id
    BENCHMARK_RUN_STATE["started_at"] = datetime.now().isoformat()
    BENCHMARK_RUN_STATE["completed_at"] = None
    BENCHMARK_RUN_STATE["error"] = None
    BENCHMARK_RUN_STATE["message"] = "Starting benchmark execution..."

    t = threading.Thread(target=_execute_benchmark_background, args=(b_id, disease), daemon=True)
    t.start()

    return {
        "status": "running",
        "benchmark_id": b_id,
        "message": f"Unified benchmark execution started for {disease or 'all'}.",
        "started_at": BENCHMARK_RUN_STATE["started_at"],
    }


@router.get("/status")
def get_classical_baselines_status() -> dict[str, Any]:
    """Returns the current execution status of the benchmark runner."""
    return BENCHMARK_RUN_STATE


@router.get("", response_model=BenchmarkSummary)
@router.get("/", response_model=BenchmarkSummary)
@router.get("/{disease}", response_model=BenchmarkSummary)
def get_classical_baselines(disease: str | None = None) -> BenchmarkSummary:
    """Returns classical benchmark evaluation metrics from official benchmark results,
    optionally filtered by disease ('diabetes', 'cardiovascular', 'cancer', or 'all').
    """
    bench_data = {}
    if BENCHMARK_FILE.exists() and BENCHMARK_FILE.is_file():
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading benchmark results: {e!s}")

    opt_data = {}
    if OPTIMIZATION_FILE.exists() and OPTIMIZATION_FILE.is_file():
        try:
            with open(OPTIMIZATION_FILE, "r", encoding="utf-8") as f:
                opt_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading optimization results: {e!s}")

    target_disease = disease.lower().strip() if disease and disease != "all" else None

    # Index evaluations
    eval_by_id: dict[str, dict[str, Any]] = {}
    for r in bench_data.get("results", []):
        r_id = r.get("model_id")
        if r_id:
            eval_by_id[r_id] = r
    for m in bench_data.get("models", []):
        m_id = m.get("model_id")
        if m_id and m_id not in eval_by_id:
            eval_by_id[m_id] = m
    for om in opt_data.get("optimized_models", []):
        om_id = om.get("model_id")
        if om_id and (om_id not in eval_by_id or not eval_by_id[om_id].get("metrics")):
            eval_by_id[om_id] = om

    results: list[dict[str, Any]] = []

    for model_id, m_def in CANONICAL_MODELS.items():
        if m_def["family"] == "quantum" or m_def.get("quantum", False):
            continue  # Classical only

        m_disease = m_def["disease"]
        if target_disease and m_disease != target_disease:
            continue

        eval_item = eval_by_id.get(model_id, {})
        metrics = eval_item.get("metrics") or {}
        is_evaluated = bool(metrics and metrics.get("accuracy") is not None)

        results.append({
            "model_id": model_id,
            "model_name": m_def["display_name"],
            "disease": m_disease,
            "family": "classical",
            "is_quantum": False,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "brier_score": metrics.get("brier_score"),
            "log_loss": metrics.get("log_loss"),
            "metrics": metrics if is_evaluated else None,
            "timings": eval_item.get("timings", {}),
            "features_used": m_def.get("classical_feature_count") or m_def.get("raw_feature_count"),
            "confusion_matrix": eval_item.get("confusion_matrix") if is_evaluated else None,
            "curves": eval_item.get("curves", {}),
            "calibration": _format_calibration(eval_item.get("calibration"), metrics),
            "evaluated": is_evaluated,
            "status": "Evaluated" if is_evaluated else "Pending Evaluation",
        })

    dis_str = f" for {target_disease.capitalize()}" if target_disease else " across all diseases"
    return BenchmarkSummary(
        status="Available",
        message=f"Official classical baseline metrics loaded ({len(results)} classical models{dis_str}).",
        available=True,
        results=results,
    )



