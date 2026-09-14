import json
import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter
from app.schemas.api import BenchmarkSummary
from app.services.ml_service import CANONICAL_MODELS

router = APIRouter()
logger = logging.getLogger(__name__)

BENCHMARK_FILE = Path("data/processed/benchmark_results.json")
OPTIMIZATION_FILE = Path("data/processed/optimization_results.json")

PRODUCTION_WINNERS = {
    "diabetes_rf_opt_v1",
    "cardiovascular_extra_trees_opt_v1",
    "cancer_stacking_opt_v1",
}


def _determine_category(model_id: str, family: str) -> str:
    """Categorizes model into Baseline, Advanced, Ensemble, Optimized, or Quantum."""
    if family == "quantum" or "quantum" in model_id or "qsvc" in model_id or "vqc" in model_id:
        return "Quantum ML"
    if "opt" in model_id:
        return "Optimized Classical"
    if "voting" in model_id or "stacking" in model_id or "ensemble" in model_id:
        return "Ensemble Classical"
    if "extra_trees" in model_id or "knn" in model_id:
        return "Advanced Classical"
    return "Baseline Classical"


def _format_model_entry(m_id: str, m_def: dict[str, Any], eval_data: dict[str, Any]) -> dict[str, Any]:
    metrics = eval_data.get("metrics") or {}
    m_disease = m_def["disease"]
    m_family = m_def["family"]
    is_q = m_family == "quantum" or m_def.get("quantum", False)
    is_evaluated = bool(metrics and metrics.get("accuracy") is not None)

    timings = eval_data.get("timings") or {}
    inf_time = timings.get("inference_time_ms")
    qubits = m_def.get("qubit_count")

    if not is_evaluated:
        cost_str = f"Circuit Sim ({qubits}Q)" if qubits else "Circuit Sim"
    elif inf_time is not None:
        cost_str = f"{inf_time:.2f} ms"
        if is_q and qubits:
            cost_str += f" ({qubits}Q sim)"
    else:
        cost_str = "—"

    category = _determine_category(m_id, m_family)
    is_prod = m_id in PRODUCTION_WINNERS

    return {
        "model_id": m_id,
        "model_name": m_def["display_name"],
        "family": m_family,
        "is_quantum": is_q,
        "disease": m_disease,
        "category": category,
        "production": is_prod,
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
        "computational_cost": cost_str,
        "metrics": metrics if is_evaluated else None,
        "timings": timings,
        "features_used": m_def.get("classical_feature_count") or m_def.get("raw_feature_count"),
        "qubits": qubits,
        "confusion_matrix": eval_data.get("confusion_matrix") if is_evaluated else None,
        "evaluated": is_evaluated,
        "status": "Evaluated" if is_evaluated else "Pending Evaluation",
    }


@router.get("", response_model=BenchmarkSummary)
@router.get("/", response_model=BenchmarkSummary)
@router.get("/{disease}", response_model=BenchmarkSummary)
def get_hybrid_comparison(disease: str | None = None) -> BenchmarkSummary:
    """Returns quantum vs classical comparison data from official benchmark and optimization results,
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

    # Build evaluation index
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
        m_disease = m_def["disease"]
        if target_disease and m_disease != target_disease:
            continue
        eval_item = eval_by_id.get(model_id, {})
        results.append(_format_model_entry(model_id, m_def, eval_item))

    dis_str = f" for {target_disease.capitalize()}" if target_disease else " across all diseases"
    return BenchmarkSummary(
        status="Available",
        message=f"Hybrid comparison matrix loaded ({len(results)} models{dis_str}).",
        available=True,
        results=results,
    )


