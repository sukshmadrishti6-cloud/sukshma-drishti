"""Authoritative Evaluation API endpoint for multi-disease models."""
import json
import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, status
from app.schemas.api import EvaluationModel, EvaluationResponse
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


def _load_benchmark_data() -> dict[str, Any]:
    if not BENCHMARK_FILE.exists():
        return {}
    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load benchmark results: {e!s}")
        return {}


def _load_optimization_data() -> dict[str, Any]:
    if not OPTIMIZATION_FILE.exists():
        return {}
    try:
        with open(OPTIMIZATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load optimization results: {e!s}")
        return {}


def _extract_curves(item: dict[str, Any]) -> dict[str, Any] | None:
    """Extracts standardized ROC and PR curves safely from benchmark or optimization item."""
    curves_dict = item.get("curves") if isinstance(item.get("curves"), dict) else {}
    roc_curve = curves_dict.get("roc_curve") or item.get("roc_curve")
    pr_curve = curves_dict.get("pr_curve") or item.get("pr_curve")
    if roc_curve or pr_curve:
        return {
            "roc_curve": roc_curve,
            "pr_curve": pr_curve,
        }
    return None


@router.get("", response_model=EvaluationResponse)
@router.get("/", response_model=EvaluationResponse)
@router.get("/{disease}", response_model=EvaluationResponse)
def get_disease_evaluation(disease: str = "all") -> EvaluationResponse:
    """Returns evaluation metrics, confusion matrices, performance curves, and calibration diagrams
    for a specified disease ('diabetes', 'cardiovascular', 'cancer', or 'all').

    Authoritatively serves all 67 canonical models (23 Diabetes, 22 Cardio, 22 Cancer).
    """
    disease_clean = (disease or "all").lower().strip()

    valid_diseases = {"diabetes", "cardiovascular", "cancer", "all"}
    if disease_clean not in valid_diseases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{disease}' not recognized. Supported: diabetes, cardiovascular, cancer, all.",
        )

    bench_data = _load_benchmark_data()
    opt_data = _load_optimization_data()

    # Build lookup dictionaries from holdout evaluation files
    eval_by_id: dict[str, dict[str, Any]] = {}

    # 1. Benchmark results
    for r in bench_data.get("results", []):
        r_id = r.get("model_id")
        if r_id:
            eval_by_id[r_id] = r

    # 2. Benchmark models (diabetes canonical list)
    for m in bench_data.get("models", []):
        m_id = m.get("model_id")
        if m_id and m_id not in eval_by_id:
            eval_by_id[m_id] = m

    # 3. Optimization results
    opt_by_id: dict[str, dict[str, Any]] = {}
    for om in opt_data.get("optimized_models", []):
        om_id = om.get("model_id")
        if om_id:
            opt_by_id[om_id] = om
            if om_id not in eval_by_id:
                eval_by_id[om_id] = om

    models_eval: list[EvaluationModel] = []

    # Iterate over canonical models in authoritative order
    for model_id, m_def in CANONICAL_MODELS.items():
        m_disease = m_def["disease"]
        if disease_clean != "all" and disease_clean != m_disease:
            continue

        eval_item = eval_by_id.get(model_id, {})
        opt_item = opt_by_id.get(model_id, {})

        metrics = eval_item.get("metrics") or opt_item.get("metrics")
        cm = eval_item.get("confusion_matrix") or opt_item.get("confusion_matrix")
        curves = _extract_curves(eval_item) or _extract_curves(opt_item)
        calib = eval_item.get("calibration") or opt_item.get("calibration")
        timings = eval_item.get("timings") or opt_item.get("timings") or {}
        circuit_specs = eval_item.get("circuit_specs") or eval_item.get("circuit_metadata")

        is_q = m_def["family"] == "quantum" or m_def.get("quantum", False)
        is_evaluated = bool(metrics and metrics.get("accuracy") is not None)
        is_prod = model_id in PRODUCTION_WINNERS

        if not calib:
            calib = {"status": "available" if (metrics and "brier_score" in metrics) else "unsupported"}

        status_str = "Evaluated" if is_evaluated else "Pending Evaluation"
        if "opt" in model_id:
            status_str = "Optimized & Evaluated" if is_evaluated else "Optimized"

        models_eval.append(EvaluationModel(
            model_id=model_id,
            model_name=m_def["display_name"],
            version=m_def.get("version", "v1.0"),
            disease=m_disease,
            family=m_def["family"],
            is_quantum=is_q,
            status=status_str,
            evaluated=is_evaluated,
            production=is_prod,
            features_used=m_def.get("classical_feature_count") or m_def.get("raw_feature_count"),
            qubits=m_def.get("qubit_count"),
            metrics=metrics,
            confusion_matrix=cm,
            curves=curves,
            calibration=calib,
            timings=timings,
            score_semantics=m_def.get("score_semantics", "calibrated_probability"),
            optimal_hyperparameters=opt_item.get("best_params"),
            circuit_specs=circuit_specs,
        ))

    return EvaluationResponse(
        disease=disease_clean,
        total_models=len(models_eval),
        evaluated_models_count=sum(1 for m in models_eval if m.evaluated),
        circuit_only_models_count=sum(1 for m in models_eval if not m.evaluated),
        evaluation_protocol=bench_data.get("evaluation_protocol", {
            "type": "80/20 Stratified Holdout Test Split",
            "isolated_test_partition": True,
        }),
        models=models_eval,
    )


