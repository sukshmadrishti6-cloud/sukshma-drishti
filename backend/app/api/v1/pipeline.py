from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.api import PipelineMetadata, PipelineStage

router = APIRouter()

DISEASE_CONFIGS = {
    "diabetes": {
        "dataset_name": "Pima Indians Diabetes (Approved)",
        "dataset_version": "v3.0 (Real Raw)",
        "paths": [Path("data/raw/pima_diabetes.csv")],
        "expected_rows": 768,
        "raw_features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "target": "Outcome",
        "engineered_count": 3,
        "stages": [
            PipelineStage(stage_name="Missing Value Imputation", method="Median (Train-only)"),
            PipelineStage(stage_name="Outlier Handling", method="IQR Bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (Train-only)"),
            PipelineStage(stage_name="Feature Engineering", method="Glucose_BMI, Insulin_Glucose, AgeGroup"),
            PipelineStage(stage_name="Scaling", method="StandardScaler (Train-only)"),
            PipelineStage(stage_name="Quantum Reduction", method="SelectKBest(mutual_info_classif, k=4)"),
            PipelineStage(stage_name="Quantum Angle Scaling", method="MinMax [-pi, pi]"),
        ],
    },
    "cardiovascular": {
        "dataset_name": "UCI Heart Disease (Approved)",
        "dataset_version": "v1.0 (Real Raw)",
        "paths": [Path("data/raw/cardiovascular.csv"), Path("data/cardiovascular/cardiovascular.csv")],
        "expected_rows": 303,
        "raw_features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "target": "target",
        "engineered_count": 0,
        "stages": [
            PipelineStage(stage_name="Missing Value Imputation", method="Median Imputation (Train-only)"),
            PipelineStage(stage_name="Outlier Clipping", method="IQR Winsorization (Train-only)"),
            PipelineStage(stage_name="Categorical/Ordinal Encoding", method="Domain Value Encoding"),
            PipelineStage(stage_name="Feature Scaling", method="StandardScaler (Train-only)"),
            PipelineStage(stage_name="Quantum Reduction", method="SelectKBest(mutual_info_classif, k=4)"),
            PipelineStage(stage_name="Quantum Angle Encoding", method="MinMax [-pi, pi]"),
        ],
    },
    "cancer": {
        "dataset_name": "Wisconsin Breast Cancer Diagnostic (Approved)",
        "dataset_version": "v1.0 (Real Raw)",
        "paths": [Path("data/raw/cancer.csv"), Path("data/cancer/cancer.csv")],
        "expected_rows": 569,
        "raw_features": [
            "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
            "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
            "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error",
            "area_error", "smoothness_error", "compactness_error", "concavity_error",
            "concave_points_error", "symmetry_error", "fractal_dimension_error",
            "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
            "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
            "worst_fractal_dimension"
        ],
        "target": "target",
        "engineered_count": 0,
        "stages": [
            PipelineStage(stage_name="Missing Value Imputation", method="Median Imputation (Train-only)"),
            PipelineStage(stage_name="Multicollinearity Handling", method="Correlation Analysis & Standardization"),
            PipelineStage(stage_name="Feature Scaling", method="StandardScaler (Train-only)"),
            PipelineStage(stage_name="Quantum Reduction", method="SelectKBest(mutual_info_classif, k=4)"),
            PipelineStage(stage_name="Quantum Angle Encoding", method="MinMax [-pi, pi]"),
        ],
    },
}


def _resolve_pipeline_for_disease(dis_key: str) -> PipelineMetadata:
    cfg = DISEASE_CONFIGS.get(dis_key)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{dis_key}' not recognized. Supported: diabetes, cardiovascular, cancer.",
        )

    # Check file exists
    data_file = next((p for p in cfg["paths"] if p.exists() and p.is_file()), None)

    status_str = "Awaiting Data"
    sample_count = None
    class_dist = None
    imbalance = None

    if data_file:
        try:
            df = pd.read_csv(data_file)
            target_col = cfg["target"]
            if target_col not in df.columns and "Outcome" in df.columns:
                target_col = "Outcome"

            sample_count = len(df)
            if target_col in df.columns:
                val_counts = df[target_col].value_counts().to_dict()
                class_dist = {str(k): int(v) for k, v in val_counts.items()}
                c0 = class_dist.get("0", 0)
                c1 = class_dist.get("1", 0)
                if c0 > 0 and c1 > 0:
                    ratio = max(c0, c1) / min(c0, c1)
                    pct0 = (c0 / sample_count) * 100
                    pct1 = (c1 / sample_count) * 100
                    imbalance = f"{ratio:.2f} : 1 ({pct0:.1f}% Class 0 / {pct1:.1f}% Class 1)"


            status_str = "Ready" if sample_count == cfg["expected_rows"] else "Schema Validated"
        except Exception:
            status_str = "Validation Error"

    raw_cnt = len(cfg["raw_features"])
    eng_cnt = cfg["engineered_count"]

    return PipelineMetadata(
        disease=dis_key,
        dataset_name=cfg["dataset_name"],
        dataset_version=cfg["dataset_version"],
        status=status_str,
        sample_count=sample_count or cfg["expected_rows"],
        class_distribution=class_dist,
        imbalance_ratio=imbalance,
        raw_feature_count=raw_cnt,
        engineered_feature_count=eng_cnt,
        total_feature_count=raw_cnt + eng_cnt,
        feature_names=cfg["raw_features"],
        target_variable=cfg["target"],
        split_strategy="80/20 Stratified (Seed 42)",
        pipeline_version="v2.0",
        stages=cfg["stages"],
    )


@router.get("", response_model=PipelineMetadata)
@router.get("/", response_model=PipelineMetadata)
def get_pipeline_metadata(disease: str = Query("diabetes", description="Target disease (diabetes, cardiovascular, cancer)")) -> PipelineMetadata:
    """Returns pipeline execution metadata and dynamically verifies dataset readiness for specified disease."""
    dis_key = (disease or "diabetes").lower().strip()
    return _resolve_pipeline_for_disease(dis_key)


@router.get("/{disease}", response_model=PipelineMetadata)
def get_pipeline_metadata_by_path(disease: str) -> PipelineMetadata:
    """Returns pipeline execution metadata for a specified disease via path parameter."""
    dis_key = (disease or "diabetes").lower().strip()
    return _resolve_pipeline_for_disease(dis_key)

