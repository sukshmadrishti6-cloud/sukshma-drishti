from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    backend_engine: str
    models_available: list[dict[str, Any]] | None = None

class APIError(BaseModel):
    code: str
    message: str
    request_id: str | None = None

class ModelSummary(BaseModel):
    model_id: str
    disease: str = "diabetes"
    display_name: str
    model_family: str
    model_type: str | None = None
    version: str
    artifact_id: str | None = None
    artifact_version: str = "v3.0"
    is_quantum: bool = False
    quantum: bool = False
    qubit_count: int | None = None
    raw_feature_count: int = 8
    classical_feature_count: int | None = 11
    quantum_feature_count: int | None = None
    artifact_path: str | None = None
    manifest_path: str | None = None
    status: str
    trained: bool = True
    evaluated: bool = True
    production: bool = False
    experimental: bool = False
    pipeline_version: str | None = "v2.0"
    dataset_version: str | None = None
    probability_support: bool = False
    explainability_support: bool = False
    evaluation_support: bool = True
    score_semantics: str | None = None
    selection_reason: str | None = None
    selection_metric: str | None = None

class ModelsResponse(BaseModel):
    models: list[ModelSummary]
    total: int = 0
    disease: str | None = None

class EvaluationModel(BaseModel):
    model_id: str
    model_name: str
    version: str | None = None
    disease: str
    family: str
    is_quantum: bool = False
    status: str
    evaluated: bool = True
    production: bool = False
    features_used: int | None = None
    qubits: int | None = None
    metrics: dict[str, Any] | None = None
    confusion_matrix: dict[str, Any] | None = None
    curves: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    score_semantics: str | None = None
    optimal_hyperparameters: dict[str, Any] | None = None
    circuit_specs: dict[str, Any] | None = None

class EvaluationResponse(BaseModel):
    disease: str
    total_models: int
    evaluated_models_count: int
    circuit_only_models_count: int
    evaluation_protocol: dict[str, Any]
    models: list[EvaluationModel]


class DatasetMetadata(BaseModel):
    name: str
    version: str
    hash: str
    status: str
    description: str | None = None

class PipelineStage(BaseModel):
    stage_name: str
    method: str
    description: str | None = None

class PipelineMetadata(BaseModel):
    disease: str = "diabetes"
    dataset_name: str = "Pima Indians Diabetes (Approved)"
    dataset_version: str
    status: str
    sample_count: int | None = None
    class_distribution: dict[str, int] | None = None
    imbalance_ratio: str | None = None
    raw_feature_count: int
    engineered_feature_count: int
    total_feature_count: int
    feature_names: list[str] | None = None
    target_variable: str
    split_strategy: str
    stages: list[PipelineStage]
    pipeline_version: str

class BenchmarkResult(BaseModel):
    dataset_version: str
    pipeline_version: str
    model_id: str
    model_version: str
    metrics: dict[str, float]
    timestamp: str
    experiment_id: str | None = None
    is_official: bool

class BenchmarkSummary(BaseModel):
    status: str
    message: str
    available: bool
    results: list[dict[str, Any]] = []

class QuantumCapability(BaseModel):
    qubit_count: int
    model_id: str | None = None
    trained: bool
    evaluated: bool
    circuit_generation_supported: bool
    inference_supported: bool
    benchmark_supported: bool
    status: str  # 'evaluated' | 'circuit_only' | 'unavailable'
    label: str
    description: str
    limitations: list[str] = Field(default_factory=list)

class ExperimentCreateRequest(BaseModel):
    title: str
    disease: str | None = "diabetes"
    dataset_name: str | None = None
    dataset_hash: str
    qubit_count: int
    feature_map: str
    ansatz: str
    optimizer: str
    shots: int
    status: str | None = None
    config: dict[str, Any] | None = None

class QuantumExperiment(BaseModel):
    id: str
    title: str
    disease: str | None = "diabetes"
    dataset_hash: str
    qubit_count: int
    feature_map: str
    ansatz: str
    optimizer: str
    shots: int
    status: str = "queued"
    execution_backend: str = "AerSimulator (local)"
    execution_status: str = "queued"
    evaluation_status: str = "unavailable"
    model_status: str = "not_trained"
    model_id: str | None = None
    evaluation_reason: str | None = None
    limitations: list[str] = Field(default_factory=list)
    circuit_text: str | None = None
    circuit_depth: int | None = None
    num_parameters: int | None = None
    gate_counts: dict[str, int] | None = None
    metrics: dict[str, Any] | None = None
    confusion_matrix: dict[str, Any] | None = None
    curves: dict[str, Any] | None = None
    timings: dict[str, float] | None = None
    scaling: list[dict[str, Any]] | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    config: dict[str, Any] | None = None

class ExperimentStatus(BaseModel):
    id: str
    status: str # queued, running, completed, failed, cancelled, unavailable
    message: str | None = None

class ExplanationContribution(BaseModel):
    name: str
    value: float
    contribution: float
    direction: str # positive, negative, neutral
    normalized_weight: float

class ExplanationResult(BaseModel):
    model_name: str
    model_version: str
    explanation_method: str
    predicted_class: int | None = None
    contributions: list[ExplanationContribution] = []
    inference_context: str | None = None
    limitations: list[str] = []
    status: str # available, unavailable, partial, unsupported

class DiseaseSummary(BaseModel):
    id: str
    name: str
    status: str  # 'available' | 'coming_soon'
    description: str
    category: str
    version: str = "v1.0"


class DiseaseDetail(DiseaseSummary):
    supported_models: list[dict[str, Any]] = []
    input_features_count: int | None = None
    feature_names: list[str] | None = None


class DiseaseListResponse(BaseModel):
    items: list[DiseaseSummary]
    total: int


class CardiovascularInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., ge=50, le=600, description="Serum cholestoral in mg/dl")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: float = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: float = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-4) colored by flourosopy")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)")


class CancerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(..., ge=0.0, le=50.0, description="Mean of distances from center to points on the perimeter")
    mean_texture: float = Field(..., ge=0.0, le=60.0, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., ge=0.0, le=300.0, description="Mean size of the core tumor perimeter")
    mean_area: float = Field(..., ge=0.0, le=3000.0, description="Mean area of the core tumor")
    mean_smoothness: float = Field(..., ge=0.0, le=1.0, description="Mean of local variation in radius lengths")
    mean_compactness: float = Field(..., ge=0.0, le=1.0, description="Mean of perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., ge=0.0, le=1.0, description="Mean of severity of concave portions of the contour")
    mean_concave_points: float = Field(..., ge=0.0, le=1.0, description="Mean for number of concave portions of the contour")
    mean_symmetry: float = Field(..., ge=0.0, le=1.0, description="Mean symmetry of cell nucleus")
    mean_fractal_dimension: float = Field(..., ge=0.0, le=1.0, description="Mean coastline approximation - 1")
    radius_error: float = Field(..., ge=0.0, le=10.0, description="Standard error for the mean of distances")
    texture_error: float = Field(..., ge=0.0, le=10.0, description="Standard error for gray-scale values")
    perimeter_error: float = Field(..., ge=0.0, le=50.0, description="Standard error for perimeter")
    area_error: float = Field(..., ge=0.0, le=600.0, description="Standard error for area")
    smoothness_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for local variation")
    compactness_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for compactness")
    concavity_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for concavity")
    concave_points_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for concave points")
    symmetry_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for symmetry")
    fractal_dimension_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for fractal dimension")
    worst_radius: float = Field(..., ge=0.0, le=60.0, description="Worst or largest mean value for radius")
    worst_texture: float = Field(..., ge=0.0, le=70.0, description="Worst or largest mean value for texture")
    worst_perimeter: float = Field(..., ge=0.0, le=350.0, description="Worst or largest mean value for perimeter")
    worst_area: float = Field(..., ge=0.0, le=5000.0, description="Worst or largest mean value for area")
    worst_smoothness: float = Field(..., ge=0.0, le=1.0, description="Worst or largest mean value for smoothness")
    worst_compactness: float = Field(..., ge=0.0, le=2.0, description="Worst or largest mean value for compactness")
    worst_concavity: float = Field(..., ge=0.0, le=2.0, description="Worst or largest mean value for concavity")
    worst_concave_points: float = Field(..., ge=0.0, le=1.0, description="Worst or largest mean value for concave points")
    worst_symmetry: float = Field(..., ge=0.0, le=1.0, description="Worst or largest mean value for symmetry")
    worst_fractal_dimension: float = Field(..., ge=0.0, le=1.0, description="Worst or largest mean value for fractal dimension")




class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] | list[float]
    model_id: str
    explain: bool = False
    evaluation_context: str | None = Field(default="single_sample", description="Evaluation context: single_sample or holdout_reference")
    disease: str | None = Field(default="diabetes", description="Target disease identifier (default: diabetes)")

class AuditMetadata(BaseModel):
    timestamp: str
    request_id: str | None = None
    inference_time_ms: float

class RecommendationFactor(BaseModel):
    name: str
    value: float | None = None
    contribution: float | None = None
    direction: str | None = None
    summary: str


class RecommendationResponse(BaseModel):
    id: str | None = None
    prediction_record_id: str | None = None
    disease: str
    summary: str
    priority_actions: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)
    professional_follow_up: str
    when_to_seek_care: str
    factors: list[RecommendationFactor] = Field(default_factory=list)
    disclaimer: str
    rule_set_version: str
    created_at: str | None = None


class RecommendationListResponse(BaseModel):
    items: list[RecommendationResponse]
    total: int
    page: int
    page_size: int


class InferenceResponse(BaseModel):
    inference_id: str | None = None
    disease: str = "diabetes"
    predicted_class: int
    probability: float | None = None
    decision_score: float | None = None
    score_semantics: str | None = None
    model_output: str | None = None
    model_used: str
    model_family: str
    artifact_id: str
    artifact_version: str = "v3.0"
    pipeline_version: str = "v2.0"
    explanation: ExplanationResult | None = None
    recommendation: RecommendationResponse | None = None
    audit: AuditMetadata


# Authentication Schemas
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str = "authenticated"
    created_at: str | None = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# Prediction History Schemas
class PredictionHistoryItem(BaseModel):
    id: str
    disease: str
    model_name: str
    model_version: str
    prediction: int
    risk_score: float | None = None
    risk_category: str | None = None
    score_semantics: str | None = None
    created_at: str


class PredictionHistoryDetail(PredictionHistoryItem):
    input_data: dict[str, Any]
    explanation: ExplanationResult | dict[str, Any] | None = None
    recommendation: RecommendationResponse | dict[str, Any] | str | None = None
    source: str = "user_inference"


class PredictionHistoryListResponse(BaseModel):
    items: list[PredictionHistoryItem]
    total: int
    page: int
    page_size: int



