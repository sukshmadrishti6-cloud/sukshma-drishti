export type TabType = 
  | 'overview' 
  | 'diseases'
  | 'history'
  | 'recommendations'
  | 'data-pipeline' 
  | 'classical-baselines' 
  | 'quantum-lab' 
  | 'hybrid-comparison' 
  | 'explainability' 
  | 'inference';

export interface RecommendationFactor {
  name: string;
  value?: number | null;
  contribution?: number | null;
  direction?: 'positive' | 'negative' | 'neutral' | string;
  summary: string;
}

export interface RecommendationResponse {
  id?: string | null;
  prediction_record_id?: string | null;
  disease: string;
  summary: string;
  priority_actions: string[];
  monitoring: string[];
  professional_follow_up: string;
  when_to_seek_care: string;
  factors: RecommendationFactor[];
  disclaimer: string;
  rule_set_version: string;
  created_at?: string | null;
}

export interface RecommendationListResponse {
  items: RecommendationResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface DiseaseSummary {
  id: string;
  name: string;
  status: 'available' | 'coming_soon' | string;
  description: string;
  category: string;
  version: string;
}

export interface DiseaseDetail extends DiseaseSummary {
  supported_models: Array<{
    id: string;
    name: string;
    version: string;
    is_quantum: boolean;
    qubit_count?: number | null;
  }>;
  input_features_count?: number | null;
  feature_names?: string[] | null;
}

export interface DiseaseListResponse {
  items: DiseaseSummary[];
  total: number;
}

export interface ExtractedFeatureItem {
  name: string;
  value: number;
  unit?: string | null;
  confidence: number;
  confidence_level: 'high' | 'medium' | 'low';
  original_name: string;
  original_value: number | string;
  original_unit?: string | null;
  source_snippet?: string | null;
  conversion_applied?: boolean;
  requires_verification?: boolean;
}

export interface MissingFeatureItem {
  name: string;
  display_name: string;
  description?: string | null;
  reason: string;
}

export interface ReportScanResponse {
  success: boolean;
  filename: string;
  file_type: string;
  disease: string;
  report_category: string;
  compatible: boolean;
  extracted_features: Record<string, ExtractedFeatureItem>;
  missing_features: string[];
  missing_features_detail: MissingFeatureItem[];
  total_features_required: number;
  total_features_detected: number;
  raw_text_summary?: string | null;
  extraction_method?: string | null;
  text_detected?: boolean;
  extracted_text_length?: number;
  message: string;
  validation_warnings: string[];
}


export interface ModelMetric {
  id: string;
  name: string;
  type: 'classical' | 'quantum' | 'hybrid';
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  rocAuc?: number;
  prAuc?: number;
  sensitivity?: number;
  specificity?: number;
  brier?: number;
  logLoss?: number;
  trainingTimeSec?: number;
  inferenceLatencyMs?: number;
  quantumQubits?: number;
  ansatzDepth?: number;
  entanglementScheme?: string;
  description: string;
  confusionMatrix?: {
    tp: number;
    fp: number;
    fn: number;
    tn: number;
  };
}

export interface InferenceRequest {
  model_id: string;
  features: Record<string, number> | number[];
  explain?: boolean;
  disease?: string;
  evaluation_context?: string;
}

export interface InferenceResponse {
  inference_id?: string | null;
  disease?: string;
  predicted_class: number;
  probability?: number | null;
  decision_score?: number | null;
  score_semantics?: string | null;
  model_output: string;
  model_used: string;
  model_family: string;
  artifact_id: string;
  artifact_version?: string;
  pipeline_version?: string;
  explanation?: ExplanationResult;
  recommendation?: RecommendationResponse;
  audit: {
    timestamp: string;
    inference_time_ms: number;
  };
}

export interface PredictionHistoryItem {
  id: string;
  disease: string;
  model_name: string;
  model_version: string;
  prediction: number;
  risk_score?: number | null;
  risk_category?: string | null;
  score_semantics?: string | null;
  created_at: string;
}

export interface PredictionHistoryDetail extends PredictionHistoryItem {
  input_data: Record<string, number>;
  explanation?: ExplanationResult | Record<string, any> | null;
  recommendation?: RecommendationResponse | Record<string, any> | string | null;
  source?: string;
}

export interface PredictionHistoryListResponse {
  items: PredictionHistoryItem[];
  total: number;
  page: number;
  page_size: number;
}


export interface FeatureContribution {
  name: string;
  value: number;
  contribution: number;
  direction: 'positive' | 'negative' | 'neutral';
  normalized_weight: number;
}

export interface ExplanationResult {
  model_name: string;
  model_version: string;
  explanation_method: string;
  predicted_class: number;
  status: string;
  contributions: FeatureContribution[];
  limitations: string[];
}

export interface DatasetMetadata {
  id: string;
  name: string;
  records: number;
  rawFeatures: number;
  engineeredFeatures: number;
  target: string;
}

export interface QuantumExperiment {
  id: string;
  title: string;
  dataset_hash: string;
  qubit_count: number;
  feature_map: string;
  ansatz: string;
  optimizer: string;
  shots: number;
  status?: string;
  execution_backend?: string;
  circuit_text?: string | null;
  circuit_depth?: number | null;
  num_parameters?: number | null;
  gate_counts?: Record<string, number> | null;
  metrics?: Record<string, any> | null;
  confusion_matrix?: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    matrix?: number[][];
  } | null;
  curves?: {
    roc_curve?: { fpr: number[]; tpr: number[]; auc: number };
    pr_curve?: { precision: number[]; recall: number[]; auc: number };
  } | null;
  timings?: Record<string, number> | null;
  error?: string | null;
  execution_status?: string;
  evaluation_status?: string;
  model_status?: string;
  model_id?: string | null;
  evaluation_reason?: string | null;
  limitations?: string[];
  created_at?: string;
  completed_at?: string;
}

export interface QuantumCapability {
  qubit_count: number;
  model_id?: string | null;
  trained: boolean;
  evaluated: boolean;
  circuit_generation_supported: boolean;
  inference_supported: boolean;
  benchmark_supported: boolean;
  status: 'evaluated' | 'circuit_only' | 'unavailable';
  label: string;
  description: string;
}

export interface GlobalFeatureImportanceItem {
  name: string;
  importance: number;
  normalized_weight: number;
}

export interface GlobalExplanation {
  model_name: string;
  model_family: string;
  method: string;
  status: string;
  features: GlobalFeatureImportanceItem[];
  limitations: string[];
  pipeline_version?: string;
  metadata?: Record<string, any>;
}

export interface BenchmarkRunStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  benchmark_id: string | null;
  message: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}


export interface PipelineStage {
  stage_name: string;
  method: string;
}

export interface PipelineMetadata {
  disease?: string;
  dataset_name?: string;
  dataset_version: string;
  status: string;
  sample_count?: number;
  class_distribution?: Record<string, number>;
  imbalance_ratio?: string;
  raw_feature_count: number;
  engineered_feature_count: number;
  total_feature_count: number;
  feature_names?: string[];
  target_variable: string;
  split_strategy: string;
  pipeline_version: string;
  stages: PipelineStage[];
}


export interface ModelSummary {
  model_id: string;
  display_name: string;
  model_family: string;
  disease?: string;
  version: string;
  is_quantum: boolean;
  qubit_count: number | null;
  artifact_path: string;
  manifest_path: string;
  status: string;
  trained: boolean;
  evaluated: boolean;
  production: boolean;
  experimental: boolean;
  quantum: boolean;
  selection_reason?: string | null;
  selection_metric?: string | null;
  frozen_threshold?: number | null;
  holdout_test_metrics?: Record<string, any> | null;
  score_semantics?: string | null;
  pipeline_version: string | null;
  dataset_version: string | null;
  probability_support: boolean;
  explainability_support: boolean;
}

export interface ModelsResponse {
  total: number;
  disease?: string | null;
  models: ModelSummary[];
}

export interface EvaluationModel {
  model_id: string;
  model_name: string;
  version?: string | null;
  disease: string;
  family: string;
  is_quantum: boolean;
  status: string;
  evaluated: boolean;
  production: boolean;
  features_used?: number | null;
  qubits?: number | null;
  metrics?: Record<string, any> | null;
  confusion_matrix?: Record<string, any> | null;
  curves?: {
    roc_curve?: { fpr: number[]; tpr: number[]; thresholds?: number[]; auc?: number };
    pr_curve?: { precision: number[]; recall: number[]; thresholds?: number[]; auc?: number };
  } | null;
  calibration?: Record<string, any> | null;
  timings?: Record<string, any> | null;
  score_semantics?: string | null;
  optimal_hyperparameters?: Record<string, any> | null;
  circuit_specs?: Record<string, any> | null;
}

export interface EvaluationResponse {
  disease: string;
  total_models: number;
  evaluated_models_count: number;
  circuit_only_models_count: number;
  evaluation_protocol: Record<string, any>;
  models: EvaluationModel[];
}

export interface BenchmarkSummary {
  status: string;
  message: string;
  available: boolean;
  results: any[];
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  created_at?: string | null;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

