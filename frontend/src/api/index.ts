import { apiClient } from './client';
import {
  InferenceRequest,
  InferenceResponse,
  QuantumExperiment,
  ModelsResponse,
  PipelineMetadata,
  BenchmarkSummary,
  EvaluationResponse,
  User,
  AuthToken,
  RegisterRequest,
  LoginRequest,
  DiseaseListResponse,
  DiseaseDetail,
  PredictionHistoryListResponse,
  PredictionHistoryDetail,
  RecommendationListResponse,
  RecommendationResponse,
  ReportScanResponse,
} from '../types';

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  backend_engine?: string;
  models_available: Array<{
    artifact_id: string;
    name: string;
    family: string;
    status: string;
  }>;
}

export interface ExperimentCreateRequest {
  title: string;
  dataset_name: string;
  dataset_hash: string;
  qubit_count: number;
  feature_map: string;
  ansatz: string;
  optimizer: string;
  shots: number;
  status?: string;
}

export interface ExperimentStatus {
  id: string;
  status: string;
  message: string;
}

export type ExperimentResponse = ExperimentStatus;

export const api = {
  // Auth API
  register: (data: RegisterRequest) => apiClient.post<User>('/auth/register', data),
  login: (data: LoginRequest) => apiClient.post<AuthToken>('/auth/login', data),
  getMe: () => apiClient.get<User>('/auth/me'),

  getHealth: () => apiClient.get<HealthResponse>('/health'),
  
  getDiseases: () => apiClient.get<DiseaseListResponse>('/diseases'),
  getDiseaseDetail: (id: string) => apiClient.get<DiseaseDetail>(`/diseases/${id}`),
  getModels: (disease?: string) => apiClient.get<ModelsResponse>(`/models${disease ? `?disease=${disease}` : ''}`),
  getPipelineMetadata: (disease?: string) => apiClient.get<PipelineMetadata>(`/pipeline${disease ? `?disease=${disease}` : ''}`),
  getClassicalBaselines: (disease?: string) => apiClient.get<BenchmarkSummary>(`/baselines${disease ? `?disease=${disease}` : ''}`),
  getEvaluation: (disease: string) => apiClient.get<EvaluationResponse>(`/evaluation/${disease}`),
  runBaselines: (disease?: string) => apiClient.post<{ status: string; benchmark_id: string; message: string }>(`/baselines/run${disease ? `?disease=${disease}` : ''}`, {}),
  getBaselinesStatus: () => apiClient.get<{ status: string; benchmark_id: string | null; message: string; started_at: string | null; completed_at: string | null; error: string | null }>('/baselines/status'),
  getHybridComparison: (disease?: string) => apiClient.get<BenchmarkSummary>(`/comparison${disease ? `?disease=${disease}` : ''}`),
  getGlobalExplanation: (modelId: string) => apiClient.get<any>(`/models/${modelId}/explain/global`),
  
  // Medical Report Scanner API
  scanMedicalReport: (file: File, disease: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('disease', disease);
    return apiClient.upload<ReportScanResponse>('/report/scan', formData);
  },

  runInference: (request: { features: any, model_id: string, explain: boolean, evaluation_context?: string }) => 
    apiClient.post<InferenceResponse>('/predict', request),

  runDiseaseInference: (disease: string, request: { features: any, model_id: string, explain?: boolean, evaluation_context?: string }) =>
    apiClient.post<InferenceResponse>(`/predict/${disease}`, request),

  // Prediction History API
  getHistory: (page = 1, pageSize = 20, disease?: string) =>
    apiClient.get<PredictionHistoryListResponse>(`/history?page=${page}&page_size=${pageSize}${disease ? `&disease=${disease}` : ''}`),
  getHistoryDetail: (id: string) =>
    apiClient.get<PredictionHistoryDetail>(`/history/${id}`),
  deleteHistory: (id: string) =>
    apiClient.delete<{ status: string; id: string }>(`/history/${id}`),

  // Recommendations API
  getRecommendations: (page = 1, pageSize = 20, disease?: string) =>
    apiClient.get<RecommendationListResponse>(`/recommendations?page=${page}&page_size=${pageSize}${disease ? `&disease=${disease}` : ''}`),
  getRecommendationDetail: (predictionId: string) =>
    apiClient.get<RecommendationResponse>(`/recommendations/${predictionId}`),
    
  createExperiment: (request: ExperimentCreateRequest) => 
    apiClient.post<ExperimentStatus>('/experiments/run', request),
    
  getExperiment: (id: string) =>
    apiClient.get<QuantumExperiment>(`/experiments/${id}`),
    
  getExperimentStatus: (id: string) =>
    apiClient.get<ExperimentStatus>(`/experiments/${id}/status`),

  getQuantumCapabilities: (disease?: string) =>
    apiClient.get<import('../types').QuantumCapability[]>(`/experiments/capabilities${disease ? `?disease=${disease}` : ''}`),

  getLatestExperiment: (qubitCount: number = 4, modelType?: string, disease?: string) =>
    apiClient.get<QuantumExperiment | null>(`/experiments/latest?qubit_count=${qubitCount}${modelType ? `&model_type=${modelType}` : ''}${disease ? `&disease=${disease}` : ''}`),
};
