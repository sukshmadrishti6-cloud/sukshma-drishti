import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  Play,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  ShieldCheck,
  Sparkles,
  UserCheck,
  FileSearch,
  Edit3,
} from 'lucide-react';
import { DiseaseDetail, InferenceResponse, ModelSummary, TabType } from '../types';
import { api } from '../api';
import { PredictionResultView } from './PredictionResultView';
import { MedicalReportScanner } from './MedicalReportScanner';

interface DiseaseAssessmentFormProps {
  diseaseId: string;
  onBack: () => void;
  setActiveTab: (tab: TabType) => void;
  onOpenAuthModal: () => void;
  currentUser: any;
}

export const DiseaseAssessmentForm: React.FC<DiseaseAssessmentFormProps> = ({
  diseaseId,
  onBack,
  setActiveTab,
  onOpenAuthModal,
  currentUser,
}) => {
  const [diseaseDetail, setDiseaseDetail] = useState<DiseaseDetail | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [loadingMetadata, setLoadingMetadata] = useState(true);

  // Input Mode: 'scan' (Primary / Recommended) vs 'manual' (Secondary / Fallback)
  const [inputMode, setInputMode] = useState<'scan' | 'manual'>('scan');

  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [loadingInference, setLoadingInference] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explain, setExplain] = useState(true);

  // Initialize feature defaults based on disease schema
  useEffect(() => {
    let mounted = true;

    Promise.all([
      api.getDiseaseDetail(diseaseId),
      api.getModels(diseaseId),
    ])
      .then(([detail, modelsRes]) => {
        if (!mounted) return;

        setDiseaseDetail(detail);

        // Filter models relevant to this disease and place production winner first
        const registeredModels = [...(modelsRes.models || [])].sort((a, b) => {
          if (a.production && !b.production) return -1;
          if (!a.production && b.production) return 1;
          return 0;
        });
        setAvailableModels(registeredModels);

        if (registeredModels.length > 0) {
          const prodWinner = registeredModels.find((m) => m.production) || registeredModels[0];
          setSelectedModelId(prodWinner.model_id);
        }

        // Initialize feature input dictionary
        if (detail.feature_names && detail.feature_names.length > 0) {
          const defaultInputs: Record<string, number> = {};
          detail.feature_names.forEach((fn) => {
            defaultInputs[fn] = 0;
          });
          setInputs(defaultInputs);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || 'Failed to load disease assessment metadata');
        }
      })
      .finally(() => {
        if (mounted) setLoadingMetadata(false);
      });

    return () => {
      mounted = false;
    };
  }, [diseaseId]);

  const handleInputChange = (field: string, value: string) => {
    const num = value === '' ? 0 : Number(value);
    setInputs((prev) => ({ ...prev, [field]: num }));
  };

  const handleLoadExample = () => {
    if (diseaseId === 'diabetes') {
      setInputs({
        Pregnancies: 6,
        Glucose: 148,
        BloodPressure: 72,
        SkinThickness: 35,
        Insulin: 0,
        BMI: 33.6,
        DiabetesPedigreeFunction: 0.627,
        Age: 50,
      });
    } else if (diseaseId === 'cardiovascular') {
      setInputs({
        age: 58,
        sex: 1,
        cp: 2,
        trestbps: 140,
        chol: 240,
        fbs: 1,
        restecg: 0,
        thalach: 150,
        exang: 0,
        oldpeak: 1.5,
        slope: 1,
        ca: 0,
        thal: 2,
      });
    } else if (diseaseId === 'cancer') {
      setInputs({
        mean_radius: 17.99,
        mean_texture: 10.38,
        mean_perimeter: 122.8,
        mean_area: 1001.0,
        mean_smoothness: 0.1184,
        mean_compactness: 0.2776,
        mean_concavity: 0.3001,
        mean_concave_points: 0.1471,
        mean_symmetry: 0.2419,
        mean_fractal_dimension: 0.07871,
        radius_error: 1.095,
        texture_error: 0.9053,
        perimeter_error: 8.589,
        area_error: 153.4,
        smoothness_error: 0.006399,
        compactness_error: 0.04904,
        concavity_error: 0.05373,
        concave_points_error: 0.01587,
        symmetry_error: 0.03003,
        fractal_dimension_error: 0.006193,
        worst_radius: 25.38,
        worst_texture: 17.33,
        worst_perimeter: 184.6,
        worst_area: 2019.0,
        worst_smoothness: 0.1622,
        worst_compactness: 0.6656,
        worst_concavity: 0.7119,
        worst_concave_points: 0.2654,
        worst_symmetry: 0.4601,
        worst_fractal_dimension: 0.1189,
      });
    }
  };

  const handleClear = () => {
    if (diseaseDetail?.feature_names) {
      const reset: Record<string, number> = {};
      diseaseDetail.feature_names.forEach((fn) => {
        reset[fn] = 0;
      });
      setInputs(reset);
    }
    setResult(null);
    setError(null);
  };

  const executeInference = async (featurePayload: Record<string, number>) => {
    if (!currentUser) {
      onOpenAuthModal();
      return;
    }

    if (!selectedModelId) {
      setError('Please select a valid model from the registry.');
      return;
    }

    setLoadingInference(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.runDiseaseInference(diseaseId, {
        features: featurePayload,
        model_id: selectedModelId,
        explain: explain,
        evaluation_context: 'single_sample',
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Assessment execution failed.');
    } finally {
      setLoadingInference(false);
    }
  };

  const handleRunAssessment = async () => {
    await executeInference(inputs);
  };

  const handleVerifiedFeaturesFromScanner = async (verifiedFeatures: Record<string, number>) => {
    setInputs(verifiedFeatures);
    await executeInference(verifiedFeatures);
  };

  const selectedModel = availableModels.find((m) => m.model_id === selectedModelId);

  if (loadingMetadata) {
    return (
      <div className="flex-1 overflow-y-auto p-8 flex items-center justify-center">
        <div className="flex items-center gap-3 text-slate-500 text-sm font-medium">
          <RefreshCw className="w-5 h-5 animate-spin text-emerald-600" />
          <span>Loading assessment metadata from backend...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 select-none">
      {/* Navigation Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-xs font-bold text-slate-600 hover:text-slate-900 bg-white px-3.5 py-2 rounded-xl border border-slate-200 shadow-xs"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Diseases Directory</span>
        </button>

        {!currentUser && (
          <div className="flex items-center gap-2 text-amber-800 bg-amber-50 px-3 py-1.5 rounded-xl border border-amber-200 text-xs font-medium">
            <UserCheck className="w-4 h-4 text-amber-600" />
            <span>Sign in required to save & generate personalized guidance</span>
          </div>
        )}
      </div>

      {/* Result View if Prediction is Complete */}
      {result ? (
        <PredictionResultView
          result={result}
          onReset={() => setResult(null)}
          setActiveTab={setActiveTab}
        />
      ) : (
        <>
          {/* Assessment Title Card */}
          <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs flex flex-col md:flex-row justify-between md:items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
                  {diseaseDetail?.category} Screening
                </span>
                <span className="text-xs text-slate-400 font-mono">{diseaseDetail?.version}</span>
              </div>
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight mt-1">
                {diseaseDetail?.name} Assessment
              </h1>
              <p className="text-xs text-slate-600 mt-1 max-w-2xl">
                {diseaseDetail?.description}
              </p>
            </div>

            {inputMode === 'manual' && (
              <div className="flex gap-2">
                <button
                  onClick={handleLoadExample}
                  disabled={loadingInference}
                  className="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
                >
                  Load Sample
                </button>
                <button
                  onClick={handleClear}
                  disabled={loadingInference}
                  className="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
                >
                  Clear Form
                </button>
              </div>
            )}
          </div>

          {/* Primary Input Mode Selector */}
          <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs space-y-4">
            <div className="text-left">
              <h2 className="text-base font-bold text-slate-900">How would you like to provide your medical data?</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Scan your medical document for automatic extraction or input the clinical values manually.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Option 1: Scan Medical Report (PRIMARY & RECOMMENDED) */}
              <button
                type="button"
                onClick={() => setInputMode('scan')}
                className={`p-5 rounded-2xl border text-left transition-all duration-200 flex items-start gap-4 cursor-pointer ${
                  inputMode === 'scan'
                    ? 'bg-emerald-50/50 border-emerald-500 shadow-sm ring-2 ring-emerald-500/20'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/60'
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 transition-colors ${
                    inputMode === 'scan' ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'
                  }`}
                >
                  <FileSearch className="w-6 h-6" />
                </div>
                <div className="space-y-1 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-slate-900">Scan Medical Report</span>
                    <span className="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300">
                      Recommended
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    Upload your medical report (PDF or photo). Clinical indicators are extracted, unit-normalized, and verified.
                  </p>
                </div>
              </button>

              {/* Option 2: Manual Data Entry (SECONDARY / FALLBACK) */}
              <button
                type="button"
                onClick={() => setInputMode('manual')}
                className={`p-5 rounded-2xl border text-left transition-all duration-200 flex items-start gap-4 cursor-pointer ${
                  inputMode === 'manual'
                    ? 'bg-emerald-50/50 border-emerald-500 shadow-sm ring-2 ring-emerald-500/20'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/60'
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 transition-colors ${
                    inputMode === 'manual' ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'
                  }`}
                >
                  <Edit3 className="w-6 h-6" />
                </div>
                <div className="space-y-1 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-slate-900">Manual Data Entry</span>
                    <span className="text-[10px] font-semibold text-slate-500 uppercase px-2 py-0.5 rounded-full bg-slate-100 border border-slate-200">
                      Fallback
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    Directly enter medical parameters yourself into the structured disease assessment form.
                  </p>
                </div>
              </button>
            </div>
          </div>

          {/* Model Selector Bar (Shared by both input methods) */}
          <div className="bg-white rounded-[2rem] border border-slate-200 p-5 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <span className="block text-xs font-bold text-slate-900">Target Prediction Model</span>
                <span className="text-[11px] text-slate-500">
                  {selectedModel?.production ? '★ Production Champion' : 'Registered Registry Model'}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                disabled={loadingInference}
                className="text-xs border border-slate-200 rounded-xl p-2.5 bg-slate-50 font-medium focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
              >
                {availableModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.production ? '★ [RECOMMENDED] ' : ''}{m.display_name} ({m.is_quantum ? 'Quantum' : 'Classical'}) - {m.version}
                  </option>
                ))}
              </select>

              <div className="flex items-center gap-1.5 text-xs text-slate-700">
                <input
                  type="checkbox"
                  id="explain_shared"
                  checked={explain}
                  onChange={(e) => setExplain(e.target.checked)}
                  disabled={loadingInference}
                  className="rounded text-emerald-600 focus:ring-emerald-500"
                />
                <label htmlFor="explain_shared" className="cursor-pointer">Explainability</label>
              </div>
            </div>
          </div>

          {/* Main Content: Scanner or Manual Form */}
          {inputMode === 'scan' ? (
            <MedicalReportScanner
              diseaseId={diseaseId}
              diseaseName={diseaseDetail?.name || diseaseId}
              onVerifiedFeatures={handleVerifiedFeaturesFromScanner}
              onSwitchToManual={() => setInputMode('manual')}
            />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column: Model Configuration & Safeguards */}
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs space-y-4">
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-600" />
                    Model Details
                  </h3>

                  {selectedModel && (
                    <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Tier:</span>
                        <span className="font-semibold text-slate-800">
                          {selectedModel.production ? (
                            <span className="text-emerald-700 font-bold">Recommended Production Winner</span>
                          ) : (
                            'Alternative Evaluated'
                          )}
                        </span>
                      </div>
                      {selectedModel.frozen_threshold !== undefined && selectedModel.frozen_threshold !== null && (
                        <div className="flex justify-between">
                          <span className="text-slate-500">Decision Threshold:</span>
                          <span className="font-mono font-semibold text-slate-800">
                            {selectedModel.frozen_threshold.toFixed(2)}
                          </span>
                        </div>
                      )}
                      {selectedModel.selection_reason && (
                        <p className="text-slate-600 pt-1 leading-relaxed border-t border-slate-200/60 mt-1">
                          {selectedModel.selection_reason}
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs space-y-3">
                  <h3 className="text-base font-bold text-slate-900">Pipeline Safeguards</h3>
                  <div className="space-y-2 text-xs text-slate-600">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>Authoritative Backend Validation</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>Training-Fitted Scaling & Imputation</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>Platt Probability Calibration</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>Deterministic Rule-Set Versioning</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Dynamic Form Fields */}
              <div className="lg:col-span-2 space-y-6">
                <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <h3 className="text-base font-bold text-slate-900">
                      Biomedical & Morphometric Indicators ({diseaseDetail?.feature_names?.length || 0} features)
                    </h3>
                  </div>

                  {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>{error}</span>
                    </div>
                  )}

                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 text-xs">
                    {diseaseDetail?.feature_names?.map((fn) => (
                      <div key={fn} className="space-y-1">
                        <label className="block font-semibold text-slate-700 truncate" title={fn}>
                          {fn}
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={inputs[fn] !== undefined ? inputs[fn] : ''}
                          onChange={(e) => handleInputChange(fn, e.target.value)}
                          disabled={loadingInference}
                          className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 font-mono text-slate-900 focus:bg-white focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                        />
                      </div>
                    ))}
                  </div>

                  <div className="pt-4 border-t border-slate-100 flex justify-end">
                    <button
                      onClick={handleRunAssessment}
                      disabled={loadingInference}
                      className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-all shadow-sm disabled:opacity-50"
                    >
                      {loadingInference ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Executing Model Inference...</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          <span>Run Disease Risk Assessment</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
