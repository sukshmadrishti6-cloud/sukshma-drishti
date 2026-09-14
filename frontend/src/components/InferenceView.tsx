import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, FileText, CheckCircle2, RefreshCw, AlertTriangle, Activity, Heart, Sparkles, ChevronRight, Award, FileSearch, Edit3 } from 'lucide-react';
import { InferenceRequest, InferenceResponse, ExplanationResult, TabType, ModelSummary } from '../types';
import { api } from '../api';
import { MedicalReportScanner } from './MedicalReportScanner';

interface InferenceViewProps {
  onResult: (result: InferenceResponse | null) => void;
  setActiveTab: (tab: TabType) => void;
}

const DISEASES = [
  { id: 'diabetes', label: 'Diabetes', icon: Activity },
  { id: 'cardiovascular', label: 'Cardiovascular', icon: Heart },
  { id: 'cancer', label: 'Breast Cancer', icon: Sparkles },
];

const PRESETS: Record<string, Record<string, number>> = {
  diabetes: {
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50,
  },
  cardiovascular: {
    age: 58,
    sex: 1,
    cp: 2,
    trestbps: 135,
    chol: 240,
    fbs: 0,
    restecg: 1,
    thalach: 152,
    exang: 0,
    oldpeak: 1.2,
    slope: 1,
    ca: 0,
    thal: 2,
  },
  cancer: {
    mean_radius: 14.2,
    mean_texture: 19.3,
    mean_perimeter: 92.5,
    mean_area: 654.0,
    mean_smoothness: 0.096,
    mean_compactness: 0.104,
    mean_concavity: 0.088,
    mean_concave_points: 0.048,
    mean_symmetry: 0.181,
    mean_fractal_dimension: 0.062,
    radius_error: 0.40,
    texture_error: 1.21,
    perimeter_error: 2.86,
    area_error: 40.3,
    smoothness_error: 0.007,
    compactness_error: 0.025,
    concavity_error: 0.031,
    concave_points_error: 0.011,
    symmetry_error: 0.020,
    fractal_dimension_error: 0.003,
    worst_radius: 16.2,
    worst_texture: 25.6,
    worst_perimeter: 107.2,
    worst_area: 880.5,
    worst_smoothness: 0.132,
    worst_compactness: 0.254,
    worst_concavity: 0.272,
    worst_concave_points: 0.114,
    worst_symmetry: 0.290,
    worst_fractal_dimension: 0.083,
  },
};

export const InferenceView: React.FC<InferenceViewProps> = ({ onResult, setActiveTab }) => {
  const [selectedDisease, setSelectedDisease] = useState<string>('diabetes');
  const [inputs, setInputs] = useState<Record<string, number>>(PRESETS.diabetes);
  const [inputMode, setInputMode] = useState<'scan' | 'manual'>('scan');
  const [cancerSubTab, setCancerSubTab] = useState<'mean' | 'error' | 'worst'>('mean');

  const [availableModels, setAvailableModels] = useState<ModelSummary[]>([]);
  const [modelName, setModelName] = useState<string>('');
  const [modelsLoading, setModelsLoading] = useState<boolean>(true);
  const [evalContext, setEvalContext] = useState<string>('single_sample');
  const [explain, setExplain] = useState<boolean>(true);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleVerifiedFeaturesFromScanner = async (verifiedFeatures: Record<string, number>) => {
    setInputs(verifiedFeatures);
    if (!modelName) {
      setError('Please select a valid model from the registry.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    onResult(null);

    try {
      const res = await api.runDiseaseInference(selectedDisease, {
        features: verifiedFeatures,
        model_id: modelName,
        explain: explain,
        evaluation_context: evalContext,
      });
      setResult(res);
      onResult(res);
    } catch (err: any) {
      setError(err.message || 'An error occurred during inference.');
    } finally {
      setLoading(false);
    }
  };

  // Load models on disease change
  useEffect(() => {
    let mounted = true;
    setModelsLoading(true);
    setInputs(PRESETS[selectedDisease] || {});
    setResult(null);
    setError(null);
    onResult(null);

    api.getModels(selectedDisease)
      .then(res => {
        if (mounted && res && res.models && res.models.length > 0) {
          const sortedModels = [...res.models].sort((a, b) => {
            if (a.production && !b.production) return -1;
            if (!a.production && b.production) return 1;
            return 0;
          });
          setAvailableModels(sortedModels);
          const prodWinner = sortedModels.find(m => m.production) || sortedModels[0];
          setModelName(prodWinner.model_id);
        } else if (mounted) {
          setAvailableModels([]);
          setModelName('');
        }
      })
      .catch(err => {
        console.error(`Failed to fetch ${selectedDisease} models:`, err);
      })
      .finally(() => {
        if (mounted) setModelsLoading(false);
      });
    return () => { mounted = false; };
  }, [selectedDisease]);

  const handleLoadExample = () => {
    setInputs(PRESETS[selectedDisease] || {});
  };

  const handleClear = () => {
    const cleared: Record<string, number> = {};
    Object.keys(PRESETS[selectedDisease] || {}).forEach(k => {
      cleared[k] = 0;
    });
    setInputs(cleared);
    setResult(null);
    setError(null);
    onResult(null);
  };

  const handleRunInference = async () => {
    if (!modelName) {
      setError('Please select a valid model from the registry.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    onResult(null);

    try {
      const res = await api.runDiseaseInference(selectedDisease, {
        features: inputs,
        model_id: modelName,
        explain: explain,
        evaluation_context: evalContext,
      });
      setResult(res);
      onResult(res);
    } catch (err: any) {
      setError(err.message || 'An error occurred during inference.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field: string, value: string) => {
    setInputs(prev => ({ ...prev, [field]: Number(value) }));
  };

  // Helper to split cancer features
  const getCancerKeysForTab = () => {
    const allKeys = Object.keys(PRESETS.cancer);
    if (cancerSubTab === 'mean') return allKeys.filter(k => k.startsWith('mean_'));
    if (cancerSubTab === 'error') return allKeys.filter(k => k.endsWith('_error'));
    return allKeys.filter(k => k.startsWith('worst_'));
  };

  const currentFeatureKeys = selectedDisease === 'cancer'
    ? getCancerKeysForTab()
    : Object.keys(PRESETS[selectedDisease] || {});

  const selectedModelObj = availableModels.find(m => m.model_id === modelName);

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header with Disease Tabs */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Play className="text-orange-500" />
            Biomedical Clinical Inference
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Execute single-sample clinical predictions across Diabetes, Cardiovascular, and Cancer domains.
          </p>
        </div>

        {/* Disease Selection Tabs */}
        <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-2xl border border-slate-200 shrink-0">
          {DISEASES.map(d => {
            const Icon = d.icon;
            const isSelected = selectedDisease === d.id;
            return (
              <button
                key={d.id}
                onClick={() => setSelectedDisease(d.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                  isSelected
                    ? 'bg-white text-slate-900 shadow-xs border border-slate-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-orange-500' : 'text-slate-400'}`} />
                {d.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Configuration & Status */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center justify-between">
              <span>Model Selection</span>
              {selectedModelObj?.production && (
                <span className="flex items-center gap-1 text-[11px] font-bold text-orange-600 bg-orange-50 px-2 py-0.5 rounded-md border border-orange-200">
                  <Award className="w-3 h-3" /> Production Winner
                </span>
              )}
            </h3>
            <div className="space-y-4 text-sm">
              <div>
                <label className="block text-slate-600 mb-1 font-semibold text-xs uppercase tracking-wider">
                  Target Model ({availableModels.length} available)
                </label>
                <select 
                  className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500 text-xs font-medium"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  disabled={modelsLoading || availableModels.length === 0}
                >
                  {availableModels.length > 0 ? (
                    availableModels.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.production ? '★ ' : ''}{m.display_name} ({m.is_quantum ? 'Quantum' : 'Classical'}) - {m.version}
                      </option>
                    ))
                  ) : (
                    <option value="">{modelsLoading ? 'Loading models from registry...' : 'No models available'}</option>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-slate-600 mb-1 font-semibold text-xs uppercase tracking-wider">Evaluation Context</label>
                <select 
                  className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500 text-xs"
                  value={evalContext}
                  onChange={(e) => setEvalContext(e.target.value)}
                  disabled={loading}
                >
                  <option value="single_sample">Standard Clinical Inference (Single-Sample)</option>
                  <option value="holdout_reference">Holdout Population Reference (Stratified Test Split)</option>
                </select>
              </div>

              <div className="flex items-center gap-2 pt-2">
                <input 
                  type="checkbox" 
                  id="explain" 
                  checked={explain} 
                  onChange={(e) => setExplain(e.target.checked)}
                  className="rounded border-slate-300 text-orange-600 focus:ring-orange-500"
                />
                <label htmlFor="explain" className="text-slate-700 font-medium cursor-pointer text-xs">
                  Generate Explainability & Local Feature Attribution
                </label>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Pipeline Execution Checklist</h3>
            <div className="space-y-3 text-sm text-slate-600">
              <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Strict Schema Validation ({selectedDisease})</div>
              <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Train-Fitted Scaler Preprocessing</div>
              <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Cross-Disease Boundary Guard</div>
              <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Local Feature Explainer</div>
              <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-500" /> Clinical Recommendation Engine</div>
            </div>
          </div>
        </div>

        {/* Right Column: Input Mode Selector, Scanner / Manual Form, & Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Mode Selector */}
          <div className="bg-white rounded-[2rem] border border-orange-100 p-4 shadow-xs flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200">
              <button
                type="button"
                onClick={() => setInputMode('scan')}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  inputMode === 'scan'
                    ? 'bg-white text-slate-900 shadow-2xs border border-slate-200'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <FileSearch className="w-3.5 h-3.5 text-orange-500" />
                <span>Scan Report (Recommended)</span>
              </button>
              <button
                type="button"
                onClick={() => setInputMode('manual')}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  inputMode === 'manual'
                    ? 'bg-white text-slate-900 shadow-2xs border border-slate-200'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Edit3 className="w-3.5 h-3.5 text-slate-500" />
                <span>Manual Feature Inputs</span>
              </button>
            </div>

            <span className="text-[11px] text-slate-500 hidden sm:inline">
              {inputMode === 'scan' ? 'AI OCR & Parameter Parser' : 'Direct Parameter Input'}
            </span>
          </div>

          {inputMode === 'scan' ? (
            <MedicalReportScanner
              diseaseId={selectedDisease}
              diseaseName={selectedDisease.toUpperCase()}
              onVerifiedFeatures={handleVerifiedFeaturesFromScanner}
              onSwitchToManual={() => setInputMode('manual')}
            />
          ) : (
            <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
                <div>
                  <h3 className="text-lg font-bold text-slate-900">
                    Biomedical Predictor Features ({selectedDisease.toUpperCase()})
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Input raw patient measurements. All normalizations and transformations are handled automatically.
                  </p>
                </div>
                <div className="flex gap-2 shrink-0">
                  <button 
                    onClick={handleLoadExample} 
                    className="text-xs px-3 py-1.5 bg-orange-50 text-orange-700 font-semibold rounded-lg hover:bg-orange-100 disabled:opacity-50 transition-colors" 
                    disabled={loading}
                  >
                    Load Preset
                  </button>
                  <button 
                    onClick={handleClear} 
                    className="text-xs px-3 py-1.5 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 disabled:opacity-50 transition-colors" 
                    disabled={loading}
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Cancer Feature Sub-Tabs */}
              {selectedDisease === 'cancer' && (
                <div className="flex gap-2 mb-4 bg-slate-100 p-1 rounded-xl text-xs">
                  <button
                    onClick={() => setCancerSubTab('mean')}
                    className={`flex-1 py-1.5 rounded-lg font-bold transition-all ${
                      cancerSubTab === 'mean' ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Mean Features (10)
                  </button>
                  <button
                    onClick={() => setCancerSubTab('error')}
                    className={`flex-1 py-1.5 rounded-lg font-bold transition-all ${
                      cancerSubTab === 'error' ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Standard Errors (10)
                  </button>
                  <button
                    onClick={() => setCancerSubTab('worst')}
                    className={`flex-1 py-1.5 rounded-lg font-bold transition-all ${
                      cancerSubTab === 'worst' ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Worst Values (10)
                  </button>
                </div>
              )}

              {/* Grid of Inputs */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 text-sm">
                {currentFeatureKeys.map((key) => (
                  <div key={key}>
                    <label className="block text-slate-600 mb-1 text-[11px] font-medium truncate" title={key}>
                      {key.replace(/_/g, ' ')}
                    </label>
                    <input 
                      type="number" 
                      step="any"
                      value={inputs[key] !== undefined ? inputs[key] : ''} 
                      onChange={(e) => handleChange(key, e.target.value)}
                      className="w-full border border-slate-200 rounded-xl p-2 bg-slate-50 font-mono text-xs focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500" 
                      disabled={loading}
                    />
                  </div>
                ))}
              </div>

              <div className="mt-6 flex justify-end">
                <button 
                  onClick={handleRunInference}
                  disabled={loading || !modelName}
                  className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 disabled:opacity-50 transition-colors shadow-xs"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} 
                  {loading ? 'Evaluating...' : 'Run Clinical Inference'}
                </button>
              </div>
            </div>
          )}

          {/* Inference Result View */}
          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold text-slate-900">Inference Assessment Result</h3>
              <div className="flex gap-2">
                <button 
                  disabled={!result || !result.explanation} 
                  onClick={() => setActiveTab('explainability')}
                  className="text-xs px-3 py-1.5 bg-blue-50 text-blue-600 font-semibold rounded-lg disabled:opacity-50 flex items-center gap-1 transition-colors"
                >
                  Full Explainability <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {loading && (
              <div className="text-slate-400 text-sm flex justify-center items-center h-28 border border-dashed border-slate-200 rounded-xl">
                <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Running model inference across {selectedDisease} pipeline...
              </div>
            )}
            
            {!loading && error && (
              <div className="text-red-500 text-sm flex justify-center items-center h-28 border border-dashed border-red-200 bg-red-50 rounded-xl px-4 text-center">
                <AlertTriangle className="w-5 h-5 mr-2 shrink-0" /> {error}
              </div>
            )}

            {!loading && !result && !error && (
              <div className="text-slate-400 text-sm flex justify-center items-center h-28 border border-dashed border-slate-200 rounded-xl">
                Click &quot;Run Clinical Inference&quot; to execute real-time model prediction.
              </div>
            )}

            {!loading && result && (
              <>
                <div className="mb-6 flex flex-wrap gap-6 items-center p-4 rounded-2xl bg-slate-50 border border-slate-100">
                  <div>
                    <span className="block text-slate-500 text-xs uppercase font-semibold mb-1">Predicted Class</span>
                    <span className={`text-2xl font-bold ${result.predicted_class === 1 ? 'text-red-600' : 'text-emerald-600'}`}>
                      {result.predicted_class === 1 
                        ? (selectedDisease === 'cancer' ? 'Malignant (Class 1)' : 'High Risk / Positive (Class 1)')
                        : (selectedDisease === 'cancer' ? 'Benign (Class 0)' : 'Low Risk / Negative (Class 0)')}
                    </span>
                  </div>

                  {result.probability !== undefined && result.probability !== null && (
                    <div>
                      <span className="block text-slate-500 text-xs uppercase font-semibold mb-1">Calibrated Probability</span>
                      <span className="text-2xl font-bold text-slate-900">
                        {(result.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {result.decision_score !== undefined && result.decision_score !== null && (
                    <div>
                      <span className="block text-slate-500 text-xs uppercase font-semibold mb-1">Quantum Decision Margin</span>
                      <span className="text-2xl font-bold text-purple-700 font-mono">
                        {result.decision_score.toFixed(4)}
                      </span>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="block text-slate-500 mb-0.5 text-[10px] uppercase font-semibold">Inference ID</span>
                    <span className="font-medium text-slate-900 font-mono text-[11px] truncate block" title={result.inference_id || 'N/A'}>{result.inference_id || 'N/A'}</span>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="block text-slate-500 mb-0.5 text-[10px] uppercase font-semibold">Model Artifact</span>
                    <span className="font-medium text-slate-900 truncate block" title={result.artifact_id}>{result.artifact_id}</span>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="block text-slate-500 mb-0.5 text-[10px] uppercase font-semibold">Score Semantics</span>
                    <span className="font-medium text-slate-900 font-mono text-[10px] truncate block" title={result.score_semantics || 'probability'}>{result.score_semantics || 'calibrated_probability'}</span>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="block text-slate-500 mb-0.5 text-[10px] uppercase font-semibold">Latency</span>
                    <span className="font-medium text-slate-900 font-mono">{result.audit?.inference_time_ms ? `${result.audit.inference_time_ms.toFixed(2)} ms` : '—'}</span>
                  </div>
                </div>

                {/* Inline Feature Contributions Preview */}
                {result.explanation && result.explanation.contributions && result.explanation.contributions.length > 0 && (
                  <div className="mt-4 p-4 border border-blue-100 bg-blue-50/50 rounded-xl">
                    <span className="text-blue-900 text-xs font-bold uppercase tracking-wider block mb-2">
                      Key Feature Drivers ({result.explanation.explanation_method})
                    </span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {result.explanation.contributions.slice(0, 4).map((c, idx) => (
                        <div key={idx} className="flex justify-between items-center text-xs bg-white p-2 rounded-lg border border-blue-100">
                          <span className="font-mono text-slate-700">{c.name}</span>
                          <span className={`font-mono font-bold ${c.contribution >= 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                            {c.contribution >= 0 ? `+${c.contribution.toFixed(4)}` : c.contribution.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

