import React, { useState, useEffect, useRef } from 'react';
import { TabType } from '../types';
import { LayoutDashboard, ShieldAlert, Activity, Loader2, Play, RefreshCw, CheckCircle2, Heart, Activity as DiabetesIcon, Dna } from 'lucide-react';
import { api } from '../api';

interface ClassicalBaselinesViewProps {
  setActiveTab: (tab: TabType) => void;
}

const formatMetric = (val: any, digits: number = 4): string => {
  if (val === null || val === undefined || isNaN(Number(val))) return '—';
  return Number(val).toFixed(digits);
};

export const ClassicalBaselinesView: React.FC<ClassicalBaselinesViewProps> = () => {
  const [selectedDisease, setSelectedDisease] = useState<string>('diabetes');
  const [evalData, setEvalData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Model selector for detailed artifacts
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [activeCurveTab, setActiveCurveTab] = useState<'roc' | 'pr'>('roc');
  
  // Execution lifecycle state
  const [isRunning, setIsRunning] = useState(false);
  const [runMessage, setRunMessage] = useState<string | null>(null);
  const pollTimerRef = useRef<any>(null);

  const diseases = [
    { id: 'diabetes', name: 'Diabetes', icon: DiabetesIcon, color: 'orange' },
    { id: 'cardiovascular', name: 'Cardiovascular', icon: Heart, color: 'rose' },
    { id: 'cancer', name: 'Cancer', icon: Dna, color: 'amber' },
  ];

  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'Sensitivity', 'Specificity', 'Brier', 'Log Loss'];

  const loadEvaluationData = (diseaseId: string) => {
    setLoading(true);
    setError(null);
    return api.getEvaluation(diseaseId)
      .then(data => {
        setEvalData(data);
        if (data.models && data.models.length > 0) {
          const firstEval = data.models.find((m: any) => m.evaluated) || data.models[0];
          setSelectedModelId(firstEval.model_id);
        }
        setLoading(false);
      })
      .catch(err => {
        console.warn(`Failed to load evaluation for ${diseaseId}:`, err);
        setError(err.message || `Failed to load evaluation data for ${diseaseId}.`);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadEvaluationData(selectedDisease);
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [selectedDisease]);

  const handleRunBaselines = async () => {
    setIsRunning(true);
    setError(null);
    setRunMessage('Initiating unified classical benchmark execution...');

    try {
      const runRes = await api.runBaselines();
      setRunMessage(runRes.message || 'Executing benchmark on holdout test partition...');

      pollTimerRef.current = setInterval(async () => {
        try {
          const statusRes = await api.getBaselinesStatus();
          if (statusRes.status === 'running') {
            setRunMessage(statusRes.message || 'Training & evaluating models...');
          } else if (statusRes.status === 'completed') {
            clearInterval(pollTimerRef.current);
            setIsRunning(false);
            setRunMessage('Benchmark completed successfully! Updating results.');
            await loadEvaluationData(selectedDisease);
            setTimeout(() => setRunMessage(null), 4000);
          } else if (statusRes.status === 'failed') {
            clearInterval(pollTimerRef.current);
            setIsRunning(false);
            setError(statusRes.error || 'Benchmark execution failed.');
            setRunMessage(null);
          }
        } catch (pollErr: any) {
          clearInterval(pollTimerRef.current);
          setIsRunning(false);
          setError(pollErr.message || 'Error tracking benchmark status.');
        }
      }, 1200);
    } catch (err: any) {
      setIsRunning(false);
      setError(err.message || 'Failed to trigger benchmark runner.');
    }
  };

  const modelsList = evalData?.models || [];
  const selectedModel = modelsList.find((m: any) => m.model_id === selectedModelId) || modelsList[0];

  const rocData = selectedModel?.curves?.roc_curve;
  const prData = selectedModel?.curves?.pr_curve;
  const hasRocPoints = rocData?.fpr && rocData?.tpr && Array.isArray(rocData.fpr) && rocData.fpr.length > 0;
  const hasPrPoints = prData?.precision && prData?.recall && Array.isArray(prData.precision) && prData.precision.length > 0;
  const calData = selectedModel?.calibration;
  const hasCalPoints = calData?.prob_pred && calData?.prob_true && Array.isArray(calData.prob_pred) && calData.prob_pred.length > 0;

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <LayoutDashboard className="text-orange-500" />
            Evaluation & Model Benchmarks
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Authoritative 80/20 holdout evaluation artifacts, confusion matrices, ROC/PR curves & calibration for all diseases.
          </p>
        </div>
        <button 
          onClick={handleRunBaselines}
          disabled={isRunning}
          className="bg-orange-500 hover:bg-orange-600 text-white px-5 py-2.5 rounded-xl text-sm font-semibold transition-all shadow-xs flex items-center gap-2 disabled:opacity-50 shrink-0"
        >
          {isRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {isRunning ? 'Running Baselines...' : 'Run Classical Baselines'}
        </button>
      </div>

      {/* Disease Selection Tabs */}
      <div className="flex flex-wrap gap-3">
        {diseases.map(d => {
          const Icon = d.icon;
          const isActive = selectedDisease === d.id;
          return (
            <button
              key={d.id}
              onClick={() => setSelectedDisease(d.id)}
              className={`flex items-center gap-2 px-5 py-3 rounded-2xl text-xs font-bold transition-all ${
                isActive
                  ? 'bg-slate-900 text-white shadow-md'
                  : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-orange-400' : 'text-slate-400'}`} />
              {d.name} Evaluation
            </button>
          );
        })}
      </div>

      {runMessage && (
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4 text-blue-800 text-sm flex items-center gap-3 animate-fade-in">
          {isRunning ? <Loader2 className="w-5 h-5 text-blue-500 animate-spin shrink-0" /> : <CheckCircle2 className="w-5 h-5 text-blue-500 shrink-0" />}
          <span>{runMessage}</span>
        </div>
      )}

      {loading && (
        <div className="flex justify-center items-center h-48">
          <Loader2 className="w-8 h-8 text-orange-500 animate-spin" />
        </div>
      )}

      {error && !loading && (
        <div className="bg-amber-50 border border-amber-200 rounded-[2rem] p-6 text-amber-800 text-sm flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" /> 
          <div>
            <strong className="block mb-1">Evaluation Data Notice</strong>
            {error}
          </div>
        </div>
      )}

      {/* Evaluation Metrics Matrix */}
      {!loading && (
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs overflow-x-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-orange-500" />
              Model Evaluation Matrix ({selectedDisease.toUpperCase()})
            </h3>
            <div className="text-xs text-slate-500 font-mono">
              Evaluated: <strong className="text-emerald-600">{evalData?.evaluated_models_count || modelsList.length}</strong> / {evalData?.total_models || modelsList.length} models
            </div>
          </div>
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 font-semibold rounded-tl-xl">Model</th>
                {metrics.map(m => (
                  <th key={m} className="px-4 py-3 font-semibold">{m}</th>
                ))}
                <th className="px-4 py-3 font-semibold rounded-tr-xl">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {modelsList.length > 0 ? (
                modelsList.map((m: any) => {
                  const isEval = m.evaluated !== false && m.metrics?.accuracy !== undefined && m.metrics?.accuracy !== null;
                  const isSelected = selectedModelId === m.model_id;
                  const met = m.metrics || {};
                  return (
                    <tr 
                      key={m.model_id} 
                      onClick={() => isEval && setSelectedModelId(m.model_id)}
                      className={`transition-colors ${isEval ? 'cursor-pointer' : 'opacity-60 cursor-not-allowed'} ${
                        isSelected ? 'bg-orange-50/60 font-medium' : 'hover:bg-slate-50/50'
                      }`}
                    >
                      <td className="px-4 py-4 font-medium text-slate-900 flex items-center gap-2">
                        {isSelected && <div className="w-2 h-2 rounded-full bg-orange-500" />}
                        <span>{m.model_name || m.name || m.model_id}</span>
                        {m.production && (
                          <span className="px-1.5 py-0.5 bg-emerald-100 text-emerald-800 text-[10px] font-bold rounded-md uppercase">
                            Winner
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.accuracy)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.precision)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.recall)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.f1_score)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.roc_auc)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.pr_auc)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.sensitivity)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.specificity)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.brier_score)}</td>
                      <td className="px-4 py-4 font-mono text-slate-700">{formatMetric(met.log_loss)}</td>
                      <td className="px-4 py-4">
                        {isEval ? (
                          <span className="px-2 py-1 bg-emerald-50 text-emerald-700 rounded-md text-xs font-semibold">Evaluated</span>
                        ) : (
                          <span className="px-2 py-1 bg-purple-50 text-purple-700 rounded-md text-xs font-semibold">Not Evaluated / Circuit Only</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={12} className="px-4 py-8 text-center text-slate-400">
                    No models registered or evaluated for this disease partition.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Model Selector Pills */}
      {modelsList.filter((m: any) => m.evaluated !== false).length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pt-2">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider mr-2">Inspect Artifacts For:</span>
          {modelsList.filter((m: any) => m.evaluated !== false).map((m: any) => (
            <button
              key={m.model_id}
              onClick={() => setSelectedModelId(m.model_id)}
              className={`px-4 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                selectedModelId === m.model_id 
                  ? 'bg-slate-900 text-white shadow-xs' 
                  : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {m.model_name || m.name || m.model_id}
            </button>
          ))}
        </div>
      )}

      {/* Detailed Artifact Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* 1. Confusion Matrix Card */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-bold text-slate-900">Confusion Matrix</h4>
              <span className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded-full font-mono truncate max-w-[150px]" title={selectedModel?.model_name || selectedModel?.model_id}>
                {selectedModel?.model_name || selectedModel?.model_id || 'Model'}
              </span>
            </div>

            {selectedModel?.confusion_matrix && selectedModel?.confusion_matrix?.tn !== undefined ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
                  <div className="bg-emerald-50 border border-emerald-200 p-3 rounded-xl">
                    <span className="text-[10px] uppercase text-emerald-600 font-bold block mb-1">True Neg (TN)</span>
                    <span className="text-2xl font-bold text-emerald-800">{selectedModel.confusion_matrix.tn}</span>
                    <span className="text-[10px] text-emerald-600 block mt-1">Normal correctly identified</span>
                  </div>
                  <div className="bg-amber-50 border border-amber-200 p-3 rounded-xl">
                    <span className="text-[10px] uppercase text-amber-600 font-bold block mb-1">False Pos (FP)</span>
                    <span className="text-2xl font-bold text-amber-800">{selectedModel.confusion_matrix.fp}</span>
                    <span className="text-[10px] text-amber-600 block mt-1">Type I False Alarm</span>
                  </div>
                  <div className="bg-red-50 border border-red-200 p-3 rounded-xl">
                    <span className="text-[10px] uppercase text-red-600 font-bold block mb-1">False Neg (FN)</span>
                    <span className="text-2xl font-bold text-red-800">{selectedModel.confusion_matrix.fn}</span>
                    <span className="text-[10px] text-red-600 block mt-1">Type II Missed Disease</span>
                  </div>
                  <div className="bg-blue-50 border border-blue-200 p-3 rounded-xl">
                    <span className="text-[10px] uppercase text-blue-600 font-bold block mb-1">True Pos (TP)</span>
                    <span className="text-2xl font-bold text-blue-800">{selectedModel.confusion_matrix.tp}</span>
                    <span className="text-[10px] text-blue-600 block mt-1">Disease correctly detected</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 flex justify-between text-xs text-slate-500">
                  <span>Recall: <strong>{selectedModel.metrics?.recall ? `${(Number(selectedModel.metrics.recall) * 100).toFixed(1)}%` : '—'}</strong></span>
                  <span>Specificity: <strong>{selectedModel.metrics?.specificity ? `${(Number(selectedModel.metrics.specificity) * 100).toFixed(1)}%` : '—'}</strong></span>
                </div>
              </div>
            ) : (
              <div className="flex justify-center items-center h-48 text-slate-400 text-sm">
                Confusion Matrix unavailable
              </div>
            )}
          </div>
        </div>

        {/* 2. ROC / PR Curves Card */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-3">
              <h4 className="font-bold text-slate-900">Performance Curves</h4>
              <div className="flex gap-1 bg-slate-100 p-0.5 rounded-lg text-xs font-semibold">
                <button
                  onClick={() => setActiveCurveTab('roc')}
                  className={`px-2 py-0.5 rounded-md transition-colors ${activeCurveTab === 'roc' ? 'bg-white shadow-xs text-orange-600' : 'text-slate-600'}`}
                >
                  ROC
                </button>
                <button
                  onClick={() => setActiveCurveTab('pr')}
                  className={`px-2 py-0.5 rounded-md transition-colors ${activeCurveTab === 'pr' ? 'bg-white shadow-xs text-indigo-600' : 'text-slate-600'}`}
                >
                  PR
                </button>
              </div>
            </div>

            {activeCurveTab === 'roc' ? (
              hasRocPoints ? (
                <div>
                  <div className="relative w-full h-44 bg-slate-50 rounded-xl p-2 border border-slate-100 flex flex-col justify-between">
                    <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                      <line x1="0" y1="100" x2="100" y2="0" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3,3" />
                      <polyline
                        fill="none"
                        stroke="#f97316"
                        strokeWidth="2.5"
                        points={rocData.fpr.map((x: number, i: number) => {
                          const y = rocData.tpr[i];
                          return `${x * 100},${100 - y * 100}`;
                        }).join(' ')}
                      />
                    </svg>
                    <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                      <span>FPR: 0.0</span>
                      <span className="font-bold text-orange-600">AUC: {formatMetric(rocData.auc ?? selectedModel?.metrics?.roc_auc)}</span>
                      <span>1.0</span>
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-500 text-center mt-2">
                    True Positive Rate vs False Positive Rate
                  </p>
                </div>
              ) : (
                <div className="flex justify-center items-center h-48 text-slate-400 text-sm text-center p-4">
                  ROC curve coordinates not evaluated for this model artifact
                </div>
              )
            ) : (
              hasPrPoints ? (
                <div>
                  <div className="relative w-full h-44 bg-slate-50 rounded-xl p-2 border border-slate-100 flex flex-col justify-between">
                    <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                      <line x1="0" y1="65" x2="100" y2="65" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3,3" />
                      <polyline
                        fill="none"
                        stroke="#6366f1"
                        strokeWidth="2.5"
                        points={prData.recall.map((x: number, i: number) => {
                          const y = prData.precision[i];
                          return `${x * 100},${100 - y * 100}`;
                        }).join(' ')}
                      />
                    </svg>
                    <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                      <span>Recall: 0.0</span>
                      <span className="font-bold text-indigo-600">PR-AUC: {formatMetric(prData.auc ?? selectedModel?.metrics?.pr_auc)}</span>
                      <span>1.0</span>
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-500 text-center mt-2">
                    Precision vs Recall Curve
                  </p>
                </div>
              ) : (
                <div className="flex justify-center items-center h-48 text-slate-400 text-sm text-center p-4">
                  PR curve coordinates not evaluated for this model artifact
                </div>
              )
            )}
          </div>
        </div>

        {/* 3. Probability Calibration Card */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-3">
              <h4 className="font-bold text-slate-900">Probability Calibration</h4>
              <span className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded-full font-mono">
                Reliability Diagram
              </span>
            </div>

            {hasCalPoints ? (
              <div>
                <div className="relative w-full h-44 bg-slate-50 rounded-xl p-2 border border-slate-100 flex flex-col justify-between">
                  <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                    <line x1="0" y1="100" x2="100" y2="0" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3,3" />
                    <polyline
                      fill="none"
                      stroke="#0ea5e9"
                      strokeWidth="2.5"
                      points={calData.prob_pred.map((x: number, i: number) => {
                        const y = calData.prob_true[i];
                        return `${x * 100},${100 - y * 100}`;
                      }).join(' ')}
                    />
                    {calData.prob_pred.map((x: number, i: number) => {
                      const y = calData.prob_true[i];
                      return (
                        <circle key={i} cx={x * 100} cy={100 - y * 100} r="3" fill="#0284c7" />
                      );
                    })}
                  </svg>
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                    <span>Predicted: 0.0</span>
                    <span className="font-bold text-sky-600">Brier: {formatMetric(selectedModel?.metrics?.brier_score)}</span>
                    <span>1.0</span>
                  </div>
                </div>
                <div className="flex justify-between items-center text-[10px] text-slate-500 mt-2 px-1">
                  <span>Uniform Probability Bins</span>
                  <span>Observed vs Predicted Fraction</span>
                </div>
              </div>
            ) : (
              <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-4 text-xs text-amber-900 flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <strong className="block font-semibold mb-1">Calibration Notice</strong>
                  {selectedModel?.calibration?.reason || "Reliability curve is evaluated on test holdout partition."}
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

