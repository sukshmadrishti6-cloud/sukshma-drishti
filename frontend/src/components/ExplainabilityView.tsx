import React, { useState, useEffect } from 'react';
import { TabType, InferenceResponse, GlobalExplanation } from '../types';
import { Eye, ShieldAlert, BarChart3, Info } from 'lucide-react';
import { api } from '../api';

interface ExplainabilityViewProps {
  setActiveTab: (tab: TabType) => void;
  result: InferenceResponse | null;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({ setActiveTab, result }) => {
  const expl = result?.explanation;

  const [selectedDisease, setSelectedDisease] = useState<string>('diabetes');
  const [availableModels, setAvailableModels] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedGlobalModel, setSelectedGlobalModel] = useState<string>(
    result?.artifact_id || ''
  );
  const [globalData, setGlobalData] = useState<GlobalExplanation | null>(null);
  const [globalLoading, setGlobalLoading] = useState<boolean>(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    api.getModels(selectedDisease)
      .then(res => {
        if (mounted && res && res.models && res.models.length > 0) {
          const models = res.models.map(m => ({ id: m.model_id, name: m.display_name }));
          setAvailableModels(models);
          if (!result?.artifact_id || !models.some(m => m.id === result.artifact_id)) {
            setSelectedGlobalModel(models[0].id);
          }
        }
      })
      .catch(err => {
        console.warn('Failed to fetch models for explainability:', err);
      });
    return () => { mounted = false; };
  }, [selectedDisease]);

  useEffect(() => {
    if (result?.artifact_id) {
      setSelectedGlobalModel(result.artifact_id);
    }
  }, [result?.artifact_id]);

  useEffect(() => {
    if (!selectedGlobalModel) return;
    let mounted = true;
    setGlobalLoading(true);
    setGlobalError(null);

    api.getGlobalExplanation(selectedGlobalModel)
      .then((data: GlobalExplanation) => {
        if (mounted) {
          setGlobalData(data);
          setGlobalLoading(false);
        }
      })
      .catch((err: any) => {
        if (mounted) {
          setGlobalError(err.message || 'Failed to load global explanation.');
          setGlobalLoading(false);
        }
      });

    return () => { mounted = false; };
  }, [selectedGlobalModel]);

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Eye className="text-blue-500" />
            Model Explainability & Attribution
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Analyze global population-level feature importances and instance-specific local prediction attributions across all diseases.
          </p>
        </div>
      </div>

      {/* Global Feature Importance Section */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-500" />
              Global Population Feature Importance
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              How the trained architecture prioritizes diagnostic factors across the overall training population.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs font-semibold text-slate-500">Disease:</label>
              <select
                value={selectedDisease}
                onChange={(e) => setSelectedDisease(e.target.value)}
                className="border border-slate-200 rounded-xl px-3 py-1.5 text-xs font-semibold bg-slate-50 text-slate-700 focus:ring-2 focus:ring-blue-500/20"
              >
                <option value="diabetes">Diabetes</option>
                <option value="cardiovascular">Cardiovascular</option>
                <option value="cancer">Cancer</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-xs font-semibold text-slate-500">Model:</label>
              <select
                value={selectedGlobalModel}
                onChange={(e) => setSelectedGlobalModel(e.target.value)}
                className="border border-slate-200 rounded-xl px-3 py-1.5 text-xs font-semibold bg-slate-50 text-slate-700 focus:ring-2 focus:ring-blue-500/20"
              >
                {availableModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {globalLoading ? (
          <div className="flex justify-center items-center h-48 text-slate-400 text-sm">
            Calculating feature attribution...
          </div>
        ) : globalError ? (
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800">
            {globalError}
          </div>
        ) : globalData ? (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3 text-xs bg-slate-50 p-3 rounded-xl border border-slate-100">
              <span>Method: <strong className="font-mono text-slate-800">{globalData.method}</strong></span>
              <span>Status: <strong className={`font-semibold capitalize ${globalData.status === 'available' ? 'text-emerald-600' : globalData.status === 'partial' ? 'text-blue-600' : 'text-amber-600'}`}>{globalData.status}</strong></span>
              <span>Pipeline: <strong className="font-mono text-slate-800">{globalData.pipeline_version || 'v2.0'}</strong></span>
            </div>

            {globalData.status === 'unsupported' ? (
              <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-5 text-xs text-amber-900 flex items-start gap-3">
                <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <strong className="block font-bold">Global Importance Unsupported for this Kernel</strong>
                  <ul className="list-disc list-inside space-y-1 mt-1 text-slate-700">
                    {globalData.limitations.map((lim, i) => (
                      <li key={i}>{lim}</li>
                    ))}
                  </ul>
                  <p className="pt-2 text-slate-600">
                    To inspect feature attributions for this model, run an instance in the <strong>Inference</strong> tab to compute local numerical sensitivity gradients.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4 pt-1">
                {/* Method Explanation Callout */}
                {globalData.method === 'permutation_importance' && (
                  <div className="bg-blue-50/60 border border-blue-100 rounded-xl p-3.5 text-xs text-blue-900 flex items-start gap-2.5">
                    <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                    <div>
                      <strong className="block font-semibold">Model-Agnostic Permutation Feature Importance</strong>
                      <p className="mt-0.5 text-slate-600 leading-relaxed">
                        Permutation importance measures the drop in evaluation score (ROC-AUC) on the holdout test set when each feature's values are randomly permuted. Permutation importance provides an empirical, model-agnostic measure of feature reliance for non-linear SVM kernels.
                      </p>
                    </div>
                  </div>
                )}

                {globalData.method === 'standardized_coefficient_magnitude' && (
                  <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 text-xs text-slate-700 flex items-start gap-2">
                    <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      Linear additive regression coefficients in standardized z-score feature space. Magnitudes reflect the relative change in log-odds per standard deviation.
                    </p>
                  </div>
                )}

                {globalData.method === 'mean_decrease_impurity' && (
                  <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 text-xs text-slate-700 flex items-start gap-2">
                    <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      Mean decrease in Gini impurity (MDI) across all decision tree splits in the Random Forest ensemble.
                    </p>
                  </div>
                )}

                {globalData.method === 'gain_feature_importance' && (
                  <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 text-xs text-slate-700 flex items-start gap-2">
                    <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      Total gain metric representing the relative fractional contribution of each feature to the gradient boosted decision trees in XGBoost.
                    </p>
                  </div>
                )}

                {globalData.method === 'quantum_mutual_info_ranking' && (
                  <div className="bg-purple-50/60 border border-purple-100 rounded-xl p-3 text-xs text-purple-900 flex items-start gap-2">
                    <Info className="w-4 h-4 text-purple-600 shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      Non-parametric mutual information ranking utilized during training-fitted SelectKBest projection to construct the quantum feature register.
                    </p>
                  </div>
                )}

                {/* Ranked Feature Bars */}
                <div className="space-y-3 pt-1">
                  {globalData.features.map((feat, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-semibold text-slate-700">
                        <span className="flex items-center gap-2">
                          <span className="text-[10px] font-mono text-slate-400 w-4">{idx + 1}.</span>
                          {feat.name}
                        </span>
                        <span className="font-mono text-slate-500">
                          {feat.importance.toFixed(4)} ({feat.normalized_weight.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            globalData.model_family === 'quantum' ? 'bg-purple-500' : 'bg-blue-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(2, feat.normalized_weight))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {globalData.limitations.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 space-y-1">
                    <span className="font-semibold text-slate-700 flex items-center gap-1">
                      <Info className="w-3.5 h-3.5 text-slate-400" /> Methodological Limitations:
                    </span>
                    <ul className="list-disc list-inside pl-1 space-y-0.5">
                      {globalData.limitations.map((lim, i) => (
                        <li key={i}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : null}
      </div>

      {/* Local Instance Attribution Section */}
      {!expl ? (
        <div className="bg-white rounded-[2rem] border border-orange-100 p-8 shadow-xs flex flex-col justify-center items-center text-center">
          <p className="text-sm font-semibold text-slate-700 mb-1">No Local Prediction Explanation Loaded</p>
          <p className="text-xs text-slate-500 max-w-md">
            Execute a prediction in the Inference tab with <strong>"Generate Explanation"</strong> checked to inspect sample-specific feature contributions.
          </p>
          <button 
            onClick={() => setActiveTab('inference')} 
            className="mt-4 px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-600 rounded-xl text-xs font-semibold transition-colors"
          >
            Go to Inference
          </button>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
              <h3 className="text-base font-bold text-slate-900 mb-3">Local Context</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Local Method</span>
                  <span className="font-semibold font-mono">{expl.explanation_method}</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Status</span>
                  <span className="font-semibold capitalize text-emerald-600">{expl.status}</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Predicted Risk</span>
                  <span className={`font-semibold ${expl.predicted_class === 1 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {expl.predicted_class === 1 ? 'High Risk (1)' : 'Low Risk (0)'}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
              <h3 className="text-base font-bold text-slate-900 mb-3">Inference Target</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Model Name</span>
                  <span className="font-semibold">{expl.model_name}</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Version</span>
                  <span className="font-semibold font-mono">{expl.model_version}</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-500">Family</span>
                  <span className="font-semibold capitalize">{result?.model_family || 'N/A'}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
              <h3 className="text-base font-bold text-slate-900 mb-3">Responsible AI Notice</h3>
              <div className="space-y-2 text-xs">
                <div className="flex gap-2">
                  <ShieldAlert className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                  <p className="text-[11px] text-slate-600 leading-relaxed">
                    Attributions describe mathematical associations within the trained model and do NOT establish medical causality.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-slate-900">Instance Feature Contributions</h3>
              <div className="text-xs text-slate-500">
                Predicted Probability: <strong className="text-slate-800">{result?.probability !== undefined && result?.probability !== null ? `${(result.probability * 100).toFixed(1)}%` : '—'}</strong>
              </div>
            </div>

            <div className="space-y-2">
              {expl.contributions.map((feat: any, i: number) => (
                <div key={i} className="flex justify-between items-center text-xs p-2.5 bg-slate-50 rounded-xl hover:bg-slate-100/60 transition-colors">
                  <div className="w-1/3 font-semibold text-slate-700 truncate" title={feat.name}>
                    {feat.name}
                  </div>
                  <div className="w-1/4 text-slate-500 text-right font-mono">
                    Val: {feat.value?.toFixed(2)}
                  </div>
                  <div className="w-1/4 text-right">
                    <span className="text-[10px] text-slate-400 font-mono mr-2">({feat.normalized_weight?.toFixed(1)}%)</span>
                    <span className={`font-bold font-mono ${
                      feat.direction === 'positive' 
                        ? 'text-red-500' 
                        : feat.direction === 'negative' 
                        ? 'text-emerald-500' 
                        : 'text-slate-400'
                    }`}>
                      {feat.direction === 'positive' ? '+' : ''}{feat.contribution?.toFixed(4)}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {expl.limitations && expl.limitations.length > 0 && (
              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500">
                <span className="font-semibold text-slate-700">Instance Limitations:</span>
                <ul className="list-disc list-inside mt-1 space-y-0.5">
                  {expl.limitations.map((limit: string, idx: number) => (
                    <li key={idx}>{limit}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
