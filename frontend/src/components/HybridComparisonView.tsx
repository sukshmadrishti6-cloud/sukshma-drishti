import React, { useState, useEffect } from 'react';
import { TabType, BenchmarkSummary } from '../types';
import { Scale, BarChart2, CheckCircle2, ShieldAlert, Loader2, ArrowRight, Cpu, Layers, Heart, Activity as DiabetesIcon, Dna } from 'lucide-react';
import { api } from '../api';

interface HybridComparisonViewProps {
  setActiveTab: (tab: TabType) => void;
}

export const HybridComparisonView: React.FC<HybridComparisonViewProps> = ({ setActiveTab }) => {
  const [selectedDisease, setSelectedDisease] = useState<string>('diabetes');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [benchmark, setBenchmark] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const diseases = [
    { id: 'diabetes', name: 'Diabetes', icon: DiabetesIcon },
    { id: 'cardiovascular', name: 'Cardiovascular', icon: Heart },
    { id: 'cancer', name: 'Cancer', icon: Dna },
  ];

  const loadData = (diseaseId: string) => {
    setLoading(true);
    setError(null);
    api.getHybridComparison(diseaseId)
      .then(data => {
        setBenchmark(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load hybrid comparison');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData(selectedDisease);
  }, [selectedDisease]);

  const results = benchmark?.results || [];
  const filteredResults = selectedCategory === 'All'
    ? results
    : results.filter((r: any) => {
        if (selectedCategory === 'Quantum ML') return r.is_quantum || r.category === 'Quantum ML';
        return r.category === selectedCategory;
      });

  const classicalResults = results.filter((r: any) => !r.is_quantum && r.f1_score !== undefined && r.f1_score !== null);
  const quantumResults = results.filter((r: any) => r.is_quantum && r.f1_score !== undefined && r.f1_score !== null);

  let bestF1Model: any = null;
  let bestAucModel: any = null;

  if (results.length > 0) {
    const evaluatedOnly = results.filter((r: any) => r.f1_score !== undefined && r.f1_score !== null);
    if (evaluatedOnly.length > 0) {
      bestF1Model = [...evaluatedOnly].sort((a: any, b: any) => (b.f1_score || 0) - (a.f1_score || 0))[0];
      bestAucModel = [...evaluatedOnly].sort((a: any, b: any) => (b.roc_auc || 0) - (a.roc_auc || 0))[0];
    }
  }

  const avgClassicalF1 = classicalResults.length 
    ? classicalResults.reduce((acc: number, r: any) => acc + (r.f1_score || 0), 0) / classicalResults.length 
    : null;
  const avgQuantumF1 = quantumResults.length 
    ? quantumResults.reduce((acc: number, r: any) => acc + (r.f1_score || 0), 0) / quantumResults.length 
    : null;

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Scale className="text-orange-500" />
            Hybrid Quantum vs Classical Comparison
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Empirical comparative benchmark between classical ML and quantum machine learning architectures across disease domains.
          </p>
        </div>
        <button
          onClick={() => setActiveTab('classical-baselines')}
          className="bg-orange-50 hover:bg-orange-100 text-orange-600 px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shrink-0"
        >
          View Detailed Benchmarks <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Disease Tabs */}
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
              {d.name} Comparison
            </button>
          );
        })}
      </div>
      
      {loading && (
        <div className="flex justify-center items-center h-48">
          <Loader2 className="w-8 h-8 text-orange-500 animate-spin" />
        </div>
      )}

      {(error || (benchmark && !benchmark.available)) && !loading && (
        <div className="bg-amber-50 border border-amber-200 rounded-[2rem] p-6 text-amber-800 text-sm flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" /> 
          <div>
            <strong className="block mb-1">Unavailable API Data</strong>
            {error || benchmark?.message || "Hybrid benchmark comparison cannot be loaded."}
          </div>
        </div>
      )}

      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Common Evaluation Protocol</h3>
            <p className="text-sm text-slate-600 leading-relaxed mb-4">
              All models for {selectedDisease.toUpperCase()} are evaluated on identical 80/20 stratified holdout test split partitions with leakage-safe pipeline transformations fitted exclusively on training data.
            </p>
            <div className="text-xs text-slate-500 font-mono">
              Status: <span className="text-emerald-600 font-bold">{benchmark?.available ? 'Available (Holdout Enforced)' : 'Awaiting benchmark generation'}</span>
            </div>
          </div>

          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Quantum Transparency & Disclosure</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Simulated on <strong>Qiskit Aer</strong> (local ideal state-vector simulator). QSVC uses 2-repetition linear <code className="font-mono text-purple-700 bg-purple-50 px-1 rounded">ZZFeatureMap</code> with fidelity kernel. VQC uses <code className="font-mono text-purple-700 bg-purple-50 px-1 rounded">RealAmplitudes</code> ansatz with COBYLA optimization (1024 shots). Uncalibrated quantum raw decision scores (<code className="font-mono text-purple-700 bg-purple-50 px-1 rounded">uncalibrated_quantum_decision_score</code>) are decision margins, never percentage probabilities.
            </p>
            <div className="text-xs text-slate-500 font-mono mt-4">
              Simulation: <span className="text-purple-600 font-bold">Local Qiskit Aer (4-Qubit Primary Register)</span>
            </div>
          </div>
        </div>
      )}

      {/* Model Performance Comparison Matrix Table */}
      {!loading && (
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs overflow-x-auto">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-orange-500" />
              Unified Model Comparison Matrix ({selectedDisease.toUpperCase()})
            </h3>
            <div className="flex flex-wrap gap-1.5 bg-slate-100 p-1 rounded-xl text-xs">
              {['All', 'Baseline Classical', 'Advanced Classical', 'Ensemble Classical', 'Optimized Classical', 'Quantum ML'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                    selectedCategory === cat
                      ? 'bg-white text-slate-900 shadow-2xs'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 font-semibold rounded-tl-xl">Category</th>
                <th className="px-4 py-3 font-semibold">Model</th>
                <th className="px-4 py-3 font-semibold">Accuracy</th>
                <th className="px-4 py-3 font-semibold">F1 Score</th>
                <th className="px-4 py-3 font-semibold">ROC-AUC</th>
                <th className="px-4 py-3 font-semibold">PR-AUC</th>
                <th className="px-4 py-3 font-semibold">Inference Latency</th>
                <th className="px-4 py-3 font-semibold rounded-tr-xl">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredResults.length > 0 ? (
                filteredResults.map((r: any) => {
                  const isEval = r.evaluated !== false && r.f1_score !== undefined && r.f1_score !== null;
                  const isProd = r.production === true;
                  return (
                    <tr key={r.model_id} className={`hover:bg-slate-50/50 transition-colors ${isProd ? 'bg-orange-50/30' : ''}`}>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-md text-xs font-semibold ${
                          r.category === 'Quantum ML' || r.is_quantum ? 'bg-purple-50 text-purple-700' :
                          r.category === 'Optimized Classical' ? 'bg-amber-50 text-amber-700' :
                          r.category === 'Ensemble Classical' ? 'bg-indigo-50 text-indigo-700' :
                          'bg-slate-100 text-slate-700'
                        }`}>
                          {r.category || (r.is_quantum ? 'Quantum ML' : 'Classical')}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-medium text-slate-900 flex items-center gap-2">
                        {r.model_name || r.model_id}
                        {isProd && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-orange-100 text-orange-700 border border-orange-200">
                            ★ Production Winner
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-slate-700 font-mono">{isEval && r.accuracy !== undefined && r.accuracy !== null ? (r.accuracy * 100).toFixed(1) + '%' : '—'}</td>
                      <td className="px-4 py-3 text-slate-700 font-mono font-bold">{isEval ? r.f1_score.toFixed(4) : '—'}</td>
                      <td className="px-4 py-3 text-slate-700 font-mono">{isEval && r.roc_auc !== undefined && r.roc_auc !== null ? r.roc_auc.toFixed(4) : '—'}</td>
                      <td className="px-4 py-3 text-slate-700 font-mono">{isEval && r.pr_auc !== undefined && r.pr_auc !== null ? r.pr_auc.toFixed(4) : '—'}</td>
                      <td className="px-4 py-3 text-slate-700 font-mono">{r.computational_cost || '—'}</td>
                      <td className="px-4 py-3">
                        {isEval ? (
                          <span className="px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-md text-xs font-semibold">Evaluated</span>
                        ) : (
                          <span className="px-2 py-0.5 bg-purple-50 text-purple-700 rounded-md text-xs font-semibold">Circuit Only</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-slate-400">
                    No models match the selected category filter for this disease.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Visual Comparison & Observed Findings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Visual Comparison Chart */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-bold text-slate-900 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-orange-500" />
                Classical vs Quantum F1 & ROC-AUC
              </h4>
              <div className="flex items-center gap-2 text-[11px]">
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-xs bg-orange-500" /> F1</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-xs bg-blue-500" /> ROC-AUC</span>
              </div>
            </div>

            {results.filter((r: any) => r.f1_score !== undefined && r.f1_score !== null).length > 0 ? (
              <div className="space-y-4 pt-1">
                {results.filter((r: any) => r.f1_score !== undefined && r.f1_score !== null).slice(0, 8).map((m: any) => {
                  const f1Pct = ((m.f1_score || 0) * 100);
                  const aucPct = ((m.roc_auc || 0) * 100);
                  return (
                    <div key={m.model_id} className="space-y-1">
                      <div className="flex justify-between text-xs font-semibold">
                        <span className="flex items-center gap-1.5">
                          <span className={`w-1.5 h-1.5 rounded-full ${m.is_quantum ? 'bg-purple-500' : 'bg-slate-400'}`} />
                          <span className="text-slate-800">{m.model_name || m.model_id}</span>
                          {m.is_quantum && <span className="text-[10px] text-purple-600 bg-purple-50 px-1.5 rounded">{m.qubits ? `${m.qubits}Q` : 'QML'}</span>}
                        </span>
                        <span className="font-mono text-slate-500 text-[11px]">
                          F1: {(m.f1_score || 0).toFixed(3)} | AUC: {m.roc_auc ? m.roc_auc.toFixed(3) : 'N/A'}
                        </span>
                      </div>
                      
                      <div className="space-y-1">
                        <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-orange-500 h-full rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(1, f1Pct))}%` }}
                          />
                        </div>
                        {m.roc_auc && (
                          <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                            <div
                              className="bg-blue-500 h-full rounded-full transition-all duration-500"
                              style={{ width: `${Math.min(100, Math.max(1, aucPct))}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex justify-center items-center h-52 text-slate-400 text-sm">
                Awaiting benchmark metrics to generate comparison chart.
              </div>
            )}
          </div>
        </div>

        {/* Observed Findings & Reproducibility Metadata */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between space-y-6">
          <div>
            <h4 className="font-bold text-slate-900 mb-3 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              Observed Empirical Findings ({selectedDisease.toUpperCase()})
            </h4>

            {results.length > 0 ? (
              <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                  <span className="font-semibold text-slate-800 block mb-0.5">Top-Performing Model:</span>
                  {bestF1Model ? (
                    <span>
                      <strong>{bestF1Model.model_name}</strong> achieved the highest F1-Score (<strong>{bestF1Model.f1_score.toFixed(4)}</strong>) and <strong>{bestAucModel?.model_name || bestF1Model.model_name}</strong> achieved the highest ROC-AUC (<strong>{bestAucModel?.roc_auc?.toFixed(4) || bestF1Model.roc_auc?.toFixed(4) || '—'}</strong>).
                    </span>
                  ) : <span>Evaluation metrics loading.</span>}
                </div>

                <div className="bg-purple-50/50 p-3 rounded-xl border border-purple-100">
                  <span className="font-semibold text-purple-900 block mb-0.5">Classical vs Quantum Contrast:</span>
                  <span>
                    Classical models averaged F1-Score of <strong>{avgClassicalF1 ? avgClassicalF1.toFixed(3) : '—'}</strong> compared to <strong>{avgQuantumF1 ? avgQuantumF1.toFixed(3) : '—'}</strong> for quantum simulation.
                  </span>
                </div>

                <div className="bg-amber-50/50 p-3 rounded-xl border border-amber-100">
                  <span className="font-semibold text-amber-900 block mb-0.5">Quantum Advantage Assessment:</span>
                  <span>
                    No empirical quantum advantage is observed on tabular clinical datasets under 4-qubit local simulation. Classical regularized linear and tree ensembles outperform current NISQ variational circuits.
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400">No findings recorded for this partition yet.</p>
            )}
          </div>

          <div className="pt-4 border-t border-slate-100">
            <h4 className="font-bold text-slate-900 mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-slate-600" />
              Reproducibility Metadata
            </h4>
            
            <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600 font-mono">
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px]">DISEASE PARTITION</span>
                <span className="font-semibold text-slate-800 capitalize">{selectedDisease}</span>
                <span className="text-[10px] text-slate-500">Holdout: 80/20</span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px]">SPLIT SEED</span>
                <span className="font-semibold text-slate-800">Seed 42</span>
                <span className="text-[10px] text-slate-500 block">Stratified Holdout</span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px]">PIPELINE VERSION</span>
                <span className="font-semibold text-slate-800">v2.0 (Leakage-Free)</span>
                <span className="text-[10px] text-slate-500 block">Train-Fitted Scalers</span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[10px]">QUANTUM BACKEND</span>
                <span className="font-semibold text-slate-800">Qiskit Aer (Local)</span>
                <span className="text-[10px] text-slate-500 block">Ideal Simulator | 1024 Shots</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
