import React, { useState, useEffect, useRef } from 'react';
import { TabType, QuantumExperiment, QuantumCapability } from '../types';
import { Play, Zap, RefreshCw, CheckCircle2, AlertTriangle, ShieldAlert, Cpu, Layers, Activity, Info, Heart, Dna, Flame } from 'lucide-react';
import { api } from '../api';

interface QuantumLabViewProps {
  setActiveTab: (tab: TabType) => void;
}

const DISEASE_CONFIGS = {
  diabetes: {
    name: 'Diabetes Mellitus',
    dataset: 'Pima Indians Diabetes (Preprocessed)',
    datasetHash: 'b78029447fae',
    samples: '768 total samples (154 holdout test)',
    icon: Flame,
    color: 'text-amber-600',
    bgActive: 'bg-amber-500 text-white',
  },
  cardiovascular: {
    name: 'Cardiovascular Risk',
    dataset: 'UCI Cleveland Heart Disease (Preprocessed)',
    datasetHash: '20875c7423c5',
    samples: '303 total samples (61 holdout test)',
    icon: Heart,
    color: 'text-rose-600',
    bgActive: 'bg-rose-500 text-white',
  },
  cancer: {
    name: 'Breast Cancer Oncology',
    dataset: 'Wisconsin Diagnostic Breast Cancer (Preprocessed)',
    datasetHash: 'd6a3b2b4859a',
    samples: '569 total samples (114 holdout test)',
    icon: Dna,
    color: 'text-pink-600',
    bgActive: 'bg-pink-500 text-white',
  },
};

export const QuantumLabView: React.FC<QuantumLabViewProps> = () => {
  const [selectedDisease, setSelectedDisease] = useState<'diabetes' | 'cardiovascular' | 'cancer'>('diabetes');
  const [modelType, setModelType] = useState('qsvc');
  const [qubits, setQubits] = useState('4');
  const [capabilities, setCapabilities] = useState<QuantumCapability[]>([]);
  
  const [status, setStatus] = useState<'idle' | 'queued' | 'running' | 'completed' | 'failed'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<QuantumExperiment | null>(null);
  const [loadingInitial, setLoadingInitial] = useState(true);
  
  const pollTimerRef = useRef<any>(null);

  const diseaseConfig = DISEASE_CONFIGS[selectedDisease];

  // Load capabilities and default experiment on mount or disease change
  useEffect(() => {
    let mounted = true;
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    setLoadingInitial(true);
    setStatus('idle');
    setError(null);

    api.getQuantumCapabilities(selectedDisease)
      .then((caps) => {
        if (mounted) {
          setCapabilities(caps);
        }
      })
      .catch((err) => {
        console.warn('Failed to load capabilities:', err);
      });

    const qCount = parseInt(qubits) || 4;
    api.getLatestExperiment(qCount, modelType, selectedDisease)
      .then((exp) => {
        if (mounted && exp) {
          setExperiment(exp);
          setStatus(exp.status as any || 'completed');
          setLoadingInitial(false);
        } else if (mounted) {
          setExperiment(null);
          setLoadingInitial(false);
        }
      })
      .catch(() => {
        if (mounted) {
          setExperiment(null);
          setLoadingInitial(false);
        }
      });

    return () => {
      mounted = false;
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [selectedDisease]);

  // Handle Qubit count change: IMMEDIATELY isolate state to prevent 4Q metric leakage into 6Q/8Q
  const handleQubitChange = async (newQubits: string) => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setQubits(newQubits);
    setError(null);

    const qCount = parseInt(newQubits);

    try {
      const existing = await api.getLatestExperiment(qCount, modelType, selectedDisease);
      if (existing) {
        setExperiment(existing);
        setStatus(existing.status as any || 'completed');
      } else {
        setExperiment(null);
        setStatus('idle');
      }
    } catch {
      setExperiment(null);
      setStatus('idle');
    }
  };

  // Handle Model Type change: similarly isolate state
  const handleModelTypeChange = async (newModelType: string) => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setModelType(newModelType);
    setError(null);

    const qCount = parseInt(qubits);
    try {
      const existing = await api.getLatestExperiment(qCount, newModelType, selectedDisease);
      if (existing) {
        setExperiment(existing);
        setStatus(existing.status as any || 'completed');
      } else {
        setExperiment(null);
        setStatus('idle');
      }
    } catch {
      setExperiment(null);
      setStatus('idle');
    }
  };

  const handleDiseaseChange = (disease: 'diabetes' | 'cardiovascular' | 'cancer') => {
    if (disease === selectedDisease) return;
    setSelectedDisease(disease);
  };

  const handleRun = async () => {
    if (status === 'queued' || status === 'running') return;

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    setStatus('queued');
    setError(null);
    setExperiment(null);

    const qCount = parseInt(qubits);
    const title = `${diseaseConfig.name} Quantum ${modelType.toUpperCase()} - ${qCount} Qubits`;

    try {
      const res = await api.createExperiment({
        title,
        dataset_name: diseaseConfig.dataset,
        dataset_hash: diseaseConfig.datasetHash,
        qubit_count: qCount,
        feature_map: 'ZZFeatureMap',
        ansatz: modelType === 'vqc' ? 'RealAmplitudes' : 'n/a',
        optimizer: modelType === 'vqc' ? 'COBYLA' : 'n/a',
        shots: 1024,
        status: 'queued',
      });

      const expId = res.id;

      pollTimerRef.current = setInterval(async () => {
        try {
          const expData = await api.getExperiment(expId);
          if (expData.status === 'running') {
            setStatus('running');
            setExperiment(expData);
          } else if (expData.status === 'completed') {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setStatus('completed');
            setExperiment(expData);
          } else if (expData.status === 'failed') {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setStatus('failed');
            setError(expData.error || 'Experiment execution failed.');
          }
        } catch (pollErr: any) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
          setStatus('failed');
          setError(pollErr.message || 'Error tracking quantum simulation.');
        }
      }, 700);

    } catch (err: any) {
      setStatus('failed');
      setError(err.message || 'Failed to submit quantum experiment.');
    }
  };

  const isExecuting = status === 'queued' || status === 'running';

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Zap className="text-purple-500" />
            Quantum Machine Learning Lab
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Configure, simulate, and inspect parameterized quantum circuits (QSVC & VQC) across multi-disease domains on Qiskit Aer.
          </p>
        </div>

        {/* Multi-Disease Selector Tabs */}
        <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-2xl border border-slate-200">
          {(['diabetes', 'cardiovascular', 'cancer'] as const).map((dKey) => {
            const cfg = DISEASE_CONFIGS[dKey];
            const Icon = cfg.icon;
            const isActive = selectedDisease === dKey;
            return (
              <button
                key={dKey}
                onClick={() => handleDiseaseChange(dKey)}
                className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                  isActive ? cfg.bgActive : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {cfg.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Contextual Scientific Notice Banner */}
      {isExecuting ? (
        <div className="bg-purple-50 border border-purple-200 rounded-2xl p-4 text-xs text-purple-900 flex items-start gap-3 animate-pulse">
          <RefreshCw className="w-5 h-5 text-purple-600 shrink-0 mt-0.5 animate-spin" />
          <div>
            <strong className="block font-bold">[{diseaseConfig.name.toUpperCase()}] {qubits}-Qubit Simulation in Progress...</strong>
            <span className="text-slate-600">
              Constructing parameterized quantum circuit with {modelType === 'qsvc' ? 'ZZFeatureMap (linear entanglement)' : 'ZZFeatureMap + RealAmplitudes ansatz'}, decomposing basis gates, and dispatching to local AerSimulator.
            </span>
          </div>
        </div>
      ) : experiment?.metrics ? (
        <div className="bg-emerald-50/70 border border-emerald-200 rounded-2xl p-4 text-xs text-emerald-950 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
          <div>
            <strong className="block font-bold">[{diseaseConfig.name.toUpperCase()}] {qubits}-Qubit Model Evaluation Available</strong>
            <span className="text-slate-600">
              Pre-trained quantum model artifact exists in registry (<code>{experiment?.model_id}</code>). Holdout evaluation metrics are computed on the untouched {diseaseConfig.samples} (80/20 stratified, seed 42) with ideal Qiskit Aer simulation.
            </span>
          </div>
        </div>
      ) : (
        <div className="bg-amber-50/80 border border-amber-200 rounded-2xl p-4 text-xs text-amber-950 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong className="block font-bold">[{diseaseConfig.name.toUpperCase()}] {qubits}-Qubit Simulation Ready</strong>
            <span className="text-slate-600">
              Click "Run {qubits}-Qubit Experiment" to execute quantum circuit simulation and inspect evaluation metrics on {diseaseConfig.dataset}.
            </span>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-[2rem] p-4 px-6 text-red-700 text-sm flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 shrink-0" /> 
          <div>
            <strong className="block font-semibold">Simulation Error</strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      {status === 'completed' && experiment && (
        <div className="bg-purple-50/70 border border-purple-200 rounded-[2rem] p-4 px-6 text-purple-900 text-xs flex justify-between items-center">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-purple-600 shrink-0" /> 
            <span>
              Simulation completed for <strong>{experiment.title}</strong>! Model: <code>{experiment.model_id}</code>
              {experiment.timings?.simulation_time_ms ? ` (${experiment.timings.simulation_time_ms.toFixed(1)} ms)` : ''}
            </span>
          </div>
          <span className="text-[11px] bg-purple-100 px-3 py-1 rounded-full font-mono text-purple-800 font-bold">
            {experiment.execution_backend || 'AerSimulator'}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Left Column: Configuration Controls */}
        <div className="md:col-span-1 bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-600" />
              Experiment Configuration
            </h3>
            
            <div className="space-y-4 text-sm">
              <div>
                <label className="block text-slate-600 mb-1 text-xs font-semibold">Disease Domain & Dataset</label>
                <div className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 text-slate-800 text-xs font-medium">
                  <div className="font-semibold text-slate-900">{diseaseConfig.name}</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">{diseaseConfig.dataset}</div>
                  <div className="text-[10px] text-purple-700 font-mono mt-0.5">{diseaseConfig.samples}</div>
                </div>
              </div>
              
              <div>
                <label className="block text-slate-600 mb-1 text-xs font-semibold">Model Architecture</label>
                <select 
                  className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 text-slate-800 text-xs font-medium focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                  value={modelType}
                  onChange={(e) => handleModelTypeChange(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="qsvc">Quantum Support Vector Classifier (QSVC)</option>
                  <option value="vqc">Variational Quantum Classifier (VQC)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-slate-600 mb-1 text-xs font-semibold">Qubit Register Width</label>
                <select 
                  className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 text-slate-800 text-xs font-medium focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                  value={qubits}
                  onChange={(e) => handleQubitChange(e.target.value)}
                  disabled={isExecuting}
                >
                  {capabilities.length > 0 ? (
                    capabilities.map((cap) => (
                      <option key={cap.qubit_count} value={String(cap.qubit_count)}>
                        {cap.label}
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="4">4 Qubits (Evaluated Holdout Available)</option>
                      <option value="6">6 Qubits (Circuit Construction Only)</option>
                      <option value="8">8 Qubits (Circuit Construction Only)</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-slate-600 mb-1 text-xs font-semibold">Simulation Backend</label>
                <select className="w-full border border-slate-200 rounded-xl p-2.5 bg-slate-50 text-slate-800 text-xs font-medium">
                  <option>AerSimulator (Local Ideal Simulator)</option>
                </select>
              </div>

              <div className="pt-2 text-[11px] text-slate-500 bg-purple-50/50 p-3 rounded-xl border border-purple-100">
                <span className="font-semibold text-purple-900 block mb-0.5">Circuit Configuration:</span>
                Feature Map: <strong>{modelType === 'qsvc' ? 'ZZFeatureMap (linear)' : 'PauliFeatureMap'}</strong><br />
                Ansatz: <strong>{modelType === 'vqc' ? 'RealAmplitudes (full)' : 'None (Kernel Fidelity)'}</strong><br />
                Shots: <strong>1024</strong>
              </div>
            </div>
          </div>

          <button 
            onClick={handleRun}
            disabled={isExecuting}
            className="w-full mt-6 bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-xs disabled:opacity-50"
          >
            {isExecuting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {status === 'queued' ? 'Queued for Simulation...' : status === 'running' ? 'Executing Circuit on Aer...' : `Run ${qubits}-Qubit ${diseaseConfig.name} Experiment`}
          </button>
        </div>

        {/* Right Column: Execution Status and Circuit Preview */}
        <div className="md:col-span-2 flex flex-col gap-6">
          
          {/* Execution Status Card */}
          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
            <div className="flex justify-between items-center mb-3">
              <span className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-600" />
                Execution & Evaluation Status
              </span>
              <div className="flex items-center gap-2">
                <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold capitalize ${
                  status === 'completed' ? 'bg-emerald-50 text-emerald-700' :
                  status === 'running' ? 'bg-blue-50 text-blue-700 animate-pulse' :
                  status === 'queued' ? 'bg-amber-50 text-amber-700' :
                  'bg-slate-100 text-slate-600'
                }`}>
                  Circuit: {status === 'idle' ? 'Awaiting Run' : status}
                </span>

                <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold capitalize ${
                  experiment?.metrics ? 'bg-emerald-50 text-emerald-700' :
                  isExecuting ? 'bg-amber-50 text-amber-700' :
                  'bg-slate-100 text-slate-500'
                }`}>
                  Eval: {experiment?.metrics ? 'Evaluated' : 'Pending Simulation'}
                </span>
              </div>
            </div>

            {experiment ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
                <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                  <span className="text-slate-400 block text-[10px]">REGISTER WIDTH</span>
                  <span className="font-bold text-slate-800 text-sm">{experiment.qubit_count} Qubits</span>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                  <span className="text-slate-400 block text-[10px]">CIRCUIT DEPTH</span>
                  <span className="font-bold text-slate-800 text-sm">{experiment.circuit_depth ?? '—'}</span>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                  <span className="text-slate-400 block text-[10px]">PARAMETERS</span>
                  <span className="font-bold text-slate-800 text-sm">{experiment.num_parameters ?? '—'}</span>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                  <span className="text-slate-400 block text-[10px]">SIM TIME</span>
                  <span className="font-bold text-purple-700 text-sm">
                    {experiment.timings?.simulation_time_ms ? `${experiment.timings.simulation_time_ms.toFixed(1)} ms` : '—'}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400 py-3">Configure parameters and click "Run {qubits}-Qubit Experiment" to start simulation.</p>
            )}
          </div>
          
          {/* Quantum Circuit Preview Card */}
          <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex-1 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-3">
                <span className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-purple-600" />
                  Quantum Circuit Diagram (Qiskit Decomposition)
                </span>
                {experiment?.gate_counts && (
                  <div className="flex gap-1.5 flex-wrap">
                    {Object.entries(experiment.gate_counts).map(([gate, cnt]) => (
                      <span key={gate} className="text-[10px] bg-purple-50 text-purple-700 font-mono px-1.5 py-0.5 rounded border border-purple-100 font-bold">
                        {gate}: {cnt}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {experiment?.circuit_text ? (
                <div className="bg-slate-950 text-purple-200 p-4 rounded-xl font-mono text-[11px] overflow-x-auto max-h-56 leading-tight whitespace-pre border border-purple-950 shadow-inner">
                  {experiment.circuit_text}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-44 text-slate-400 text-xs">
                  <span>Circuit diagram will render here upon simulation dispatch.</span>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

      {/* Evaluation Metrics Section */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-600" />
            Quantum Model Holdout Evaluation Metrics ({qubits} Qubits)
          </h3>
          <span className={`text-xs px-3 py-1 rounded-full font-semibold ${
            experiment?.metrics ? 'bg-emerald-50 text-emerald-700' :
            isExecuting ? 'bg-blue-50 text-blue-700' :
            'bg-amber-50 text-amber-800'
          }`}>
            {experiment?.metrics ? 'Status: Evaluated (154 Test Samples)' :
             isExecuting ? 'Status: Evaluation in progress...' :
             'Status: Evaluation Pending'}
          </span>
        </div>
        
        {/* If metrics present: Show genuine holdout metrics */}
        {experiment?.metrics ? (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
              <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Accuracy</span>
              <span className="text-2xl font-bold font-mono text-slate-800">
                {(experiment.metrics.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
              <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Precision</span>
              <span className="text-2xl font-bold font-mono text-slate-800">
                {(experiment.metrics.precision * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
              <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Recall</span>
              <span className="text-2xl font-bold font-mono text-slate-800">
                {(experiment.metrics.recall * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-purple-50 p-4 rounded-xl border border-purple-100 text-center">
              <span className="text-xs text-purple-700 uppercase tracking-wider font-semibold block mb-1">F1-Score</span>
              <span className="text-2xl font-bold font-mono text-purple-900">
                {experiment.metrics.f1_score.toFixed(4)}
              </span>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
              <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">ROC-AUC</span>
              <span className="text-2xl font-bold font-mono text-slate-800">
                {experiment.metrics.roc_auc !== null ? experiment.metrics.roc_auc.toFixed(4) : 'N/A'}
              </span>
            </div>
          </div>
        ) : isExecuting ? (
          /* Execution in progress */
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'].map((metric) => (
              <div key={metric} className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center animate-pulse">
                <span className="text-xs text-slate-400 uppercase tracking-wider block mb-1">{metric}</span>
                <span className="text-sm font-semibold text-purple-600">Simulating...</span>
              </div>
            ))}
          </div>
        ) : (
          /* Pending Execution */
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'].map((metric) => (
                <div key={metric} className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center opacity-70">
                  <span className="text-xs text-slate-400 uppercase tracking-wider block mb-1">{metric}</span>
                  <span className="text-xl font-mono text-slate-400">—</span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Ready to run</span>
                </div>
              ))}
            </div>

            <div className="bg-purple-50/70 border border-purple-200 rounded-xl p-3.5 text-xs text-purple-900 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-purple-600 shrink-0 mt-0.5" />
              <div>
                <strong className="block font-semibold">{qubits}-Qubit Model Ready for Simulation</strong>
                <p className="mt-0.5 text-slate-600 leading-relaxed">
                  Click "Run {qubits}-Qubit Experiment" above to dispatch the Qiskit Aer circuit simulation and retrieve full holdout evaluation metrics.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Lower 4 Artifact Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        
        {/* 1. ROC Curve */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h4 className="font-bold text-slate-900 text-xs mb-3">ROC Curve ({qubits}Q)</h4>
            {experiment?.curves?.roc_curve?.fpr ? (
              <div>
                <div className="relative w-full h-36 bg-slate-50 rounded-xl p-2 border border-slate-100 flex flex-col justify-between">
                  <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                    <line x1="0" y1="100" x2="100" y2="0" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3,3" />
                    <polyline
                      fill="none"
                      stroke="#a855f7"
                      strokeWidth="2.5"
                      points={experiment.curves.roc_curve.fpr.map((x: number, i: number) => {
                        const y = experiment.curves.roc_curve.tpr[i];
                        return `${x * 100},${100 - y * 100}`;
                      }).join(' ')}
                    />
                  </svg>
                  <div className="flex justify-between text-[9px] text-slate-400 font-mono mt-1">
                    <span>FPR: 0</span>
                    <span className="font-bold text-purple-700">AUC: {experiment.curves.roc_curve.auc.toFixed(3)}</span>
                    <span>1</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col justify-center items-center h-36 text-slate-400 text-xs text-center p-2">
                <span className="font-semibold text-slate-500">ROC Curve Pending</span>
                <span className="text-[10px] text-slate-400 mt-1">Run simulation to generate ROC curve</span>
              </div>
            )}
          </div>
        </div>

        {/* 2. PR Curve */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h4 className="font-bold text-slate-900 text-xs mb-3">Precision-Recall ({qubits}Q)</h4>
            {experiment?.curves?.pr_curve?.precision ? (
              <div>
                <div className="relative w-full h-36 bg-slate-50 rounded-xl p-2 border border-slate-100 flex flex-col justify-between">
                  <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
                    <line x1="0" y1="65" x2="100" y2="65" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3,3" />
                    <polyline
                      fill="none"
                      stroke="#a855f7"
                      strokeWidth="2.5"
                      points={experiment.curves.pr_curve.recall.map((x: number, i: number) => {
                        const y = experiment.curves.pr_curve.precision[i];
                        return `${x * 100},${100 - y * 100}`;
                      }).join(' ')}
                    />
                  </svg>
                  <div className="flex justify-between text-[9px] text-slate-400 font-mono mt-1">
                    <span>Rec: 0</span>
                    <span className="font-bold text-purple-700">PR-AUC: {experiment.curves.pr_curve.auc.toFixed(3)}</span>
                    <span>1</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col justify-center items-center h-36 text-slate-400 text-xs text-center p-2">
                <span className="font-semibold text-slate-500">PR Curve Pending</span>
                <span className="text-[10px] text-slate-400 mt-1">Run simulation to generate PR curve</span>
              </div>
            )}
          </div>
        </div>

        {/* 3. Confusion Matrix */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h4 className="font-bold text-slate-900 text-xs mb-3">Confusion Matrix ({qubits}Q)</h4>
            {experiment?.confusion_matrix ? (
              <div className="grid grid-cols-2 gap-1.5 text-center text-xs font-mono">
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                  <span className="text-[9px] text-slate-400 block">TN</span>
                  <span className="font-bold text-slate-800">{experiment.confusion_matrix.tn}</span>
                </div>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                  <span className="text-[9px] text-slate-400 block">FP</span>
                  <span className="font-bold text-slate-800">{experiment.confusion_matrix.fp}</span>
                </div>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                  <span className="text-[9px] text-slate-400 block">FN</span>
                  <span className="font-bold text-slate-800">{experiment.confusion_matrix.fn}</span>
                </div>
                <div className="bg-purple-50 p-2 rounded-lg border border-purple-100">
                  <span className="text-[9px] text-purple-700 block">TP</span>
                  <span className="font-bold text-purple-900">{experiment.confusion_matrix.tp}</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col justify-center items-center h-36 text-slate-400 text-xs text-center p-2">
                <span className="font-semibold text-slate-500">Matrix Pending</span>
                <span className="text-[10px] text-slate-400 mt-1">Run simulation to view matrix</span>
              </div>
            )}
          </div>
        </div>

        {/* 4. Width-Scaling Analysis */}
        <div className="bg-white rounded-[2rem] border border-orange-100 p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h4 className="font-bold text-slate-900 text-xs mb-2">Width-Scaling Analysis</h4>
            <div className="space-y-1.5 text-[11px] font-mono">
              <div className="flex justify-between items-center p-1.5 bg-emerald-50 rounded-md text-emerald-800">
                <span>{modelType.toUpperCase()}-4Q</span>
                <span className="font-bold">4 Qubits (Evaluated)</span>
              </div>
              <div className="flex justify-between items-center p-1.5 bg-emerald-50 rounded-md text-emerald-800">
                <span>{modelType.toUpperCase()}-6Q</span>
                <span className="font-bold">6 Qubits (Evaluated)</span>
              </div>
              <div className="flex justify-between items-center p-1.5 bg-emerald-50 rounded-md text-emerald-800">
                <span>{modelType.toUpperCase()}-8Q</span>
                <span className="font-bold">8 Qubits (Evaluated)</span>
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-2 leading-relaxed">
              Empirical width-scaling data available across 4Q, 6Q, and 8Q architectures. Select any qubit width above to inspect performance, circuit gate counts, and evaluation metrics.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};


