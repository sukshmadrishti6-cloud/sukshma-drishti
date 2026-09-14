import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  FileText,
  ShieldCheck,
  Brain,
  HelpCircle,
  Activity,
  HeartPulse,
  ListChecks,
  Stethoscope,
  Info,
} from 'lucide-react';
import { InferenceResponse, TabType } from '../types';

interface PredictionResultViewProps {
  result: InferenceResponse;
  onReset?: () => void;
  setActiveTab?: (tab: TabType) => void;
}

export const PredictionResultView: React.FC<PredictionResultViewProps> = ({
  result,
  onReset,
  setActiveTab,
}) => {
  const isHighRisk = result.predicted_class === 1;
  const isQuantum = result.model_family === 'quantum' || result.score_semantics === 'uncalibrated_quantum_decision_score';

  const rec = result.recommendation;

  const renderScoreSection = () => {
    if (result.score_semantics === 'uncalibrated_quantum_decision_score') {
      return (
        <div className="p-4 bg-purple-50/80 border border-purple-200 rounded-2xl">
          <div className="flex items-center gap-2 text-purple-900 font-bold text-xs mb-1">
            <Brain className="w-4 h-4 text-purple-600" />
            <span>Quantum Raw Decision Margin</span>
          </div>
          <p className="text-2xl font-extrabold text-purple-950 font-mono">
            {result.decision_score !== undefined && result.decision_score !== null
              ? result.decision_score.toFixed(4)
              : 'N/A'}
          </p>
          <p className="text-[11px] text-purple-700 mt-1">
            Uncalibrated quantum decision score representing classification margin relative to quantum state boundary.
          </p>
        </div>
      );
    } else if (result.score_semantics === 'ensemble_vote_probability') {
      return (
        <div className="p-4 bg-blue-50/80 border border-blue-200 rounded-2xl">
          <div className="flex items-center gap-2 text-blue-900 font-bold text-xs mb-1">
            <Activity className="w-4 h-4 text-blue-600" />
            <span>Ensemble Consensus Score</span>
          </div>
          <p className="text-2xl font-extrabold text-blue-950">
            {result.probability !== undefined && result.probability !== null
              ? `${(result.probability * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
          <p className="text-[11px] text-blue-700 mt-1">
            Ensemble vote probability derived from base model consensus.
          </p>
        </div>
      );
    } else {
      return (
        <div className="p-4 bg-emerald-50/80 border border-emerald-200 rounded-2xl">
          <div className="flex items-center gap-2 text-emerald-900 font-bold text-xs mb-1">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>Calibrated Risk Probability</span>
          </div>
          <p className="text-2xl font-extrabold text-emerald-950">
            {result.probability !== undefined && result.probability !== null
              ? `${(result.probability * 100).toFixed(1)}%`
              : result.decision_score !== undefined && result.decision_score !== null
              ? result.decision_score.toFixed(4)
              : 'N/A'}
          </p>
          <p className="text-[11px] text-emerald-700 mt-1">
            Sigmoid Platt-calibrated probability estimate fit strictly on cross-validation folds.
          </p>
        </div>
      );
    }
  };

  return (
    <div className="space-y-6 select-none">
      {/* Top Banner Card */}
      <div className={`p-6 rounded-[2rem] border shadow-xs ${
        isHighRisk ? 'bg-rose-50/80 border-rose-200' : 'bg-emerald-50/80 border-emerald-200'
      }`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 ${
              isHighRisk ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
            }`}>
              {isHighRisk ? <AlertTriangle className="w-6 h-6" /> : <CheckCircle2 className="w-6 h-6" />}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full ${
                  isHighRisk ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {result.disease ? `${result.disease.toUpperCase()} SCREENING` : 'AI SCREENING RESULT'}
                </span>
                <span className="text-xs text-slate-500 font-mono">ID: {result.inference_id || 'N/A'}</span>
              </div>
              <h2 className={`text-2xl font-extrabold tracking-tight mt-1 ${
                isHighRisk ? 'text-rose-950' : 'text-emerald-950'
              }`}>
                {result.model_output || (isHighRisk ? 'Elevated Risk Detected' : 'Low Risk Detected')}
              </h2>
              <p className="text-xs text-slate-600 mt-1">
                Model: <strong>{result.model_used}</strong> ({result.model_family}) | Artifact: {result.artifact_id} ({result.artifact_version || 'v3.0'})
              </p>
            </div>
          </div>

          {onReset && (
            <button
              onClick={onReset}
              className="px-4 py-2 bg-white hover:bg-slate-50 text-slate-700 rounded-xl text-xs font-semibold border border-slate-200 shadow-xs shrink-0 self-start md:self-auto"
            >
              New Assessment
            </button>
          )}
        </div>
      </div>

      {/* Score and Semantics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {renderScoreSection()}

        <div className="p-4 bg-white border border-slate-200 rounded-2xl shadow-xs">
          <span className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
            Model Family & Type
          </span>
          <p className="text-lg font-bold text-slate-900 capitalize">{result.model_family}</p>
          <span className="text-[11px] text-slate-500 block mt-1">
            Pipeline Version: {result.pipeline_version || 'v2.0'}
          </span>
        </div>

        <div className="p-4 bg-white border border-slate-200 rounded-2xl shadow-xs">
          <span className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
            Inference Latency
          </span>
          <p className="text-lg font-bold text-slate-900">
            {result.audit?.inference_time_ms ? `${result.audit.inference_time_ms.toFixed(1)} ms` : '—'}
          </p>
          <span className="text-[11px] text-slate-500 block mt-1 truncate" title={result.audit?.timestamp}>
            Timestamp: {result.audit?.timestamp ? new Date(result.audit.timestamp).toLocaleTimeString() : 'N/A'}
          </span>
        </div>
      </div>

      {/* Recommendation Section (Phase I Integration) */}
      {rec && (
        <div className="bg-white rounded-[2rem] border border-emerald-100 p-6 shadow-xs space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900 tracking-tight">Personalized Health Guidance</h3>
                <span className="text-[11px] text-slate-500 font-mono">
                  Rule Set Version: {rec.rule_set_version}
                </span>
              </div>
            </div>
            <span className="text-[11px] font-semibold text-emerald-800 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200">
              Deterministic Rules Engine
            </span>
          </div>

          {/* Summary */}
          <div className="p-4 bg-slate-50 border border-slate-200/80 rounded-2xl">
            <p className="text-sm font-semibold text-slate-800 leading-relaxed">{rec.summary}</p>
          </div>

          {/* Priority Actions & Monitoring */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {rec.priority_actions && rec.priority_actions.length > 0 && (
              <div className="p-4 bg-rose-50/50 border border-rose-100 rounded-2xl space-y-2">
                <div className="flex items-center gap-2 font-bold text-rose-900">
                  <ListChecks className="w-4 h-4 text-rose-600" />
                  <span>Priority Actions</span>
                </div>
                <ul className="space-y-1.5 list-disc list-inside text-rose-950">
                  {rec.priority_actions.map((act, idx) => (
                    <li key={idx}>{act}</li>
                  ))}
                </ul>
              </div>
            )}

            {rec.monitoring && rec.monitoring.length > 0 && (
              <div className="p-4 bg-blue-50/50 border border-blue-100 rounded-2xl space-y-2">
                <div className="flex items-center gap-2 font-bold text-blue-900">
                  <Stethoscope className="w-4 h-4 text-blue-600" />
                  <span>Monitoring & Lifestyle Guidance</span>
                </div>
                <ul className="space-y-1.5 list-disc list-inside text-blue-950">
                  {rec.monitoring.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Professional Care Guidance */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="p-4 bg-white border border-slate-200 rounded-2xl space-y-1">
              <span className="font-bold text-slate-900 block">Professional Follow-Up</span>
              <p className="text-slate-600 leading-relaxed">{rec.professional_follow_up}</p>
            </div>
            <div className="p-4 bg-white border border-slate-200 rounded-2xl space-y-1">
              <span className="font-bold text-amber-900 block">When to Seek Care</span>
              <p className="text-amber-800 leading-relaxed">{rec.when_to_seek_care}</p>
            </div>
          </div>

          {/* Factor Attribution */}
          {rec.factors && rec.factors.length > 0 && (
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-2xl space-y-2 text-xs">
              <span className="font-bold text-slate-900 flex items-center gap-1.5">
                <Info className="w-4 h-4 text-slate-500" />
                <span>Key Factors Contributing to Model Guidance</span>
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                {rec.factors.map((f, idx) => (
                  <div key={idx} className="p-2.5 bg-white border border-slate-200 rounded-xl text-slate-700">
                    <span className="font-semibold block text-slate-900">{f.name}</span>
                    <span className="text-[11px] text-slate-500">{f.summary}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="p-3 bg-amber-50/60 border border-amber-200/60 rounded-xl text-[11px] text-amber-800 italic">
            {rec.disclaimer}
          </div>
        </div>
      )}

      {/* Explanation Section */}
      {result.explanation && (
        <div className="bg-white rounded-[2rem] border border-blue-100 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-bold text-slate-900">Feature Importance Explanation</h3>
            {setActiveTab && (
              <button
                onClick={() => setActiveTab('explainability')}
                className="text-xs font-semibold text-blue-600 hover:text-blue-700 bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200/60"
              >
                Full Visual Attribution →
              </button>
            )}
          </div>
          <p className="text-xs text-slate-600">
            Method: <strong>{result.explanation.explanation_method}</strong>
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {result.explanation.contributions?.map((c, idx) => (
              <div key={idx} className="p-3 bg-slate-50 rounded-xl border border-slate-200 flex justify-between items-center">
                <div>
                  <span className="font-bold text-slate-900 block">{c.name}</span>
                  <span className="text-[11px] text-slate-500">Value: {c.value}</span>
                </div>
                <div className="text-right">
                  <span className={`font-mono font-bold block ${c.contribution >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                    {c.contribution >= 0 ? `+${c.contribution.toFixed(3)}` : c.contribution.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-slate-400 capitalize">{c.direction}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
