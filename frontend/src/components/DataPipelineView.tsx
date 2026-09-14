import React, { useState, useEffect } from 'react';
import { TabType, PipelineMetadata } from '../types';
import { GitMerge, CheckCircle2, ArrowRight, Database, ShieldAlert, Loader2, Activity, Heart, Sparkles, Layers, BarChart2 } from 'lucide-react';
import { api } from '../api';

interface DataPipelineViewProps {
  setActiveTab: (tab: TabType) => void;
}

const DISEASES = [
  { id: 'diabetes', label: 'Diabetes', icon: Activity, color: 'text-sky-600 bg-sky-50 border-sky-200' },
  { id: 'cardiovascular', label: 'Cardiovascular', icon: Heart, color: 'text-rose-600 bg-rose-50 border-rose-200' },
  { id: 'cancer', label: 'Breast Cancer', icon: Sparkles, color: 'text-pink-600 bg-pink-50 border-pink-200' },
];

export const DataPipelineView: React.FC<DataPipelineViewProps> = ({ setActiveTab }) => {
  const [selectedDisease, setSelectedDisease] = useState<string>('diabetes');
  const [pipeline, setPipeline] = useState<PipelineMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    api.getPipelineMetadata(selectedDisease)
      .then(data => {
        if (mounted) {
          setPipeline(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (mounted) {
          setError(err.message || 'Failed to load pipeline metadata');
          setLoading(false);
        }
      });
    return () => { mounted = false; };
  }, [selectedDisease]);

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <GitMerge className="text-orange-500" />
            Data Pipeline & Feature Engineering
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Authoritative, leakage-free biomedical preprocessing pipelines across Diabetes, Cardiovascular, and Cancer datasets.
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

      {loading && (
        <div className="flex justify-center items-center h-48">
          <Loader2 className="w-8 h-8 text-orange-500 animate-spin" />
        </div>
      )}

      {error && !loading && (
        <div className="bg-red-50 border border-red-200 rounded-[2rem] p-6 text-red-800 text-sm flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
          <div>
            <strong className="block mb-1">Pipeline Verification Error</strong>
            {error}
          </div>
        </div>
      )}

      {pipeline && !loading && (
        <>
          {/* Top Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">Dataset Status</span>
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${pipeline.status === 'Ready' ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                <span className="text-lg font-bold text-slate-900">{pipeline.status}</span>
              </div>
              <span className="text-xs text-slate-400 mt-1 block truncate">{pipeline.dataset_name || pipeline.dataset_version}</span>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">Total Samples</span>
              <span className="text-2xl font-bold text-slate-900">{pipeline.sample_count || 0}</span>
              <span className="text-xs text-slate-400 mt-1 block">80/20 Stratified Split</span>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">Feature Dimension</span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold text-slate-900">{pipeline.total_feature_count}</span>
                <span className="text-xs text-slate-500">({pipeline.raw_feature_count} raw + {pipeline.engineered_feature_count} eng)</span>
              </div>
              <span className="text-xs text-slate-400 mt-1 block">Target: {pipeline.target_variable}</span>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">Class Balance</span>
              <span className="text-sm font-bold text-slate-900 block truncate">
                {pipeline.imbalance_ratio ? pipeline.imbalance_ratio.split('(')[0].trim() : 'Balanced'}
              </span>
              <span className="text-xs text-slate-400 mt-1 block truncate">
                {pipeline.imbalance_ratio && pipeline.imbalance_ratio.includes('(')
                  ? pipeline.imbalance_ratio.split('(')[1].replace(')', '')
                  : 'Stratified distribution'}
              </span>
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Metadata & Class Distribution */}
            <div className="space-y-6">
              <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
                  <Database className="w-5 h-5 text-orange-500" />
                  Dataset Specifications
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Dataset Name</span>
                    <span className="font-semibold text-slate-900">{pipeline.dataset_name || pipeline.dataset_version}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Disease Domain</span>
                    <span className="font-semibold text-slate-900 capitalize">{pipeline.disease || selectedDisease}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Pipeline Version</span>
                    <span className="font-mono text-slate-900">{pipeline.pipeline_version}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Raw Predictor Features</span>
                    <span className="font-mono font-bold text-slate-900">{pipeline.raw_feature_count}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Engineered Features</span>
                    <span className="font-mono font-bold text-slate-900">{pipeline.engineered_feature_count}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Total Classical Features</span>
                    <span className="font-mono font-bold text-slate-900">{pipeline.total_feature_count}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Data Split Strategy</span>
                    <span className="font-mono text-slate-900">{pipeline.split_strategy}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100 text-sm">
                    <span className="text-slate-600">Target Label</span>
                    <span className="font-mono font-bold text-orange-600">{pipeline.target_variable}</span>
                  </div>
                </div>

                {/* Class Distribution Breakdown */}
                {pipeline.class_distribution && (
                  <div className="mt-6 pt-5 border-t border-slate-100">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                      <BarChart2 className="w-3.5 h-3.5 text-slate-400" />
                      Class Distribution (Ground Truth)
                    </h4>
                    <div className="flex gap-4">
                      {Object.entries(pipeline.class_distribution).map(([cls, cnt]) => {
                        const countNum = Number(cnt);
                        const total = pipeline.sample_count || 1;
                        const pct = ((countNum / total) * 100).toFixed(1);
                        const isPos = cls === '1';
                        return (
                          <div key={cls} className={`flex-1 p-3 rounded-xl border ${isPos ? 'bg-orange-50 border-orange-200' : 'bg-slate-50 border-slate-200'}`}>
                            <div className="text-xs font-semibold text-slate-500">
                              {isPos ? 'Positive / Malignant (Class 1)' : 'Negative / Benign (Class 0)'}
                            </div>
                            <div className="text-xl font-bold text-slate-900 mt-1">{cnt} <span className="text-xs font-normal text-slate-500">({pct}%)</span></div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Feature Dictionary */}
              {pipeline.feature_names && pipeline.feature_names.length > 0 && (
                <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs">
                  <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-3">
                    <Layers className="w-5 h-5 text-orange-500" />
                    Biomedical Feature Schema ({pipeline.feature_names.length} Features)
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    Strictly validated feature names loaded directly by the biomedical preprocessor.
                  </p>
                  <div className="flex flex-wrap gap-2 max-h-56 overflow-y-auto pr-1">
                    {pipeline.feature_names.map((fname, idx) => (
                      <span
                        key={idx}
                        className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 font-mono text-xs"
                      >
                        {fname}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right: Pipeline Stages Execution Flow */}
            <div className="bg-white rounded-[2rem] border border-orange-100 p-6 shadow-xs flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">Sequential Pipeline Stages</h3>
                <p className="text-xs text-slate-500 mb-6">
                  Preprocessors are fitted strictly on the 80% training partition to prevent test-set data leakage.
                </p>
                <div className="space-y-4">
                  {pipeline.stages.map((stage, idx) => (
                    <div key={idx} className="relative flex items-start gap-4 group">
                      {idx < pipeline.stages.length - 1 && (
                        <div className="absolute left-3.5 top-7 bottom-0 w-0.5 bg-slate-100 -mb-4" />
                      )}
                      <div className="w-7 h-7 rounded-full bg-emerald-50 border border-emerald-300 text-emerald-700 flex items-center justify-center shrink-0 mt-0.5 font-bold text-xs shadow-2xs">
                        {idx + 1}
                      </div>
                      <div className="flex-1 bg-slate-50/70 border border-slate-100 hover:border-orange-200 transition-colors rounded-xl p-3.5">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-900 font-bold text-sm">{stage.stage_name}</span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100/70 text-emerald-800">
                            Leakage-Free
                          </span>
                        </div>
                        <span className="text-slate-600 font-mono text-xs block mt-1">{stage.method}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-8 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>Reproducibility: Random Seed 42</span>
                <button
                  onClick={() => setActiveTab('hybrid-comparison')}
                  className="inline-flex items-center gap-1 font-bold text-orange-600 hover:text-orange-700 transition-colors"
                >
                  View Model Benchmarks <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

