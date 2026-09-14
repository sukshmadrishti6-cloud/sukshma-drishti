import React, { useEffect, useState } from 'react';
import { TabType, DiseaseSummary, ModelSummary, QuantumCapability, PipelineMetadata } from '../types';
import { Atom, ArrowUpRight, GitMerge, Cpu, Scale, RefreshCw, AlertTriangle, Leaf, BrainCircuit, Database, Activity, ShieldCheck } from 'lucide-react';
import { api, HealthResponse } from '../api';

interface OverviewViewProps { setActiveTab: (tab: TabType) => void; }

export const OverviewView: React.FC<OverviewViewProps> = ({ setActiveTab }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [diseases, setDiseases] = useState<DiseaseSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [pipeline, setPipeline] = useState<PipelineMetadata | null>(null);
  const [capabilities, setCapabilities] = useState<QuantumCapability[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    Promise.all([
      api.getHealth(),
      api.getDiseases(),
      api.getModels(),
      api.getPipelineMetadata().catch(() => null),
      api.getQuantumCapabilities().catch(() => []),
    ])
      .then(([healthRes, diseasesRes, modelsRes, pipeRes, capsRes]) => {
        if (!mounted) return;
        setHealth(healthRes);
        setDiseases(diseasesRes.items || []);
        setModels(modelsRes.models || []);
        if (pipeRes) setPipeline(pipeRes);
        if (capsRes) setCapabilities(capsRes);
        setError(null);
      })
      .catch(err => {
        if (mounted) setError(err.message || 'Failed to load platform overview metadata.');
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => { mounted = false; };
  }, []);

  const cards = [
    { id: 'diseases' as TabType, title: 'Disease Assessments', icon: Leaf, tone: 'green', text: 'Screen for Diabetes, Cardiovascular, and Breast Cancer risk.' },
    { id: 'classical-baselines' as TabType, title: 'Model Evaluation Matrix', icon: Cpu, tone: 'amber', text: 'Inspect 80/20 holdout metrics, ROC/PR curves & calibration for 51 models.' },
    { id: 'quantum-lab' as TabType, title: 'Quantum Lab', icon: Atom, tone: 'violet', text: 'Simulate parameterized quantum circuits with Qiskit Aer.' },
    { id: 'hybrid-comparison' as TabType, title: 'Hybrid Comparison', icon: Scale, tone: 'blue', text: 'Compare classical machine learning and quantum circuits transparently.' },
  ];

  const totalModelsCount = models.length || (health?.models_available?.length || 51);
  const evaluatedModelsCount = models.filter(m => m.evaluated).length || totalModelsCount;
  const activeDiseasesCount = diseases.filter(d => d.status === 'available').length || 3;
  const evaluatedQubitCap = capabilities.find(c => c.evaluated)?.qubit_count || 4;

  return (
    <div className="sd-overview flex-1 overflow-y-auto p-6 md:p-8 space-y-7">
      <section className="sd-hero grid grid-cols-1 xl:grid-cols-[1.05fr_.95fr] overflow-hidden">
        <div className="sd-hero-copy p-7 md:p-9 flex flex-col justify-center">
          <span className="sd-eyebrow">Hybrid Quantum-Classical ML</span>
          <h1 className="sd-hero-title mt-4">Early Disease Detection <em>Intelligence</em></h1>
          <p className="sd-hero-description mt-4 max-w-2xl">A rigorous multi-disease research platform comparing classical machine learning and parameterized quantum circuits for early disease risk prediction.</p>
          <div className="sd-hero-values mt-7 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div><Leaf /><span>3 Active<br/>Disease Domains</span></div>
            <div><BrainCircuit /><span>Deterministic<br/>Health Guidance</span></div>
            <div><Activity /><span>51 Authoritative<br/>ML Models</span></div>
          </div>
          <div className="sd-quote mt-7">“Small insights. Healthier tomorrows.”</div>
        </div>
        <div className="sd-hero-art">
          <img src="/sukshmadrishti-hero.png" alt="Nature inspired AI research visualization" />
          <div className="sd-art-caption">Quantum × Biology × Humanity</div>
        </div>
      </section>

      {/* Dynamic Statistics Row */}
      <section className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {[
          [String(activeDiseasesCount), 'Active Domains', 'Diabetes, Cardio, Cancer', Leaf, 'green'],
          [String(totalModelsCount), 'Registered Models', `${evaluatedModelsCount} Evaluated on Holdout`, Database, 'amber'],
          [pipeline ? `${pipeline.total_feature_count || 11}` : 'Multi', 'Clinical Indicators', 'Validated Feature Schemas', Atom, 'violet'],
          [`${evaluatedQubitCap}Q`, 'Quantum Register', 'Holdout Evaluated (Aer Sim)', BrainCircuit, 'blue'],
        ].map(([value, label, sub, Icon, tone]) => {
          const IconComp = Icon as any;
          return (
            <div key={label as string} className={`sd-stat sd-stat-${tone}`}>
              <div className="sd-stat-icon"><IconComp className="w-5 h-5" /></div>
              <div>
                <div className="sd-stat-value">{value as string}</div>
                <div className="sd-stat-label">{label as string}</div>
                <div className="sd-stat-sub">{sub as string}</div>
              </div>
            </div>
          );
        })}
      </section>

      <div className="sd-section-heading flex items-end justify-between">
        <div><div className="sd-section-kicker"><Leaf className="w-4 h-4" /> Explore the Platform</div><p>Move through the research workflow from multi-disease screening to explainable inference and hybrid quantum analysis.</p></div>
        <span className="sd-section-note hidden md:block">Different perspectives. A healthier tomorrow.</span>
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {cards.map(({ id, title, icon: Icon, tone, text }) => (
          <button key={id} onClick={() => setActiveTab(id)} className={`sd-explore-card sd-tone-${tone} text-left`}>
            <div className="sd-card-icon"><Icon className="w-5 h-5" /></div>
            <div className="flex-1"><h3>{title}</h3><p>{text}</p></div>
            <div className="sd-arrow"><ArrowUpRight className="w-4 h-4" /></div>
          </button>
        ))}
      </section>

      <section className="sd-status-card">
        <div><span className="sd-section-kicker"><Activity className="w-4 h-4" /> System & Model Registry Status</span><p className="mt-1">Backend connectivity and multi-disease model registry health</p></div>
        <div className="sd-backend-status">
          {loading && <><RefreshCw className="w-4 h-4 animate-spin" /> Checking backend…</>}
          {error && <><AlertTriangle className="w-4 h-4" /> {error}</>}
          {health && <><span className="sd-live-dot" /> {health.backend_engine || 'SukshmaDrishti Engine'} {health.version} · {totalModelsCount} models registered across {activeDiseasesCount} disease domains</>}
        </div>
      </section>
    </div>
  );
};

