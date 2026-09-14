import React, { useState, useEffect } from 'react';
import { Activity, Heart, Stethoscope, ArrowRight, ShieldAlert, Sparkles, CheckCircle2, Clock } from 'lucide-react';
import { DiseaseSummary, TabType } from '../types';
import { api } from '../api';
import { DiseaseAssessmentForm } from './DiseaseAssessmentForm';

interface DiseasesViewProps {
  setActiveTab: (tab: TabType) => void;
  onOpenAuthModal: () => void;
  currentUser: any;
}

export const DiseasesView: React.FC<DiseasesViewProps> = ({
  setActiveTab,
  onOpenAuthModal,
  currentUser,
}) => {
  const [diseases, setDiseases] = useState<DiseaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDiseaseId, setSelectedDiseaseId] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    api
      .getDiseases()
      .then((res) => {
        if (mounted && res && res.items) {
          setDiseases(res.items);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message || 'Failed to fetch registered disease domains');
        }
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const getDiseaseIcon = (id: string) => {
    switch (id) {
      case 'diabetes':
        return Activity;
      case 'cardiovascular':
        return Heart;
      case 'cancer':
        return Stethoscope;
      default:
        return Sparkles;
    }
  };

  const getDiseaseGradient = (id: string) => {
    switch (id) {
      case 'diabetes':
        return 'from-emerald-500/10 to-teal-500/10 border-emerald-200 text-emerald-700';
      case 'cardiovascular':
        return 'from-rose-500/10 to-pink-500/10 border-rose-200 text-rose-700';
      case 'cancer':
        return 'from-amber-500/10 to-orange-500/10 border-amber-200 text-amber-700';
      default:
        return 'from-blue-500/10 to-indigo-500/10 border-blue-200 text-blue-700';
    }
  };

  if (selectedDiseaseId) {
    return (
      <DiseaseAssessmentForm
        diseaseId={selectedDiseaseId}
        onBack={() => setSelectedDiseaseId(null)}
        setActiveTab={setActiveTab}
        onOpenAuthModal={onOpenAuthModal}
        currentUser={currentUser}
      />
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 select-none">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-emerald-900 via-teal-900 to-slate-900 rounded-[2rem] p-8 text-white shadow-lg relative overflow-hidden">
        <div className="absolute right-0 top-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/20 rounded-full text-emerald-300 text-xs font-semibold border border-emerald-400/30">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Multi-Disease Risk Assessment Directory</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight">Disease Assessment Suite</h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            Select a specialized AI risk assessment pipeline. Our platform evaluates physiological and morphometric features using calibrated classical ensemble and quantum machine learning models.
          </p>
        </div>
      </div>

      {/* Medical Safety Banner */}
      <div className="bg-amber-50/80 border border-amber-200/80 rounded-2xl p-4 flex items-start gap-3 text-amber-900 text-xs shadow-xs">
        <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <p className="font-bold text-amber-950">Important Clinical Disclaimer</p>
          <p className="mt-0.5 text-amber-800">
            SukshmaDrishti assessment tools provide statistical risk screening and informational guidance based on scientific model outputs. They do <strong>NOT</strong> constitute a definitive medical diagnosis, emergency triage, or clinical treatment prescription. Always consult a qualified healthcare professional regarding health concerns.
          </p>
        </div>
      </div>

      {/* Disease Cards Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-64 bg-slate-100 rounded-[2rem] animate-pulse" />
          ))}
        </div>
      ) : error ? (
        <div className="p-6 bg-red-50 border border-red-200 text-red-700 rounded-2xl text-sm">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {diseases.map((d) => {
            const Icon = getDiseaseIcon(d.id);
            const style = getDiseaseGradient(d.id);
            const isAvailable = d.status === 'available';

            return (
              <div
                key={d.id}
                className={`bg-white rounded-[2rem] border p-6 shadow-xs flex flex-col justify-between transition-all duration-300 hover:shadow-md ${
                  isAvailable ? 'hover:-translate-y-1' : 'opacity-75'
                }`}
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${style} border flex items-center justify-center`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    {isAvailable ? (
                      <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200/60">
                        <CheckCircle2 className="w-3 h-3" /> Ready
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-full border border-slate-200">
                        <Clock className="w-3 h-3" /> Coming Soon
                      </span>
                    )}
                  </div>

                  <div>
                    <h3 className="text-xl font-bold text-slate-900 tracking-tight">{d.name}</h3>
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mt-0.5">
                      Category: {d.category}
                    </span>
                  </div>

                  <p className="text-slate-600 text-xs leading-relaxed">{d.description}</p>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
                  <span className="text-xs font-mono text-slate-400">{d.version}</span>
                  {isAvailable ? (
                    <button
                      onClick={() => setSelectedDiseaseId(d.id)}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white text-xs font-semibold rounded-xl flex items-center gap-1.5 transition-all shadow-xs"
                    >
                      <span>Start Assessment</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  ) : (
                    <button
                      disabled
                      className="px-4 py-2 bg-slate-100 text-slate-400 text-xs font-semibold rounded-xl cursor-not-allowed"
                    >
                      Unavailable
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
