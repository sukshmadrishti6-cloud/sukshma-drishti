import React, { useState, useEffect } from 'react';
import {
  History,
  Calendar,
  Eye,
  Trash2,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Filter,
  X,
  UserCheck,
} from 'lucide-react';
import { PredictionHistoryDetail, PredictionHistoryItem, TabType } from '../types';
import { api } from '../api';
import { PredictionResultView } from './PredictionResultView';

interface HistoryViewProps {
  setActiveTab: (tab: TabType) => void;
  currentUser: any;
  onOpenAuthModal: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  setActiveTab,
  currentUser,
  onOpenAuthModal,
}) => {
  const [items, setItems] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [diseaseFilter, setDiseaseFilter] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);
  const [detailRecord, setDetailRecord] = useState<PredictionHistoryDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchHistory = () => {
    if (!currentUser) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);

    api
      .getHistory(page, pageSize, diseaseFilter || undefined)
      .then((res) => {
        setItems(res.items || []);
        setTotal(res.total || 0);
      })
      .catch((err) => {
        setError(err.message || 'Failed to fetch prediction history records');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, [currentUser, page, diseaseFilter]);

  const handleOpenDetail = (id: string) => {
    setSelectedRecordId(id);
    setLoadingDetail(true);
    api
      .getHistoryDetail(id)
      .then((detail) => {
        setDetailRecord(detail);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load record details');
      })
      .finally(() => {
        setLoadingDetail(false);
      });
  };

  const handleDeleteRecord = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this prediction history record?')) {
      return;
    }
    try {
      await api.deleteHistory(id);
      if (selectedRecordId === id) {
        setSelectedRecordId(null);
        setDetailRecord(null);
      }
      fetchHistory();
    } catch (err: any) {
      alert(err.message || 'Failed to delete history record.');
    }
  };

  if (!currentUser) {
    return (
      <div className="flex-1 overflow-y-auto p-8 flex flex-col items-center justify-center text-center space-y-4 select-none">
        <div className="w-16 h-16 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center">
          <UserCheck className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900">Authentication Required</h2>
        <p className="text-sm text-slate-600 max-w-md">
          Sign in to view your secure patient assessment history, review previous risk scores, and track health recommendations.
        </p>
        <button
          onClick={onOpenAuthModal}
          className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-semibold shadow-sm transition-all"
        >
          Sign In / Register
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 select-none">
      {/* Header Banner */}
      <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <History className="w-6 h-6 text-emerald-600" />
            <span>Health History & Assessment Records</span>
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            Review your persistent disease risk assessments, model predictions, and associated health recommendations.
          </p>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={diseaseFilter}
            onChange={(e) => {
              setDiseaseFilter(e.target.value);
              setPage(1);
            }}
            className="text-xs border border-slate-200 rounded-xl p-2 bg-slate-50 font-medium"
          >
            <option value="">All Diseases</option>
            <option value="diabetes">Diabetes</option>
            <option value="cardiovascular">Cardiovascular</option>
            <option value="cancer">Cancer</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-[2rem] border border-slate-200 p-6 shadow-xs space-y-4">
        {loading ? (
          <div className="py-12 flex justify-center items-center text-slate-500 text-xs font-medium">
            <RefreshCw className="w-5 h-5 animate-spin text-emerald-600 mr-2" /> Loading health records...
          </div>
        ) : error ? (
          <div className="p-4 bg-red-50 text-red-700 text-xs rounded-xl border border-red-200">
            {error}
          </div>
        ) : items.length === 0 ? (
          <div className="py-12 text-center text-slate-400 text-xs space-y-2">
            <p>No prediction history records found.</p>
            <button
              onClick={() => setActiveTab('diseases')}
              className="text-xs text-emerald-600 font-semibold hover:underline"
            >
              Run a new disease assessment
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                  <th className="pb-3 px-3">Disease</th>
                  <th className="pb-3 px-3">Prediction</th>
                  <th className="pb-3 px-3">Risk Score</th>
                  <th className="pb-3 px-3">Model</th>
                  <th className="pb-3 px-3">Date</th>
                  <th className="pb-3 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {items.map((r) => {
                  const isHigh = r.prediction === 1;
                  return (
                    <tr
                      key={r.id}
                      onClick={() => handleOpenDetail(r.id)}
                      className="hover:bg-slate-50/80 cursor-pointer transition-colors"
                    >
                      <td className="py-3.5 px-3">
                        <span className="capitalize font-bold text-slate-900">{r.disease}</span>
                      </td>
                      <td className="py-3.5 px-3">
                        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                          isHigh ? 'bg-rose-50 text-rose-700 border border-rose-200/60' : 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                        }`}>
                          {isHigh ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                          {r.risk_category || (isHigh ? 'High Risk' : 'Low Risk')}
                        </span>
                      </td>
                      <td className="py-3.5 px-3 font-mono">
                        {r.score_semantics === 'uncalibrated_quantum_decision_score'
                          ? `${r.risk_score?.toFixed(4)} (Margin)`
                          : r.risk_score !== null && r.risk_score !== undefined
                          ? `${(r.risk_score * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-3.5 px-3 text-slate-600">
                        {r.model_name} <span className="text-[10px] text-slate-400 font-mono">({r.model_version})</span>
                      </td>
                      <td className="py-3.5 px-3 text-slate-500 font-mono text-[11px]">
                        {new Date(r.created_at).toLocaleDateString()}{' '}
                        {new Date(r.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="py-3.5 px-3 text-right space-x-1">
                        <button
                          onClick={() => handleOpenDetail(r.id)}
                          className="p-1.5 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="View Assessment Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteRecord(r.id, e)}
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete Record"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {total > pageSize && (
          <div className="flex items-center justify-between pt-4 border-t border-slate-100 text-xs text-slate-500">
            <span>
              Showing {((page - 1) * pageSize) + 1} to {Math.min(page * pageSize, total)} of {total} records
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="font-semibold text-slate-700">{page}</span>
              <button
                onClick={() => setPage((p) => (p * pageSize < total ? p + 1 : p))}
                disabled={page * pageSize >= total}
                className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Record Detail Modal */}
      {selectedRecordId && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-[2.5rem] max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 md:p-8 space-y-6 shadow-2xl relative">
            <button
              onClick={() => {
                setSelectedRecordId(null);
                setDetailRecord(null);
              }}
              className="absolute right-6 top-6 p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-full transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <History className="w-5 h-5 text-emerald-600" />
              <span>Historical Assessment Record Detail</span>
            </h2>

            {loadingDetail ? (
              <div className="py-12 flex justify-center items-center text-slate-500 text-xs font-medium">
                <RefreshCw className="w-5 h-5 animate-spin text-emerald-600 mr-2" /> Loading record details...
              </div>
            ) : detailRecord ? (
              <div className="space-y-6">
                {/* Input Payload Grid */}
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-2xl space-y-2 text-xs">
                  <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block">
                    Original Biomedical Inputs ({Object.keys(detailRecord.input_data || {}).length} features)
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
                    {Object.entries(detailRecord.input_data || {}).map(([k, v]) => (
                      <div key={k} className="p-2 bg-white rounded-lg border border-slate-200 text-slate-800 truncate" title={`${k}: ${v}`}>
                        <span className="text-[10px] text-slate-400 block truncate">{k}</span>
                        <span className="font-bold">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Structured Prediction + Recommendation View */}
                <PredictionResultView
                  result={{
                    inference_id: detailRecord.id,
                    disease: detailRecord.disease,
                    predicted_class: detailRecord.prediction,
                    probability: detailRecord.score_semantics !== 'uncalibrated_quantum_decision_score' ? detailRecord.risk_score : undefined,
                    decision_score: detailRecord.score_semantics === 'uncalibrated_quantum_decision_score' ? detailRecord.risk_score : undefined,
                    score_semantics: detailRecord.score_semantics || undefined,
                    model_output: detailRecord.risk_category || (detailRecord.prediction === 1 ? 'High Risk' : 'Low Risk'),
                    model_used: detailRecord.model_name,
                    model_family: detailRecord.model_name.includes('qsvc') || detailRecord.model_name.includes('vqc') ? 'quantum' : 'classical',
                    artifact_id: detailRecord.model_name,
                    artifact_version: detailRecord.model_version,
                    explanation: detailRecord.explanation as any,
                    recommendation: detailRecord.recommendation as any,
                    audit: {
                      timestamp: detailRecord.created_at,
                      inference_time_ms: 0,
                    },
                  }}
                  setActiveTab={setActiveTab}
                />
              </div>
            ) : (
              <p className="text-xs text-slate-500">Record not found.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
