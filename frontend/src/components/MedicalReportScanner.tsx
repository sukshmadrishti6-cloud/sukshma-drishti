import React, { useState, useRef } from 'react';
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Edit3,
  ArrowRight,
  ShieldCheck,
  Sparkles,
  Info,
  X,
  FileSearch,
  Check,
  HelpCircle,
} from 'lucide-react';
import { ReportScanResponse, ExtractedFeatureItem } from '../types';
import { api } from '../api';

interface MedicalReportScannerProps {
  diseaseId: string;
  diseaseName: string;
  onVerifiedFeatures: (features: Record<string, number>) => void;
  onSwitchToManual: () => void;
}

const ACCEPTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.webp'];
const MAX_SIZE_MB = 10;

export const MedicalReportScanner: React.FC<MedicalReportScannerProps> = ({
  diseaseId,
  diseaseName,
  onVerifiedFeatures,
  onSwitchToManual,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<ReportScanResponse | null>(null);

  // Form state for user verification and completing missing features
  const [verifiedValues, setVerifiedValues] = useState<Record<string, number>>({});

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateSelectedFile = (selectedFile: File): boolean => {
    const ext = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type (${ext}). Please upload a PDF, PNG, JPG, JPEG, or WEBP report.`);
      return false;
    }

    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File is too large (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is ${MAX_SIZE_MB} MB.`);
      return false;
    }

    setError(null);
    return true;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (validateSelectedFile(selected)) {
        setFile(selected);
        setScanResult(null);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      if (validateSelectedFile(dropped)) {
        setFile(dropped);
        setScanResult(null);
      }
    }
  };

  const handleUploadAndScan = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setLoadingStep('Uploading encrypted document to processing pipeline...');

    try {
      setTimeout(() => setLoadingStep('Extracting clinical text and table structures...'), 600);
      setTimeout(() => setLoadingStep('Matching medical aliases and normalizing units...'), 1200);

      const result = await api.scanMedicalReport(file, diseaseId);
      setScanResult(result);

      // Populate verified values state with extracted values only (do NOT pre-populate missing features with 0)
      const initialValues: Record<string, number> = {};
      Object.entries(result.extracted_features).forEach(([key, item]) => {
        const featItem = item as ExtractedFeatureItem;
        initialValues[key] = featItem.value;
      });
      setVerifiedValues(initialValues);
    } catch (err: any) {
      setError(err.message || 'Failed to scan and parse the medical document. Please try again or use manual entry.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleValueChange = (featureName: string, val: string) => {
    if (val === '') {
      setVerifiedValues((prev) => {
        const next = { ...prev };
        delete next[featureName];
        return next;
      });
    } else {
      const num = Number(val);
      setVerifiedValues((prev) => ({ ...prev, [featureName]: num }));
    }
  };

  const handleProceedToPrediction = () => {
    if (!scanResult) return;

    // Check if any missing feature was left empty
    const uncompleted = scanResult.missing_features.filter(
      (key) => verifiedValues[key] === undefined || isNaN(verifiedValues[key])
    );

    if (uncompleted.length > 0) {
      setError(`Please provide values for the missing parameters (${uncompleted.join(', ')}) before running risk assessment.`);
      return;
    }

    setError(null);
    onVerifiedFeatures(verifiedValues);
  };

  const handleReset = () => {
    setFile(null);
    setScanResult(null);
    setError(null);
    setVerifiedValues({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };


  const renderConfidenceBadge = (tier: 'high' | 'medium' | 'low', score: number) => {
    switch (tier) {
      case 'high':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
            <Check className="w-3 h-3 text-emerald-600" /> High ({Math.round(score * 100)}%)
          </span>
        );
      case 'medium':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
            <AlertTriangle className="w-3 h-3 text-amber-600" /> Verify ({Math.round(score * 100)}%)
          </span>
        );
      case 'low':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-rose-700 bg-rose-50 px-2 py-0.5 rounded-md border border-rose-200">
            <AlertTriangle className="w-3 h-3 text-rose-600" /> Low ({Math.round(score * 100)}%)
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Box (when no scanResult yet) */}
      {!scanResult && (
        <div className="bg-white rounded-[2rem] border border-slate-200 p-6 md:p-8 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
                  Primary & Recommended
                </span>
                <span className="text-xs text-slate-500 font-medium">AI Document Scanner</span>
              </div>
              <h2 className="text-xl font-extrabold text-slate-900 tracking-tight mt-1">
                Scan Medical Report for {diseaseName}
              </h2>
              <p className="text-xs text-slate-600 mt-0.5">
                Upload your medical report (PDF or photo). Our clinical parser automatically extracts and normalizes required parameters.
              </p>
            </div>

            <button
              onClick={onSwitchToManual}
              className="text-xs font-semibold text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-3.5 py-2 rounded-xl transition-all shrink-0"
            >
              Enter Manually Instead
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-xs flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 shrink-0" />
              <div className="flex-1">
                <p className="font-semibold">Document Processing Error</p>
                <p className="mt-0.5 text-red-600">{error}</p>
              </div>
            </div>
          )}

          {/* Drag & Drop Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-[1.75rem] p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-3 ${
              isDragging
                ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]'
                : file
                ? 'border-emerald-300 bg-emerald-50/20'
                : 'border-slate-200 bg-slate-50/50 hover:bg-slate-50 hover:border-slate-300'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.webp"
              onChange={handleFileChange}
              className="hidden"
            />

            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${
              file ? 'bg-emerald-100 text-emerald-700' : 'bg-white text-slate-600 shadow-xs border border-slate-200'
            }`}>
              {file ? <FileText className="w-7 h-7" /> : <Upload className="w-7 h-7" />}
            </div>

            {file ? (
              <div className="space-y-1">
                <p className="text-sm font-bold text-slate-900">{file.name}</p>
                <p className="text-xs text-slate-500 font-mono">
                  {(file.size / 1024).toFixed(1)} KB • Click to change document
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-sm font-bold text-slate-800">
                  Drag and drop your medical report here, or <span className="text-emerald-700 underline underline-offset-2">browse files</span>
                </p>
                <p className="text-xs text-slate-500">
                  Supports PDF documents and clear images (JPG, PNG, WEBP) up to 10 MB
                </p>
              </div>
            )}
          </div>

          {/* Upload and Process Button */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <div className="text-[11px] text-slate-500 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>In-memory processing with automatic file sanitization & immediate disposal</span>
            </div>

            <button
              onClick={handleUploadAndScan}
              disabled={!file || loading}
              className="w-full sm:w-auto px-6 py-3 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-xs disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>{loadingStep || 'Scanning & Extracting Parameters...'}</span>
                </>
              ) : (
                <>
                  <FileSearch className="w-4 h-4" />
                  <span>Scan & Extract Parameters</span>
                </>
              )}
            </button>
          </div>

          {/* Privacy Notice Banner */}
          <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 text-[11px] text-slate-600 space-y-1">
            <div className="flex items-center gap-1.5 font-bold text-slate-800">
              <Info className="w-3.5 h-3.5 text-slate-500" />
              <span>Privacy & Document Handling Notice</span>
            </div>
            <p className="leading-relaxed">
              Your uploaded document is processed solely in transient memory to detect relevant clinical measurements.
              We do not permanently archive your medical records, and extracted data is verified by you prior to model evaluation.
            </p>
          </div>
        </div>
      )}

      {/* Verification Screen (when scanResult is received) */}
      {scanResult && (
        <div className="bg-white rounded-[2rem] border border-slate-200 p-6 md:p-8 shadow-xs space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full border ${
                  scanResult.compatible
                    ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
                    : 'text-amber-800 bg-amber-50 border-amber-200'
                }`}>
                  {scanResult.report_category.replace(/_/g, ' ').toUpperCase()}
                </span>
                {scanResult.extraction_method && (
                  <span className="text-xs font-medium text-slate-600 bg-slate-100 px-2 py-0.5 rounded-md border border-slate-200">
                    {scanResult.extraction_method === 'pdf_text' ? 'Direct PDF Text' : scanResult.extraction_method === 'pdf_ocr' ? 'Scanned PDF OCR' : 'Image OCR'}
                    {scanResult.extracted_text_length ? ` (${scanResult.extracted_text_length} chars)` : ''}
                  </span>
                )}
                <span className="text-xs text-slate-500 font-mono">{scanResult.filename}</span>
              </div>
              <h2 className="text-xl font-extrabold text-slate-900 tracking-tight mt-1">
                Report Extraction Verification
              </h2>
              <p className="text-xs text-slate-600 mt-0.5">{scanResult.message}</p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <span className={`px-3 py-1 rounded-xl text-xs font-bold border ${
                scanResult.total_features_detected === scanResult.total_features_required
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-amber-50 text-amber-800 border-amber-200'
              }`}>
                Detected: {scanResult.total_features_detected} / {scanResult.total_features_required} Features
              </span>
              <button
                onClick={handleReset}
                className="p-2 hover:bg-slate-100 text-slate-500 rounded-xl text-xs"
                title="Scan another document"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-xs flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 shrink-0" />
              <div className="flex-1">
                <p className="font-semibold">Verification Incomplete</p>
                <p className="mt-0.5 text-red-600">{error}</p>
              </div>
            </div>
          )}

          {/* Incompatible Report Notice (e.g. general blood report uploaded for Cancer) */}
          {!scanResult.compatible && (
            <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl text-amber-900 text-xs space-y-2">
              <div className="flex items-center gap-2 font-bold text-amber-950">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span>Pathology Report Required for Breast Cancer Assessment</span>
              </div>
              <p className="leading-relaxed">
                The Breast Cancer prediction pipeline evaluates <strong>30 microscopic cell nucleus morphometric features</strong> (such as mean radius, texture, compactness, and concavity) derived from digitized FNA biopsy slides.
                Generic laboratory blood panels do not contain these measurements.
              </p>
              <div className="pt-2 flex gap-3">
                <button
                  onClick={onSwitchToManual}
                  className="px-3.5 py-1.5 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-semibold text-xs transition-colors"
                >
                  Enter Morphology Values Manually
                </button>
                <button
                  onClick={handleReset}
                  className="px-3.5 py-1.5 bg-white border border-amber-300 text-amber-900 hover:bg-amber-100/50 rounded-lg font-semibold text-xs transition-colors"
                >
                  Upload Cytology Report
                </button>
              </div>
            </div>
          )}

          {/* Extracted Parameters Table */}
          {scanResult.total_features_detected > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Detected Parameters (Review & Edit if Needed)</span>
                </h3>
                <span className="text-[11px] text-slate-500">Click any value to adjust</span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-2xl">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase text-[10px] tracking-wider">
                      <th className="p-3">Model Parameter</th>
                      <th className="p-3">Original Report Label</th>
                      <th className="p-3">Extracted & Verified Value</th>
                      <th className="p-3">Unit</th>
                      <th className="p-3">Confidence</th>
                      <th className="p-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {Object.entries(scanResult.extracted_features).map(([key, rawItem]) => {
                      const item = rawItem as ExtractedFeatureItem;
                      return (
                        <tr key={key} className="hover:bg-slate-50/50 transition-colors">
                          <td className="p-3 font-semibold text-slate-900">{key}</td>
                          <td className="p-3 text-slate-600 font-mono text-[11px]">{item.original_name}</td>
                          <td className="p-3">
                            <div className="flex items-center gap-1.5">
                              <input
                                type="number"
                                step="any"
                                value={verifiedValues[key] !== undefined ? verifiedValues[key] : item.value}
                                onChange={(e) => handleValueChange(key, e.target.value)}
                                className="w-24 px-2 py-1 bg-slate-50 border border-slate-200 rounded-lg font-mono font-bold text-slate-900 text-xs focus:bg-white focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                              />
                              {item.conversion_applied && (
                                <span
                                  className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-200"
                                  title={`Original: ${item.original_value} ${item.original_unit || ''}`}
                                >
                                  Converted
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="p-3 text-slate-500 font-mono text-[11px]">{item.unit || '—'}</td>
                          <td className="p-3">{renderConfidenceBadge(item.confidence_level, item.confidence)}</td>
                          <td className="p-3">
                            <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 font-medium">
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" /> Extracted
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Missing Parameters Section */}
          {scanResult.missing_features_detail && scanResult.missing_features_detail.length > 0 && (
            <div className="bg-amber-50/50 border border-amber-200/80 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-amber-950 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-600" />
                    <span>Missing Parameters Required for Assessment ({scanResult.missing_features.length})</span>
                  </h3>
                  <p className="text-xs text-amber-800 mt-0.5">
                    The document did not contain these measurements. Please enter them manually to complete the evaluation.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                {scanResult.missing_features_detail.map((m) => (
                  <div key={m.name} className="p-3 bg-white border border-amber-200 rounded-xl space-y-1.5">
                    <div className="flex items-center justify-between gap-1">
                      <label className="block font-bold text-slate-800 truncate" title={m.name}>
                        {m.display_name}
                      </label>
                      <span className="text-[9px] font-bold text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded">
                        Missing
                      </span>
                    </div>
                    <input
                      type="number"
                      step="any"
                      value={verifiedValues[m.name] !== undefined ? verifiedValues[m.name] : ''}
                      onChange={(e) => handleValueChange(m.name, e.target.value)}
                      placeholder="Enter value (Required)"
                      className="w-full px-2.5 py-1.5 bg-slate-50 border border-slate-200 rounded-lg font-mono text-slate-900 text-xs focus:bg-white focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                    />
                    {m.description && (
                      <span className="text-[10px] text-slate-500 block truncate" title={m.description}>
                        {m.description}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}


          {/* Action Bar */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-100">
            <div className="flex gap-2">
              <button
                onClick={handleReset}
                className="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                Scan Another Report
              </button>
              <button
                onClick={onSwitchToManual}
                className="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                Switch to Full Manual Form
              </button>
            </div>

            <button
              onClick={handleProceedToPrediction}
              className="w-full sm:w-auto px-6 py-3 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-xs"
            >
              <span>Continue to Model Risk Assessment</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Medical Disclaimer */}
          <div className="p-3 bg-slate-50 border border-slate-200/60 rounded-xl text-[11px] text-slate-500 italic">
            Disclaimer: This assessment is generated by scientific predictive models and is intended for informational decision support. Always consult a licensed healthcare professional for clinical diagnosis.
          </div>
        </div>
      )}
    </div>
  );
};
