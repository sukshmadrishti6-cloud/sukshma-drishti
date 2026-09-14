import React, { useState, useEffect } from 'react';
import { TabType, InferenceResponse, User } from './types';
import { Navigation } from './components/Navigation';
import { OverviewView } from './components/OverviewView';
import { DiseasesView } from './components/DiseasesView';
import { HistoryView } from './components/HistoryView';
import { RecommendationsView } from './components/RecommendationsView';
import { DataPipelineView } from './components/DataPipelineView';
import { ClassicalBaselinesView } from './components/ClassicalBaselinesView';
import { QuantumLabView } from './components/QuantumLabView';
import { HybridComparisonView } from './components/HybridComparisonView';
import { ExplainabilityView } from './components/ExplainabilityView';
import { InferenceView } from './components/InferenceView';
import { AuthModal } from './components/AuthModal';
import { api } from './api';
import { Play, Search, CircleCheck, User as UserIcon } from 'lucide-react';
import { FallingLeaves } from './components/FallingLeaves';

import { ErrorBoundary } from './components/ErrorBoundary';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('diseases');
  const [lastInferenceResult, setLastInferenceResult] = useState<InferenceResponse | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    const cachedUser = localStorage.getItem('user');
    if (cachedUser) {
      try {
        setCurrentUser(JSON.parse(cachedUser));
      } catch (e) {
        // ignore JSON parse error
      }
    }
    if (token) {
      api.getMe()
        .then((user) => {
          setCurrentUser(user);
          localStorage.setItem('user', JSON.stringify(user));
        })
        .catch(() => {
          localStorage.removeItem('access_token');
          localStorage.removeItem('user');
          setCurrentUser(null);
        });
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    setCurrentUser(null);
  };

  const getPageTitle = () => {
    switch (activeTab) {
      case 'diseases':
        return 'Disease Assessment Suite';
      case 'history':
        return 'Health History';
      case 'recommendations':
        return 'Personalized Guidance';
      case 'overview':
        return 'Platform Overview';
      case 'data-pipeline':
        return 'Data Pipeline';
      case 'classical-baselines':
        return 'Classical Baselines';
      case 'quantum-lab':
        return 'Quantum Lab';
      case 'hybrid-comparison':
        return 'Hybrid Comparison';
      case 'explainability':
        return 'Explainability';
      case 'inference':
        return 'Inference';
      default:
        return 'Overview';
    }
  };

  return (
    <div className="sd-app flex h-screen w-screen text-slate-800 font-sans overflow-hidden antialiased select-none">
      <FallingLeaves />

      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentUser={currentUser}
        onOpenAuthModal={() => setIsAuthModalOpen(true)}
        onLogout={handleLogout}
      />

      <div className="sd-shell flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="sd-header h-18 flex items-center justify-between px-8 shrink-0 z-20">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-xl font-bold tracking-tight flex items-center gap-2">
                <span>SukshmaDrishti</span>
                <span className="text-xs font-mono font-normal text-slate-400">/</span>
                <span className="text-sm font-semibold text-emerald-700 capitalize">
                  {getPageTitle()}
                </span>
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="sd-search hidden lg:flex items-center gap-2">
              <Search className="w-4 h-4" />
              <span>Search platform</span>
            </div>
            <div className="sd-system-status hidden md:flex items-center gap-2">
              <CircleCheck className="w-4 h-4" />
              <span>System Ready</span>
            </div>
            {currentUser ? (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-xl text-xs font-semibold text-slate-700">
                <UserIcon className="w-3.5 h-3.5 text-emerald-600" />
                <span>{currentUser.name}</span>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthModalOpen(true)}
                className="px-3.5 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-xl text-xs font-semibold border border-emerald-200/60 transition-all"
              >
                Sign In
              </button>
            )}
            <button
              onClick={() => setActiveTab('diseases')}
              className="sd-run-button px-4 py-2 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all"
            >
              <Play className="w-3.5 h-3.5" />
              <span>Start Assessment</span>
            </button>
          </div>
        </header>

        <main className="sd-main flex-1 flex flex-col overflow-hidden relative">
          {activeTab === 'diseases' && (
            <ErrorBoundary fallbackTitle="Error loading Disease Assessment Suite">
              <DiseasesView
                setActiveTab={setActiveTab}
                onOpenAuthModal={() => setIsAuthModalOpen(true)}
                currentUser={currentUser}
              />
            </ErrorBoundary>
          )}
          {activeTab === 'history' && (
            <ErrorBoundary fallbackTitle="Error loading Health History">
              <HistoryView
                setActiveTab={setActiveTab}
                currentUser={currentUser}
                onOpenAuthModal={() => setIsAuthModalOpen(true)}
              />
            </ErrorBoundary>
          )}
          {activeTab === 'recommendations' && (
            <ErrorBoundary fallbackTitle="Error loading Personalized Guidance">
              <RecommendationsView
                setActiveTab={setActiveTab}
                currentUser={currentUser}
                onOpenAuthModal={() => setIsAuthModalOpen(true)}
              />
            </ErrorBoundary>
          )}
          {activeTab === 'overview' && (
            <ErrorBoundary fallbackTitle="Error loading Platform Overview">
              <OverviewView setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'data-pipeline' && (
            <ErrorBoundary fallbackTitle="Error loading Data Pipeline">
              <DataPipelineView setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'classical-baselines' && (
            <ErrorBoundary fallbackTitle="Error loading Classical Baselines">
              <ClassicalBaselinesView setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'quantum-lab' && (
            <ErrorBoundary fallbackTitle="Error loading Quantum Lab">
              <QuantumLabView setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'hybrid-comparison' && (
            <ErrorBoundary fallbackTitle="Error loading Hybrid Comparison">
              <HybridComparisonView setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'explainability' && (
            <ErrorBoundary fallbackTitle="Error loading Explainability">
              <ExplainabilityView result={lastInferenceResult} setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
          {activeTab === 'inference' && (
            <ErrorBoundary fallbackTitle="Error loading Inference">
              <InferenceView onResult={setLastInferenceResult} setActiveTab={setActiveTab} />
            </ErrorBoundary>
          )}
        </main>
      </div>

      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onSuccess={(user) => setCurrentUser(user)}
      />
    </div>
  );
}


