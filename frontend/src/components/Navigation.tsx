import React from 'react';
import { TabType, User } from '../types';
import {
  LayoutDashboard,
  GitMerge,
  Cpu,
  Atom,
  Scale,
  Eye,
  Play,
  Leaf,
  LogOut,
  LogIn,
  ShieldCheck,
  Stethoscope,
  History,
  HeartPulse,
} from 'lucide-react';

interface NavigationProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  currentUser: User | null;
  onOpenAuthModal: () => void;
  onLogout: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  currentUser,
  onOpenAuthModal,
  onLogout,
}) => {
  const clinicalItems: { id: TabType; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: 'diseases', label: 'Disease Assessments', icon: Stethoscope },
    { id: 'history', label: 'Health History', icon: History },
    { id: 'recommendations', label: 'Health Guidance', icon: HeartPulse },
  ];

  const researchItems: { id: TabType; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'data-pipeline', label: 'Data Pipeline', icon: GitMerge },
    { id: 'classical-baselines', label: 'Classical Baselines', icon: Cpu },
    { id: 'quantum-lab', label: 'Quantum Lab', icon: Atom },
    { id: 'hybrid-comparison', label: 'Hybrid Comparison', icon: Scale },
    { id: 'explainability', label: 'Explainability', icon: Eye },
    { id: 'inference', label: 'Quick Inference', icon: Play },
  ];

  return (
    <aside className="sd-sidebar w-64 flex flex-col justify-between p-4 select-none shrink-0 overflow-y-auto">
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-3 px-2 pt-2">
          <div className="sd-logo w-11 h-11 rounded-2xl flex items-center justify-center relative">
            <Leaf className="w-6 h-6" />
          </div>
          <div>
            <div className="sd-brand-name text-base font-bold tracking-tight">SukshmaDrishti</div>
            <p className="sd-brand-subtitle text-[11px] font-medium">Medical AI Platform</p>
          </div>
        </div>

        {/* Clinical Section */}
        <nav className="flex flex-col gap-1">
          <p className="sd-menu-label text-[10px] font-semibold uppercase tracking-widest px-3 pb-1 text-emerald-800">
            Clinical Screening
          </p>
          {clinicalItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`sd-nav-item w-full flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold ${
                  isActive ? 'sd-nav-active' : ''
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4" />
                  <span className="truncate">{item.label}</span>
                </div>
              </button>
            );
          })}
        </nav>

        {/* Research Section */}
        <nav className="flex flex-col gap-1">
          <p className="sd-menu-label text-[10px] font-semibold uppercase tracking-widest px-3 pb-1 text-slate-500">
            Scientific Research
          </p>
          {researchItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`sd-nav-item w-full flex items-center px-3.5 py-2 rounded-xl text-xs font-medium ${
                  isActive ? 'sd-nav-active' : ''
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4" />
                  <span className="truncate">{item.label}</span>
                </div>
              </button>
            );
          })}
        </nav>
      </div>


      <div className="sd-sidebar-footer mx-1 mb-2 p-3.5 rounded-2xl flex flex-col gap-3">
        {currentUser ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 min-w-0">
                <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-xs shrink-0">
                  {currentUser.name ? currentUser.name.charAt(0).toUpperCase() : 'U'}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-bold truncate text-slate-800">{currentUser.name}</p>
                  <p className="text-[10px] text-slate-500 truncate">{currentUser.email}</p>
                </div>
              </div>
              <button
                onClick={onLogout}
                title="Sign Out"
                className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
            <div className="flex items-center gap-1.5 text-[10px] font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded-md w-fit">
              <ShieldCheck className="w-3 h-3" />
              <span className="capitalize">{currentUser.role || 'authenticated'}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="sd-live-dot" />
              <span className="text-xs font-semibold text-slate-700">Guest Access</span>
            </div>
            <button
              onClick={onOpenAuthModal}
              className="w-full py-2 px-3 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-sm"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In / Register</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};

