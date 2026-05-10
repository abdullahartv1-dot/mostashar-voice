import {
  LayoutDashboard,
  Type,
  Bot,
  BarChart3,
  BookAudio,
  UserPlus,
  Mic,
  Globe,
  Sliders,
  Settings,
  HelpCircle,
  Grid2x2,
} from 'lucide-react';

const menuItems = [
  { icon: LayoutDashboard, label: 'Dashboard' },
  { icon: Type, label: 'Text-to-Speech' },
  { icon: Bot, label: 'Voice Agents' },
  { icon: BarChart3, label: 'Agent Metrics' },
  { icon: BookAudio, label: 'Voice Library' },
  { icon: UserPlus, label: 'Instant Clone', active: true },
  { icon: Mic, label: 'Pro Voice Clone' },
  { icon: Globe, label: 'Localize a Voice' },
  { icon: Sliders, label: 'Voice Changer' },
  { icon: Settings, label: 'Settings' },
  { icon: HelpCircle, label: 'Help' },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-[240px] bg-[#141414] border-r border-[#2a2a2a] flex flex-col z-50">
      {/* App Logo */}
      <div className="flex items-center gap-2 px-4 py-4">
        <Grid2x2 className="w-5 h-5 text-[#f0f0f0]" />
        <span className="text-[16px] font-semibold text-[#f0f0f0]">VoiceClip Editor</span>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 overflow-y-auto px-2 py-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.active;
          return (
            <button
              key={item.label}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-150 mb-0.5 ${
                isActive
                  ? 'bg-[#222222] text-[#f0f0f0] border-l-2 border-[#3b82f6]'
                  : 'text-[#a0a0a0] hover:bg-[#222222] hover:text-[#f0f0f0]'
              }`}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom User Section */}
      <div className="px-4 py-3 border-t border-[#2a2a2a]">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-[#3b82f6] flex items-center justify-center text-[10px] font-semibold text-white">
            U
          </div>
          <span className="text-xs text-[#a0a0a0] truncate flex-1">User&apos;s workspace</span>
          <button className="text-[#666666] hover:text-[#a0a0a0] transition-colors">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="1" />
              <circle cx="19" cy="12" r="1" />
              <circle cx="5" cy="12" r="1" />
            </svg>
          </button>
        </div>
      </div>
    </aside>
  );
}
