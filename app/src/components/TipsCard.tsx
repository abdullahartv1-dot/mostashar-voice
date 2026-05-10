import { CircleCheck } from 'lucide-react';

const tips = [
  'Avoid long silences and pauses',
  "Pacing and volume you'd like the cloned voice to match",
  'Trim your clip before submission to remove silence, clicks, pops, etc.',
];

export default function TipsCard() {
  return (
    <div className="bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] p-6 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
      <p className="text-sm font-medium text-[#f0f0f0] mb-3">For best results:</p>
      
      <div className="space-y-2.5">
        {tips.map((tip, index) => (
          <div key={index} className="flex items-start gap-2.5">
            <CircleCheck className="w-4 h-4 text-[#22c55e] flex-shrink-0 mt-0.5" />
            <span className="text-sm text-[#a0a0a0]">{tip}</span>
          </div>
        ))}
      </div>

      <a
        href="#"
        className="inline-block mt-3 text-xs text-[#3b82f6] hover:underline transition-all"
      >
        Check out our docs for more ways to improve your clone
      </a>

      <p className="text-xs text-[#666666] mt-4">
        The maximum file size is 4 MB.
      </p>
    </div>
  );
}
