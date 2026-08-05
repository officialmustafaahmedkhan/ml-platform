export default function Logo({ size = 36 }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg width={size} height={size} viewBox="0 0 64 64" className="shrink-0">
        <rect width="64" height="64" rx="14" fill="#4f46e5" />
        <path
          d="M16 46 L16 34 M32 46 L32 22 M48 46 L48 28"
          stroke="#ffffff"
          strokeWidth="6"
          strokeLinecap="round"
          fill="none"
        />
        <circle cx="16" cy="28" r="6" fill="#c7d2fe" />
        <circle cx="32" cy="14" r="6" fill="#c7d2fe" />
        <circle cx="48" cy="20" r="6" fill="#c7d2fe" />
      </svg>
      <div className="leading-tight">
        <div className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white">
          ModelMind AI
        </div>
        <div className="text-[11px] font-medium text-brand-600 dark:text-brand-400">
          Hybrid Ensemble Learning
        </div>
      </div>
    </div>
  );
}
