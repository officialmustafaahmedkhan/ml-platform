export default function Logo({ size = 36 }) {
  return (
    <div className="flex items-center gap-2.5">
      <img
        src="/logo.png"
        alt="ModelMind AI logo"
        width={size}
        height={size}
        className="shrink-0 rounded-lg object-contain"
      />
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
