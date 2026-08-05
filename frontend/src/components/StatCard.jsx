export default function StatCard({ icon, label, value, sub, accent = "brand" }) {
  const accents = {
    brand: "bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400",
    emerald: "bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400",
    amber: "bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400",
    rose: "bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400",
    sky: "bg-sky-50 text-sky-600 dark:bg-sky-500/10 dark:text-sky-400",
  };
  return (
    <div className="card flex items-start gap-4 p-5 animate-slide-up">
      <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${accents[accent]}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {label}
        </div>
        <div className="mt-0.5 text-2xl font-extrabold text-slate-900 dark:text-white">{value}</div>
        {sub && (
          <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{sub}</div>
        )}
      </div>
    </div>
  );
}
