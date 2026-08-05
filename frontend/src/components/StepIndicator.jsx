const STEPS = [
  { n: 1, label: "Upload Data" },
  { n: 2, label: "Preprocess" },
  { n: 3, label: "Train Model" },
  { n: 4, label: "Evaluate" },
  { n: 5, label: "Predict" },
];

export default function StepIndicator({ current }) {
  return (
    <ol className="flex w-full items-center">
      {STEPS.map((s, i) => {
        const active = s.n === current;
        const done = s.n < current;
        return (
          <li key={s.n} className={`flex items-center ${i < STEPS.length - 1 ? "flex-1" : ""}`}>
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold transition ${
                  done
                    ? "bg-emerald-500 text-white"
                    : active
                    ? "bg-brand-600 text-white ring-4 ring-brand-500/20"
                    : "bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-400"
                }`}
              >
                {done ? "✓" : s.n}
              </div>
              <span
                className={`hidden text-xs font-semibold sm:block ${
                  active
                    ? "text-brand-700 dark:text-brand-300"
                    : done
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-slate-400"
                }`}
              >
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div
                className={`mx-2 mb-5 h-0.5 flex-1 rounded transition ${
                  done ? "bg-emerald-500" : "bg-slate-200 dark:bg-slate-800"
                }`}
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}
