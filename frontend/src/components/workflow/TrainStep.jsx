import { useEffect, useState } from "react";
import { apiTrain } from "../../services/api";
import { ErrorBox } from "../Alert";
import Spinner from "../Spinner";

const MODEL_ICONS = {
  dt: "🌲",
  knn: "📌",
  rf: "🌳",
  voting: "🤝",
  stacking: "🧱",
};

export default function TrainStep({ datasetId, target, preprocess, onComplete }) {
  const [registry, setRegistry] = useState(null);
  const [modelType, setModelType] = useState("dt");
  const [paramsByType, setParamsByType] = useState({});
  const [tune, setTune] = useState(false);
  const [testSize, setTestSize] = useState(0.2);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiTrain
      .registry()
      .then((reg) => {
        setRegistry(reg);
        const init = {};
        Object.entries(reg).forEach(([mt, spec]) => {
          const defaults = {};
          spec.params.forEach((p) => {
            defaults[p.name] = p.allow_null ? (p.default ?? null) : p.default;
          });
          init[mt] = defaults;
        });
        setParamsByType(init);
      })
      .catch(setError);
  }, []);

  const setParam = (name, value) =>
    setParamsByType((prev) => ({ ...prev, [modelType]: { ...prev[modelType], [name]: value } }));

  const train = async () => {
    setTraining(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiTrain.train({
        dataset_id: datasetId,
        model_type: modelType,
        target_column: target,
        params: paramsByType[modelType] || {},
        preprocess,
        test_size: testSize,
        tune,
      });
      setResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setTraining(false);
    }
  };

  if (!registry) {
    return (
      <div className="flex justify-center py-16">
        <Spinner label="Loading model registry…" />
      </div>
    );
  }

  const spec = registry[modelType];
  const params = paramsByType[modelType] || {};

  return (
    <div className="space-y-5">
      <ErrorBox error={error} onClose={() => setError(null)} />

      {/* Model selection */}
      <div>
        <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">1 · Choose a model</h3>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {Object.entries(registry).map(([mt, r]) => {
            const selected = mt === modelType;
            return (
              <button
                key={mt}
                onClick={() => setModelType(mt)}
                className={`rounded-2xl border-2 p-4 text-left transition ${
                  selected
                    ? "border-brand-600 bg-brand-50 dark:border-brand-500 dark:bg-brand-500/10"
                    : "border-slate-200 hover:border-slate-300 dark:border-slate-700 dark:hover:border-slate-600"
                }`}
              >
                <div className="text-2xl">{MODEL_ICONS[mt] || "🤖"}</div>
                <div className="mt-2 text-sm font-bold text-slate-900 dark:text-white">{r.label}</div>
                <div className="mt-0.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">{r.category}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="card p-5">
        <h3 className="mb-4 text-sm font-bold text-slate-800 dark:text-slate-100">
          2 · Hyperparameters — {spec.label}
        </h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {spec.params.map((p) => (
            <div key={p.name}>
              <label className="label">
                {p.label}
                {p.help && <span title={p.help} className="ml-1 cursor-help text-slate-400">ⓘ</span>}
              </label>
              {p.type === "select" ? (
                <select className="input" value={params[p.name]} onChange={(e) => setParam(p.name, e.target.value)}>
                  {p.options.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              ) : p.allow_null ? (
                <div className="flex gap-2">
                  <input
                    type="number"
                    className="input"
                    min={p.min}
                    max={p.max}
                    value={params[p.name] ?? ""}
                    placeholder="None (auto)"
                    onChange={(e) => setParam(p.name, e.target.value === "" ? null : Number(e.target.value))}
                  />
                  {params[p.name] === null && (
                    <span className="badge bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">None</span>
                  )}
                </div>
              ) : (
                <input
                  type="number"
                  className="input"
                  min={p.min}
                  max={p.max}
                  value={params[p.name] ?? ""}
                  onChange={(e) => setParam(p.name, Number(e.target.value))}
                />
              )}
            </div>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-6 border-t border-slate-100 pt-4 dark:border-slate-800">
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
            <input type="checkbox" checked={tune} onChange={(e) => setTune(e.target.checked)} className="h-4 w-4 accent-brand-600" />
            Auto-tune (GridSearchCV)
          </label>
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
            Test size
            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              className="accent-brand-600"
            />
            <span className="badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">{Math.round(testSize * 100)}%</span>
          </label>
        </div>
      </div>

      <div className="flex justify-end">
        <button className="btn-primary" onClick={train} disabled={training}>
          {training ? "Training…" : "🚀 Train model"}
        </button>
      </div>

      {result && (
        <div className="card border-emerald-200 p-5 dark:border-emerald-500/30 animate-fade-in">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-emerald-700 dark:text-emerald-300">✓ {result.name} trained</h3>
              <p className="mt-0.5 text-xs text-slate-400">
                {result.feature_names.length} features · {result.class_names.length} classes
              </p>
            </div>
            <div className="flex gap-3">
              <MetricPill label="Accuracy" value={result.metrics.accuracy} accent="emerald" />
              <MetricPill label="Precision" value={result.metrics.precision_macro} accent="sky" />
              <MetricPill label="Recall" value={result.metrics.recall_macro} accent="amber" />
              <MetricPill label="F1" value={result.metrics.f1_macro} accent="rose" />
            </div>
          </div>
          <div className="mt-4 flex justify-end">
            <button className="btn-primary" onClick={() => onComplete(result)}>
              Continue → Evaluate
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricPill({ label, value, accent }) {
  const map = {
    emerald: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300",
    sky: "bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300",
    amber: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300",
    rose: "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300",
  };
  return (
    <div className={`rounded-xl px-3 py-2 text-center ${map[accent]}`}>
      <div className="text-lg font-extrabold">{(value * 100).toFixed(1)}%</div>
      <div className="text-[10px] font-semibold uppercase tracking-wide opacity-70">{label}</div>
    </div>
  );
}
