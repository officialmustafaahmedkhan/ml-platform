import { useEffect, useMemo, useState } from "react";
import { apiPipeline } from "../services/api";

const STAGE_META = {
  data: { label: "Data", icon: <IconData />, color: "bg-sky-500" },
  preprocess: { label: "Preprocessing", icon: <IconSliders />, color: "bg-violet-500" },
  train: { label: "Training", icon: <IconCog />, color: "bg-amber-500" },
  evaluate: { label: "Evaluation", icon: <IconTarget />, color: "bg-emerald-500" },
  predict: { label: "Ready to Predict", icon: <IconRocket />, color: "bg-rose-500" },
};

export default function Pipeline() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiPipeline.models().then((res) => {
      setModels(res.models || []);
      if (res.models?.length) setSelected(res.models[0].id);
    }).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    apiPipeline.detail(selected)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [selected]);

  const maxImportance = useMemo(() => {
    const ev = detail?.stages?.find((s) => s.stage === "evaluate");
    const imp = ev?.details?.top_importance || [];
    return imp.length ? Math.max(...imp.map((i) => Math.abs(i.value))) : 0;
  }, [detail]);

  const byStage = useMemo(() => {
    const map = {};
    (detail?.stages || []).forEach((s) => (map[s.stage] = s));
    return map;
  }, [detail]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-slate-50">Pipeline Visualization</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          End-to-end view of how each trained model was built — from raw data to predictions.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        {/* Model selector */}
        <div className="lg:col-span-1">
          <div className="card p-4">
            <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Trained models</h3>
            {models.length === 0 && (
              <p className="text-xs text-slate-400">No trained models yet. Train one in the ML Workflow.</p>
            )}
            <div className="space-y-2">
              {models.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelected(m.id)}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    selected === m.id
                      ? "border-brand-400 bg-brand-50 dark:border-brand-500/60 dark:bg-brand-500/10"
                      : "border-slate-200 hover:border-slate-300 dark:border-slate-700"
                  }`}
                >
                  <div className="truncate text-sm font-semibold text-slate-800 dark:text-slate-100">{m.name}</div>
                  <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 dark:bg-slate-800">#{m.id}</span>
                    <span className="truncate">{m.dataset_name || "—"}</span>
                  </div>
                  <div className="mt-1 text-xs font-bold text-emerald-600 dark:text-emerald-400">
                    acc {(m.accuracy * 100).toFixed(1)}%
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline flow */}
        <div className="lg:col-span-3">
          {loading && <div className="card p-8 text-center text-sm text-slate-400">Loading pipeline…</div>}
          {!loading && !detail && (
            <div className="card p-8 text-center text-sm text-slate-400">Select a model to view its pipeline.</div>
          )}

          {!loading && detail && (
            <div className="space-y-5">
              {/* Stage flow */}
              <div className="card p-5">
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-0">
                  {detail.stages.map((s, i) => (
                    <div key={s.stage} className="flex flex-1 flex-col items-center md:flex-row md:items-center">
                      <div className="flex flex-col items-center gap-2 text-center">
                        <div
                          className={`flex h-12 w-12 items-center justify-center rounded-2xl text-white shadow-md ${
                            STAGE_META[s.stage]?.color || "bg-slate-500"
                          }`}
                        >
                          {STAGE_META[s.stage]?.icon}
                        </div>
                        <div className="text-xs font-bold text-slate-700 dark:text-slate-200">
                          {STAGE_META[s.stage]?.label || s.title}
                        </div>
                        <div className="text-[10px] text-slate-400">{s.subtitle}</div>
                      </div>
                      {i < detail.stages.length - 1 && (
                        <svg
                          className="my-2 h-4 w-6 shrink-0 text-slate-300 dark:text-slate-600 md:mx-2"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          viewBox="0 0 24 24"
                        >
                          <path d="M5 12h14M13 6l6 6-6 6" />
                        </svg>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Stage details */}
              {detail.stages.map((s) => (
                <StageCard
                  key={s.stage}
                  stage={s}
                  maxImportance={maxImportance}
                />
              ))}

              {/* Timeline */}
              {detail.timeline?.length > 0 && (
                <div className="card p-5">
                  <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Model timeline</h3>
                  <div className="space-y-2">
                    {detail.timeline.map((e, i) => (
                      <div key={i} className="flex items-center gap-3 text-sm">
                        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-300">
                          {e.action}
                        </span>
                        <span className="text-xs text-slate-400">
                          {e.created_at ? new Date(e.created_at).toLocaleString() : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StageCard({ stage, maxImportance }) {
  const meta = STAGE_META[stage.stage] || {};
  return (
    <div className="card p-5">
      <div className="flex items-center gap-3">
        <div className={`flex h-9 w-9 items-center justify-center rounded-xl text-white ${meta.color || "bg-slate-500"}`}>
          {meta.icon}
        </div>
        <div>
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">{meta.label || stage.title}</h3>
          <p className="text-xs text-slate-400">{stage.subtitle}</p>
        </div>
      </div>

      {stage.items?.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
          {stage.items.map((it, j) => (
            <div key={j} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{it.label}</div>
              <div className="mt-0.5 truncate text-sm font-semibold text-slate-800 dark:text-slate-100">{it.value}</div>
            </div>
          ))}
        </div>
      )}

      {stage.stage === "evaluate" && stage.details?.top_importance?.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
            Top feature importance
          </div>
          <div className="space-y-1.5">
            {stage.details.top_importance.map((f, j) => (
              <div key={j} className="flex items-center gap-2 text-xs">
                <span className="w-32 truncate text-slate-600 dark:text-slate-300">{f.feature}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-brand-500"
                    style={{ width: `${maxImportance ? Math.abs(f.value) / maxImportance * 100 : 0}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-slate-500">{f.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {stage.stage === "data" && stage.details?.class_names?.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {stage.details.class_names.map((c, j) => (
            <span key={j} className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300">
              {c}
            </span>
          ))}
        </div>
      )}

      {stage.details?.feature_names?.length > 0 && (
        <div className="mt-3 text-xs text-slate-400">
          Features: <span className="text-slate-600 dark:text-slate-300">{stage.details.feature_names.join(", ")}</span>
        </div>
      )}
    </div>
  );
}

// --- Icons ----------------------------------------------------------------
function IconData() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </svg>
  );
}
function IconSliders() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5M14 4v4M8 10v4M13 16v4" />
    </svg>
  );
}
function IconCog() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1" />
    </svg>
  );
}
function IconTarget() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" />
    </svg>
  );
}
function IconRocket() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M14 10l-4 4M5 19c-1.5-1-2-2-2-3.5S4 13 5 12c.5 5 4 6.5 7 7M9 15c-1-4 2-10 9-12-1 8-5 11-9 12z" />
      <circle cx="15.5" cy="8.5" r="1.5" />
    </svg>
  );
}
