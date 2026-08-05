import { useEffect, useState } from "react";
import { apiPreprocess, apiRecommend } from "../../services/api";
import { ErrorBox } from "../Alert";
import Spinner from "../Spinner";

const PRIORITY_STYLE = {
  high: "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300",
  medium: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300",
  low: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300",
};

export default function PreprocessStep({ datasetId, target, onComplete }) {
  const [mode, setMode] = useState("auto");
  const [config, setConfig] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [manual, setManual] = useState({
    missing_numeric: "median",
    missing_categorical: "mode",
    encoding: "label",
    scaling: "none",
    smote: false,
  });
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setReport(null);
    setError(null);
    Promise.all([
      apiPreprocess.autoConfig(datasetId, { target }),
      apiRecommend.forDataset(datasetId, { target }),
    ])
      .then(([cfg, rec]) => {
        if (!alive) return;
        setConfig(cfg.config);
        setRecommendation(rec);
        setManual({
          missing_numeric: cfg.config.missing_numeric === "none" ? "median" : cfg.config.missing_numeric,
          missing_categorical: cfg.config.missing_categorical === "none" ? "mode" : cfg.config.missing_categorical,
          encoding: cfg.config.encoding,
          scaling: cfg.config.scaling,
          smote: cfg.config.smote,
        });
      })
      .catch((e) => alive && setError(e))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [datasetId, target]);

  const payload = () =>
    mode === "auto"
      ? { dataset_id: datasetId, mode: "auto", target_column: target }
      : {
          dataset_id: datasetId,
          mode: "manual",
          target_column: target,
          ...manual,
        };

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await apiPreprocess.run(payload());
      setReport(res);
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  const recs = recommendation;
  const profile = recs?.dataset_profile;

  const manualControls = [
    { key: "missing_numeric", label: "Numeric missing values", options: ["mean", "median", "mode", "drop"] },
    { key: "missing_categorical", label: "Categorical missing values", options: ["mode", "constant", "drop"] },
    { key: "encoding", label: "Categorical encoding", options: ["label", "onehot"] },
    { key: "scaling", label: "Feature scaling", options: ["none", "standard", "minmax"] },
  ];

  return (
    <div className="space-y-5">
      <ErrorBox error={error} onClose={() => setError(null)} />

      {/* Profile summary */}
      {profile && (
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Dataset Profile</h3>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <MiniStat label="Rows" value={profile.rows} />
            <MiniStat label="Features" value={profile.columns} />
            <MiniStat label="Missing" value={`${profile.missing_pct}%`} />
            <MiniStat label="Imbalance Ratio" value={profile.imbalance_ratio ?? "—"} />
            <MiniStat label="Classes" value={profile.num_classes} />
          </div>
        </div>
      )}

      {/* Mode toggle */}
      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Preprocessing Strategy</h3>
            <p className="text-xs text-slate-400">One-click auto mode uses the recommendation engine's heuristics.</p>
          </div>
          <div className="grid grid-cols-2 rounded-xl bg-slate-200/70 p-1 dark:bg-slate-800">
            <button
              onClick={() => setMode("auto")}
              className={`rounded-lg px-4 py-1.5 text-sm font-semibold transition ${
                mode === "auto" ? "bg-white text-slate-900 shadow dark:bg-slate-900 dark:text-white" : "text-slate-500"
              }`}
            >
              Auto
            </button>
            <button
              onClick={() => setMode("manual")}
              className={`rounded-lg px-4 py-1.5 text-sm font-semibold transition ${
                mode === "manual" ? "bg-white text-slate-900 shadow dark:bg-slate-900 dark:text-white" : "text-slate-500"
              }`}
            >
              Manual
            </button>
          </div>
        </div>

        {loading ? (
          <div className="py-8"><Spinner label="Analyzing dataset…" /></div>
        ) : mode === "auto" && config ? (
          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
              <div className="label">Recommended pipeline (auto)</div>
              <ul className="space-y-1.5 text-sm text-slate-700 dark:text-slate-300">
                <li>• Numeric missing → <b>{config.missing_numeric}</b></li>
                <li>• Categorical missing → <b>{config.missing_categorical}</b></li>
                <li>• Encoding → <b>{config.encoding}</b></li>
                <li>• Scaling → <b>{config.scaling}</b></li>
                <li>• SMOTE → <b>{config.smote ? "Yes" : "No"}</b></li>
              </ul>
            </div>
            <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-950">
              <div className="label">Why the engine recommends this</div>
              <ul className="space-y-1.5 text-sm text-slate-700 dark:text-slate-300">
                {(recommendation?.preprocessing_recommendations || []).slice(0, 4).map((r, i) => (
                  <li key={i}>
                    <span className={`badge mr-1.5 ${PRIORITY_STYLE[r.priority] || PRIORITY_STYLE.low}`}>{r.priority}</span>
                    {r.detail}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
            {manualControls.map((c) => (
              <div key={c.key}>
                <label className="label">{c.label}</label>
                <select
                  className="input"
                  value={manual[c.key]}
                  onChange={(e) => setManual((m) => ({ ...m, [c.key]: e.target.value }))}
                >
                  {c.options.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
            ))}
            <label className="flex items-center gap-3 rounded-xl border border-slate-200 p-3 dark:border-slate-700 md:col-span-2">
              <input
                type="checkbox"
                checked={manual.smote}
                onChange={(e) => setManual((m) => ({ ...m, smote: e.target.checked }))}
                className="h-4 w-4 rounded accent-brand-600"
              />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
                Apply SMOTE oversampling (handles class imbalance)
              </span>
            </label>
          </div>
        )}
      </div>

      {/* Recommended model chips */}
      {recommendation?.model_recommendations?.length > 0 && (
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Top Model Recommendations</h3>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {recommendation.model_recommendations.slice(0, 3).map((r) => (
              <div key={r.model_type} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{r.model_type}</span>
                  <span className="badge bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300">
                    {r.score}/100
                  </span>
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-slate-500 dark:text-slate-400">
                  {r.reasons.slice(0, 2).map((rs, i) => (
                    <li key={i}>{rs}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Report */}
      {report && (
        <div className="card border-emerald-200 p-5 dark:border-emerald-500/30 animate-fade-in">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-bold text-emerald-700 dark:text-emerald-300">
            ✓ Preprocessing complete
          </h3>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MiniStat label="Samples" value={report.report.sample_count} />
            <MiniStat label="Features" value={report.report.feature_count} />
            <MiniStat label="Rows dropped" value={report.report.dropped_rows} />
            <MiniStat label="SMOTE" value={report.report.smote_applied ? "Applied" : "Skipped"} />
          </div>
          {report.report.imputations.length > 0 && (
            <div className="mt-3 text-xs text-slate-500 dark:text-slate-400">
              Imputed: {report.report.imputations.join(", ")}
            </div>
          )}
        </div>
      )}

      <div className="flex justify-end">
        <button className="btn-primary" onClick={run} disabled={running || loading}>
          {running ? "Processing…" : report ? "Re-run preprocessing" : "Run preprocessing"}
        </button>
      </div>
      {report && (
        <div className="flex justify-end">
          <button className="btn-primary" onClick={() => onComplete(report.config_used)}>
            Continue → Train Model
          </button>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3 text-center dark:bg-slate-950">
      <div className="text-lg font-extrabold text-slate-900 dark:text-white">{value}</div>
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}
