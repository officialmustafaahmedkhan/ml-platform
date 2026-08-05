import { useState } from "react";
import { apiExplain, apiExport, apiPredict, downloadBlob } from "../../services/api";
import { ErrorBox } from "../Alert";

export default function PredictStep({ modelId, dataset, target, classNames = [] }) {
  const columns = (dataset?.columns || []).filter((c) => c !== target);
  const numericCols = dataset?.profile?.numeric_columns || [];
  const catCols = dataset?.profile?.categorical_columns || [];

  const [values, setValues] = useState(() => {
    const init = {};
    columns.forEach((c) => (init[c] = ""));
    return init;
  });
  const [result, setResult] = useState(null);
  const [localExp, setLocalExp] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  const [batch, setBatch] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [batchRunning, setBatchRunning] = useState(false);

  const predict = async () => {
    setPredicting(true);
    setLocalExp(null);
    setError(null);
    try {
      const res = await apiPredict.single(modelId, values);
      setResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setPredicting(false);
    }
  };

  const explain = async () => {
    setExplaining(true);
    setError(null);
    try {
      const res = await apiExplain.local(modelId, values);
      setLocalExp(res);
    } catch (e) {
      setError(e);
    } finally {
      setExplaining(false);
    }
  };

  const runBatch = async () => {
    if (!batchFile) return;
    setBatchRunning(true);
    setError(null);
    try {
      const res = await apiPredict.batch(modelId, batchFile);
      setBatchResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setBatchRunning(false);
    }
  };

  const downloadBatch = async () => {
    const blob = await apiExport.batch(batchResult.output_filename);
    downloadBlob(blob, batchResult.output_filename);
  };

  const maxProb = result?.probabilities
    ? Math.max(...Object.values(result.probabilities))
    : 0;

  return (
    <div className="space-y-5">
      <ErrorBox error={error} onClose={() => setError(null)} />

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* Manual prediction */}
        <div className="card p-5">
          <h3 className="mb-4 text-sm font-bold text-slate-800 dark:text-slate-100">🔮 Manual input</h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {columns.map((c) => (
              <div key={c}>
                <label className="label">{c}</label>
                {catCols.includes(c) ? (
                  <select className="input" value={values[c]} onChange={(e) => setValues((v) => ({ ...v, [c]: e.target.value }))}>
                    <option value="">Select…</option>
                    {(dataset?.preview || []).length
                      ? [...new Set(dataset.preview.map((r) => r[c]))].map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))
                      : null}
                  </select>
                ) : (
                  <input
                    className="input"
                    value={values[c]}
                    placeholder="Enter value"
                    onChange={(e) => setValues((v) => ({ ...v, [c]: e.target.value }))}
                  />
                )}
              </div>
            ))}
          </div>
          <button className="btn-primary mt-4 w-full" onClick={predict} disabled={predicting}>
            {predicting ? "Predicting…" : "🎯 Predict"}
          </button>

          {result && (
            <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-500/30 dark:bg-emerald-500/10 animate-fade-in">
              <div className="text-xs font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
                Predicted Class
              </div>
              <div className="mt-1 text-3xl font-extrabold text-emerald-700 dark:text-emerald-300">
                {result.prediction}
              </div>

              {result.probabilities && (
                <div className="mt-4 space-y-2">
                  {Object.entries(result.probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center gap-3">
                      <span className="w-16 shrink-0 text-xs font-semibold text-slate-600 dark:text-slate-300">{cls}</span>
                      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                        <div
                          className={`h-full rounded-full transition-all ${prob === maxProb ? "bg-emerald-500" : "bg-brand-500"}`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="w-12 shrink-0 text-right text-xs font-bold text-slate-700 dark:text-slate-200">
                        {(prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {result.explanation?.length > 0 && (
                <div className="mt-4 border-t border-emerald-200/60 pt-3 dark:border-emerald-500/20">
                  <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                    Top contributing features
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-2">
                    {result.explanation.map((f) => (
                      <span key={f.feature} className="badge bg-white text-slate-600 shadow-sm dark:bg-slate-800 dark:text-slate-300">
                        {f.feature} · {f.importance.toFixed(3)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-4 border-t border-emerald-200/60 pt-3 dark:border-emerald-500/20">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                    Why this prediction? (LIME-style)
                  </div>
                  {!localExp && (
                    <button
                      onClick={explain}
                      disabled={explaining}
                      className="text-xs font-semibold text-brand-600 hover:underline dark:text-brand-400"
                    >
                      {explaining ? "Analyzing…" : "Explain"}
                    </button>
                  )}
                </div>
                {localExp?.attributions?.length > 0 && (
                  <div className="mt-2 space-y-1.5">
                    {localExp.attributions.slice(0, 6).map((a) => (
                      <div key={a.feature} className="flex items-center gap-2 text-xs">
                        <span className="w-28 shrink-0 truncate font-medium text-slate-600 dark:text-slate-300">
                          {a.feature}
                        </span>
                        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                          <div
                            className={`absolute inset-y-0 ${a.contribution >= 0 ? "left-1/2 bg-emerald-500" : "right-1/2 bg-rose-500"}`}
                            style={{ width: `${Math.min(Math.abs(a.contribution) * 40, 50)}%` }}
                          />
                          <div className="absolute left-1/2 top-0 h-full w-px bg-slate-400/60" />
                        </div>
                        <span className="w-16 shrink-0 text-right font-bold text-slate-700 dark:text-slate-200">
                          {a.contribution >= 0 ? "+" : ""}{a.contribution.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Batch prediction */}
        <div className="card p-5">
          <h3 className="mb-4 text-sm font-bold text-slate-800 dark:text-slate-100">📦 Batch prediction (CSV)</h3>
          <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 p-8 text-center transition hover:border-brand-400 dark:border-slate-700">
            <input
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                setBatchFile(e.target.files?.[0] || null);
                setBatchResult(null);
              }}
            />
            <span className="text-3xl">📄</span>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {batchFile ? batchFile.name : "Upload CSV for batch prediction"}
            </span>
            <span className="text-xs text-slate-400">Must contain the same feature columns</span>
          </label>
          <button className="btn-primary mt-4 w-full" onClick={runBatch} disabled={!batchFile || batchRunning}>
            {batchRunning ? "Predicting batch…" : "Run batch prediction"}
          </button>

          {batchResult && (
            <div className="mt-4 animate-fade-in">
              <div className="mb-2 flex items-center justify-between">
                <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
                  {batchResult.total} predictions
                </span>
                <button onClick={downloadBatch} className="text-xs font-semibold text-brand-600 hover:underline dark:text-brand-400">
                  ⬇ Download CSV
                </button>
              </div>
              <div className="max-h-72 overflow-auto rounded-xl border border-slate-200 dark:border-slate-700">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                    <tr>
                      {Object.keys(batchResult.results[0] || {}).map((k) => (
                        <th key={k} className="px-2 py-1.5 font-semibold text-slate-500 dark:text-slate-300">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {batchResult.results.slice(0, 20).map((row, i) => (
                      <tr key={i} className="border-t border-slate-100 dark:border-slate-800">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className={`px-2 py-1 ${j === 0 ? "font-bold text-emerald-700 dark:text-emerald-300" : "text-slate-600 dark:text-slate-300"}`}>
                            {typeof v === "number" ? Number(v).toFixed(4) : v}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
