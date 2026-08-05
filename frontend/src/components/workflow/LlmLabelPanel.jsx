import { useEffect, useState } from "react";
import { apiLlm } from "../../services/api";
import { Alert } from "../Alert";
import Spinner from "../Spinner";

const PROVIDER_LABEL = {
  off: "off",
  openai: "OpenAI",
  ollama: "Ollama (local)",
};

export default function LlmLabelPanel({ datasetId, existingColumns, onLabeled }) {
  const [status, setStatus] = useState(null);
  const [column, setColumn] = useState("Outcome");
  const [numCategories, setNumCategories] = useState(3);
  const [batchSize, setBatchSize] = useState(25);
  const [maxRows, setMaxRows] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    apiLlm.status().then(setStatus).catch(() => setStatus(null));
  }, []);

  const enabled = status?.enabled === true;
  const columnTaken = existingColumns.includes(column);
  const counts = result?.counts || {};
  const totalLabeled = result?.labeled_rows ?? 0;

  const run = async () => {
    setError(null);
    setResult(null);
    setRunning(true);
    try {
      const res = await apiLlm.label(datasetId, {
        column_name: column,
        num_categories: numCategories,
        batch_size: batchSize,
        max_rows: maxRows ? Number(maxRows) : undefined,
      });
      setResult(res);
      onLabeled?.(res);
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 p-5 dark:border-slate-700">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
          ✨ LLM Outcome labeling
        </h3>
        <span className="badge bg-violet-50 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
          {status ? `Provider: ${PROVIDER_LABEL[status.provider] || status.provider}` : "…"}
        </span>
      </div>

      {status && !enabled && (
        <div className="mt-3">
          <Alert kind="info">
            LLM labeling is disabled. Set <code className="font-mono">LLM_PROVIDER=openai</code> or{" "}
            <code className="font-mono">ollama</code> in the backend <code className="font-mono">.env</code> and
            restart the server to enable it.
          </Alert>
        </div>
      )}

      {enabled && (
        <>
          {error && (
            <div className="mt-3">
              <Alert kind="error" title="Labeling failed" onClose={() => setError(null)}>
                {error?.response?.data?.detail || error?.message || String(error)}
              </Alert>
            </div>
          )}

          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label className="label">New column name</label>
              <input
                className="input"
                value={column}
                onChange={(e) => setColumn(e.target.value)}
                placeholder="Outcome"
              />
              {columnTaken && (
                <p className="mt-1 text-xs text-rose-500">Column already exists</p>
              )}
            </div>
            <div>
              <label className="label">Categories (2–6)</label>
              <input
                className="input"
                type="number"
                min={2}
                max={6}
                value={numCategories}
                onChange={(e) => setNumCategories(Math.max(2, Math.min(6, Number(e.target.value) || 2)))}
              />
            </div>
            <div>
              <label className="label">Batch size</label>
              <input
                className="input"
                type="number"
                min={1}
                max={100}
                value={batchSize}
                onChange={(e) => setBatchSize(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
              />
            </div>
            <div>
              <label className="label">Max rows (empty = all)</label>
              <input
                className="input"
                type="number"
                min={1}
                value={maxRows}
                onChange={(e) => setMaxRows(e.target.value)}
                placeholder="All"
              />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              className="btn-primary"
              disabled={running || columnTaken || !column.trim()}
              onClick={run}
            >
              {running ? "Labeling…" : "Generate labels"}
            </button>
            {running && <Spinner label="Calling LLM…" />}
          </div>

          {result && (
            <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200">
              <div className="font-semibold">
                Created “{result.column_name}” — {totalLabeled} rows labeled
                {result.counts?.Unknown != null && (
                  <span> ({result.counts.Unknown.toLocaleString()} pending)</span>
                )}
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {Object.entries(counts).map(([k, v]) => (
                  <span
                    key={k}
                    className="badge bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200"
                  >
                    {k}: {Number(v).toLocaleString()}
                  </span>
                ))}
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {(result.categories || []).map((c) => (
                  <span key={c.name} className="rounded-lg bg-white/60 px-2 py-0.5 text-xs dark:bg-black/20">
                    {c.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
