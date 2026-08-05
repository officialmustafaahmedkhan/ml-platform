import { useCallback, useEffect, useRef, useState } from "react";
import { apiDatasets } from "../../services/api";
import { ErrorBox } from "../Alert";
import Spinner from "../Spinner";
import LlmLabelPanel from "./LlmLabelPanel";

export default function UploadStep({ onComplete }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [target, setTarget] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setDatasets(await apiDatasets.list());
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const selectDataset = async (id) => {
    setError(null);
    setSelectedId(id);
    setTarget("");
    try {
      const ds = await apiDatasets.get(id);
      setDataset(ds);
      const profile = ds.profile || {};
      setTarget(profile.target_column || "");
    } catch (e) {
      setError(e);
    }
  };

  const onFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const ds = await apiDatasets.upload(file);
      await refresh();
      await selectDataset(ds.id);
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setError(err);
    } finally {
      setUploading(false);
    }
  };

  const columns = dataset ? (dataset.columns || []) : [];
  const targetCandidates = columns.filter((c) => c !== (dataset?.profile?.target_column || ""));

  const ready = dataset && target;

  return (
    <div className="space-y-5">
      <ErrorBox error={error} onClose={() => setError(null)} />

      {/* Upload + pick */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">1 · Upload a new CSV</h3>
          <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 p-8 text-center transition hover:border-brand-400 hover:bg-brand-50/40 dark:border-slate-700 dark:hover:bg-brand-500/5">
            <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFile} disabled={uploading} />
            <span className="text-3xl">📁</span>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {uploading ? "Uploading…" : "Click to upload CSV"}
            </span>
            <span className="text-xs text-slate-400">Your file stays private to your account</span>
          </label>
        </div>

        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">2 · Or pick a saved dataset</h3>
          {loading ? (
            <Spinner label="Loading datasets…" />
          ) : datasets.length === 0 ? (
            <p className="text-sm text-slate-400">No datasets yet. Upload one on the left.</p>
          ) : (
            <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
              {datasets.map((d) => (
                <button
                  key={d.id}
                  onClick={() => selectDataset(d.id)}
                  className={`flex w-full items-center justify-between rounded-xl border px-3 py-2.5 text-left text-sm transition ${
                    selectedId === d.id
                      ? "border-brand-500 bg-brand-50 dark:border-brand-500 dark:bg-brand-500/10"
                      : "border-slate-200 hover:border-slate-300 dark:border-slate-700 dark:hover:border-slate-600"
                  }`}
                >
                  <span className="min-w-0 truncate font-medium text-slate-800 dark:text-slate-100">{d.name}</span>
                  <span className="ml-2 shrink-0 text-xs text-slate-400">
                    {d.rows} rows · {d.columns?.length} cols
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Dataset overview */}
      {dataset && (
        <div className="card p-5 animate-fade-in">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
              {dataset.name}
            </h3>
            <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
              Loaded
            </span>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            <MiniStat label="Rows" value={dataset.rows} />
            <MiniStat label="Columns" value={dataset.columns?.length || 0} />
            <MiniStat label="Missing" value={`${dataset.profile?.missing_pct ?? 0}%`} />
            <MiniStat label="Classes" value={dataset.profile?.num_classes ?? "—"} />
          </div>

          <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div>
              <label className="label">3 · Target column to predict</label>
              <select className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">Select target…</option>
                {targetCandidates.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
              {dataset.profile?.class_counts && target && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(dataset.profile.class_counts).map(([cls, n]) => (
                    <span key={cls} className="badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      {cls}: {n}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div>
              <label className="label">Data preview</label>
              <div className="max-h-44 overflow-auto rounded-xl border border-slate-200 dark:border-slate-700">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                    <tr>
                      {(dataset.preview?.[0] ? Object.keys(dataset.preview[0]) : columns).map((c) => (
                        <th key={c} className="px-2 py-1.5 font-semibold text-slate-500 dark:text-slate-300">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(dataset.preview || []).slice(0, 5).map((row, i) => (
                      <tr key={i} className="border-t border-slate-100 dark:border-slate-800">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-2 py-1 text-slate-700 dark:text-slate-300">{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* LLM outcome labeling */}
      {dataset && (
        <LlmLabelPanel
          datasetId={dataset.id}
          existingColumns={columns}
          onLabeled={() => selectDataset(dataset.id)}
        />
      )}

      <div className="flex justify-end">
        <button className="btn-primary" disabled={!ready} onClick={() => onComplete({ dataset, target })}>
          Continue → Preprocessing
        </button>
      </div>
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
