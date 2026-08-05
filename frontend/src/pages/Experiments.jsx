import { useEffect, useMemo, useState } from "react";
import { apiDatasets, apiExperiments } from "../services/api";
import { ErrorBox } from "../components/Alert";
import Spinner from "../components/Spinner";

const ACTION_LABELS = {
  upload: "Uploaded dataset",
  preprocess: "Ran preprocessing",
  train: "Trained model",
  predict: "Made prediction",
  batch_predict: "Batch prediction",
  compare: "Compared models",
};

const ACTION_STYLE = {
  upload: "bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300",
  preprocess: "bg-violet-50 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300",
  train: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300",
  predict: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300",
  batch_predict: "bg-orange-50 text-orange-700 dark:bg-orange-500/10 dark:text-orange-300",
  compare: "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300",
};

function NoteEditor({ id, initial, onSaved }) {
  const [value, setValue] = useState(initial || "");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const dirty = (value || "") !== (initial || "");

  const save = async () => {
    setSaving(true);
    setSaved(false);
    try {
      await apiExperiments.updateNotes(id, value);
      setSaved(true);
      onSaved?.();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mt-3">
      <textarea
        className="input min-h-[64px] w-full resize-y text-xs"
        placeholder="Add a note about this experiment…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <div className="mt-1.5 flex items-center justify-end gap-3">
        {saved && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved ✓</span>}
        <button className="btn-primary px-3 py-1 text-xs" disabled={!dirty || saving} onClick={save}>
          {saving ? "Saving…" : dirty ? "Save note" : "Saved"}
        </button>
      </div>
    </div>
  );
}

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [action, setAction] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async (act) => {
    setLoading(true);
    setError(null);
    try {
      const params = act ? { action: act } : {};
      const [exps, ds] = await Promise.all([apiExperiments.list(params), apiDatasets.list()]);
      setExperiments(exps);
      setDatasets(ds);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const counts = useMemo(() => {
    const c = {};
    experiments.forEach((e) => {
      c[e.action] = (c[e.action] || 0) + 1;
    });
    return c;
  }, [experiments]);

  const totalNotes = useMemo(() => experiments.filter((e) => e.notes?.trim()).length, [experiments]);

  const filterActions = [
    ["", "All actions"],
    ["upload", "Uploads"],
    ["preprocess", "Preprocessing"],
    ["train", "Training"],
    ["predict", "Predictions"],
    ["compare", "Comparisons"],
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">Experiment Workspace</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Versioned activity log with annotatable notes for every experiment.
        </p>
      </div>

      <ErrorBox error={error} onClose={() => setError(null)} />

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        {filterActions.map(([val, label]) => (
          <button
            key={val}
            className={`rounded-full border px-4 py-1.5 text-xs font-semibold transition ${
              action === val
                ? "border-brand-500 bg-brand-500 text-white"
                : "border-slate-200 text-slate-600 hover:border-brand-400 hover:bg-brand-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-brand-500/10"
            }`}
            onClick={() => {
              setAction(val);
              load(val);
            }}
          >
            {label}
            {counts[val] ? ` (${counts[val]})` : ""}
          </button>
        ))}
        <span className="ml-auto text-xs text-slate-400">
          {experiments.length} events · {totalNotes} with notes
        </span>
      </div>

      {loading ? (
        <div className="flex justify-center py-24">
          <Spinner label="Loading experiments…" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Experiment log */}
          <div className="space-y-4 lg:col-span-2">
            {experiments.length === 0 ? (
              <div className="card flex min-h-[160px] items-center justify-center text-sm text-slate-400">
                No experiments yet. Upload a dataset or train a model in the workflow.
              </div>
            ) : (
              experiments.map((e) => (
                <div key={e.id} className="card p-5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`badge ${ACTION_STYLE[e.action] || ACTION_STYLE.train}`}>
                      {ACTION_LABELS[e.action] || e.action}
                    </span>
                    <span className="text-xs text-slate-400">{new Date(e.created_at).toLocaleString()}</span>
                  </div>
                  <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                    {e.dataset_name && <span className="font-medium text-slate-800 dark:text-slate-100">Dataset: {e.dataset_name}</span>}
                    {e.dataset_name && e.model_name && <span className="mx-2 text-slate-300 dark:text-slate-600">·</span>}
                    {e.model_name && <span className="font-medium text-slate-800 dark:text-slate-100">Model: {e.model_name}</span>}
                  </div>
                  {e.details && Object.keys(e.details).length > 0 && (
                    <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      <code className="font-mono">{JSON.stringify(e.details)}</code>
                    </div>
                  )}
                  <NoteEditor
                    id={e.id}
                    initial={e.notes || ""}
                    onSaved={() => load(action)}
                  />
                </div>
              ))
            )}
          </div>

          {/* Dataset versions */}
          <div className="space-y-4">
            <div className="card p-5">
              <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Dataset Versioning</h3>
              {datasets.length === 0 ? (
                <div className="text-sm text-slate-400">No datasets yet.</div>
              ) : (
                <div className="space-y-3">
                  {datasets.map((d) => (
                    <div key={d.id} className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
                      <div className="text-sm font-medium text-slate-800 dark:text-slate-100">{d.name}</div>
                      <div className="mt-0.5 text-xs text-slate-400">
                        {d.rows} rows · {d.columns?.length} cols
                      </div>
                      {(d.versions || []).length > 0 ? (
                        <ul className="mt-2 space-y-1">
                          {(d.versions || []).map((v) => (
                            <li key={`${v.version}-${v.path}`} className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                              <span className="badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">v{v.version}</span>
                              {v.rows != null ? `${v.rows} rows` : "snapshot"}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="mt-1 text-xs text-slate-400">v1 — initial upload</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
