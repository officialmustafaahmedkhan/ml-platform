import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiDashboard, apiDatasets, apiTrain } from "../services/api";
import { ErrorBox } from "../components/Alert";
import Spinner from "../components/Spinner";

const ACTION_LABELS = {
  upload: "📁 Uploaded dataset",
  preprocess: "🧹 Ran preprocessing",
  train: "🤖 Trained model",
  predict: "🔮 Made prediction",
  batch_predict: "📦 Batch prediction",
  compare: "⚖️ Compared models",
};

export default function History() {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const [dash, m, ds] = await Promise.all([
        apiDashboard.get(),
        apiTrain.models(),
        apiDatasets.list(),
      ]);
      setTimeline(dash.activity_timeline);
      setModels(m);
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

  const removeModel = async (id) => {
    if (!confirm("Delete this trained model?")) return;
    await apiTrain.remove(id);
    load();
  };

  const removeDataset = async (id) => {
    if (!confirm("Delete this dataset and its models?")) return;
    await apiDatasets.remove(id);
    load();
  };

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Spinner label="Loading history…" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">Experiment History</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Everything you've trained, uploaded and predicted.
        </p>
      </div>

      <ErrorBox error={error} onClose={() => setError(null)} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Models */}
        <div className="card p-5 lg:col-span-2">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Trained Models ({models.length})</h3>
          {models.length === 0 ? (
            <Empty text="No models yet. Train one in the workflow." />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Type</th>
                    <th className="py-2 pr-4">Accuracy</th>
                    <th className="py-2 pr-4">Created</th>
                    <th className="py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.id} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="py-2.5 pr-4 font-medium text-slate-800 dark:text-slate-100">{m.name}</td>
                      <td className="py-2.5 pr-4">
                        <span className="badge bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300">
                          {m.model_type}
                        </span>
                      </td>
                      <td className="py-2.5 pr-4">
                        {m.metrics?.metrics?.accuracy != null ? (
                          <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
                            {(m.metrics.metrics.accuracy * 100).toFixed(1)}%
                          </span>
                        ) : "—"}
                      </td>
                      <td className="py-2.5 pr-4 text-xs text-slate-400">
                        {new Date(m.created_at).toLocaleString()}
                      </td>
                      <td className="py-2.5 text-right">
                        <button
                          onClick={() => removeModel(m.id)}
                          className="text-xs font-semibold text-rose-500 hover:underline"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <h3 className="mb-3 mt-6 text-sm font-bold text-slate-800 dark:text-slate-100">Datasets ({datasets.length})</h3>
          {datasets.length === 0 ? (
            <Empty text="No datasets yet." />
          ) : (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {datasets.map((d) => (
                <li key={d.id} className="flex items-center justify-between py-2.5">
                  <div>
                    <div className="text-sm font-medium text-slate-800 dark:text-slate-100">{d.name}</div>
                    <div className="text-xs text-slate-400">
                      {d.rows} rows · {d.columns?.length} cols · {new Date(d.created_at).toLocaleString()}
                    </div>
                  </div>
                  <button onClick={() => removeDataset(d.id)} className="text-xs font-semibold text-rose-500 hover:underline">
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Timeline */}
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Activity Timeline</h3>
          {timeline.length === 0 ? (
            <Empty text="No activity yet." />
          ) : (
            <ul className="relative ml-3 border-l-2 border-slate-200 pl-5 dark:border-slate-800">
              {timeline.map((e, i) => (
                <li key={i} className="relative pb-6 last:pb-0">
                  <span className="absolute top-1 -left-[27px] h-3 w-3 rounded-full border-2 border-white bg-brand-500 dark:border-slate-900" />
                  <div className="text-sm font-medium text-slate-800 dark:text-slate-100">
                    {ACTION_LABELS[e.action] || e.action}
                  </div>
                  <div className="mt-0.5 text-xs text-slate-400">
                    {new Date(e.created_at).toLocaleString()}
                  </div>
                  {e.details?.accuracy != null && (
                    <div className="mt-1 text-xs text-emerald-600 dark:text-emerald-400">
                      Accuracy: {(e.details.accuracy * 100).toFixed(1)}%
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}

          <div className="mt-6 border-t border-slate-100 pt-4 dark:border-slate-800">
            <Link to="/workflow" className="btn-primary w-full">
              + New experiment
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function Empty({ text }) {
  return (
    <div className="flex min-h-[100px] items-center justify-center rounded-xl bg-slate-50 text-sm text-slate-400 dark:bg-slate-950">
      {text}
    </div>
  );
}
