import { useEffect, useState } from "react";
import { apiCompare, apiDatasets, apiExport, downloadBlob } from "../services/api";
import { ErrorBox } from "../components/Alert";
import Spinner from "../components/Spinner";
import ImageCard from "../components/ImageCard";

const BASE_TYPES = ["dt", "knn", "rf"];
const HYBRID_TYPES = ["voting", "stacking"];
const ALGO_LABELS = {
  dt: "Decision Tree",
  knn: "K-Nearest Neighbors",
  rf: "Random Forest",
  voting: "Hybrid Voting Ensemble",
  stacking: "Stacking Ensemble",
};

export default function Compare() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [target, setTarget] = useState("");
  const [columns, setColumns] = useState([]);
  const [hybrid, setHybrid] = useState(true);
  const [baseSel, setBaseSel] = useState(["dt", "knn", "rf"]);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiDatasets
      .list()
      .then((list) => {
        setDatasets(list);
        if (list.length) {
          setDatasetId(list[0].id);
          setColumns(list[0].columns || []);
          setTarget(list[0].profile?.target_column || "");
        }
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  const selectDataset = (id) => {
    const ds = datasets.find((d) => d.id === Number(id));
    setDatasetId(Number(id));
    setColumns(ds?.columns || []);
    setTarget(ds?.profile?.target_column || "");
  };

  const toggleBase = (mt) => {
    setBaseSel((sel) => {
      if (sel.includes(mt)) {
        return sel.length > 1 ? sel.filter((x) => x !== mt) : sel; // keep at least one
      }
      return [...sel, mt];
    });
  };

  const modelTypes = [...baseSel, ...(hybrid ? HYBRID_TYPES : [])];

  const run = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiCompare.run({
        dataset_id: datasetId,
        target_column: target,
        preprocess: { mode: "auto", target_column: target },
        model_types: modelTypes,
      });
      setResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  };

  const downloadBest = async () => {
    const blob = await apiExport.model(result.best_model.model_id);
    downloadBlob(blob, `${result.best_model.model_type}_best_model.pkl`);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">Model Comparison Engine</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Train your chosen algorithms on the same train/test split and find the winner.
        </p>
      </div>

      <ErrorBox error={error} onClose={() => setError(null)} />

      <div className="card p-5">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <label className="label">Dataset</label>
            {loading ? (
              <Spinner label="Loading…" />
            ) : (
              <select className="input" value={datasetId} onChange={(e) => selectDataset(e.target.value)}>
                <option value="">Select dataset…</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            )}
          </div>
          <div>
            <label className="label">Target column</label>
            <select className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Select target…</option>
              {columns.filter((c) => c !== target).map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 pb-2.5 text-sm font-medium text-slate-700 dark:text-slate-200">
              <input
                type="checkbox"
                checked={hybrid}
                onChange={(e) => setHybrid(e.target.checked)}
                className="h-4 w-4 accent-brand-600"
              />
              Hybrid mode (voting + stacking)
            </label>
          </div>
        </div>

        {/* Algorithm selection */}
        <div className="mt-4 border-t border-slate-100 pt-4 dark:border-slate-800">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span className="text-xs font-bold uppercase tracking-wide text-slate-400">
              Algorithms to compare
            </span>
            {BASE_TYPES.map((mt) => (
              <label key={mt} className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
                <input
                  type="checkbox"
                  checked={baseSel.includes(mt)}
                  onChange={() => toggleBase(mt)}
                  className="h-4 w-4 accent-brand-600"
                />
                {ALGO_LABELS[mt]}
              </label>
            ))}
            {HYBRID_TYPES.map((mt) => (
              <label
                key={mt}
                className={`flex items-center gap-2 text-sm font-medium ${hybrid ? "text-slate-700 dark:text-slate-200" : "text-slate-300 dark:text-slate-600"}`}
              >
                <input
                  type="checkbox"
                  checked={hybrid}
                  disabled
                  className="h-4 w-4 accent-brand-600"
                />
                {ALGO_LABELS[mt]} {hybrid && <span className="text-xs text-brand-500">(auto)</span>}
              </label>
            ))}
          </div>
          {!hybrid && (
            <p className="mt-2 text-xs text-slate-400">
              Hybrid mode is off — pick the base algorithms you want to compare.
            </p>
          )}
        </div>

        <div className="mt-4 flex justify-end">
          <button className="btn-primary" onClick={run} disabled={running || !datasetId || !target || modelTypes.length === 0}>
            {running ? "Comparing…" : "⚖️ Run comparison"}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="card p-5 animate-fade-in">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Leaderboard</h3>
              {result.best_model && (
                <div className="flex items-center gap-3">
                  <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
                    🏆 Winner: {result.best_model.model} · {(result.best_model.accuracy * 100).toFixed(1)}%
                  </span>
                  <button onClick={downloadBest} className="btn-secondary !py-1.5 text-xs">
                    ⬇ Export best
                  </button>
                </div>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Accuracy</th>
                    <th className="py-2 pr-4">Precision</th>
                    <th className="py-2 pr-4">Recall</th>
                    <th className="py-2 pr-4">F1</th>
                    <th className="py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {result.table.map((r) => {
                    const isBest = result.best_model?.model_id === r.model_id;
                    return (
                      <tr key={r.model_type} className={`border-b border-slate-100 dark:border-slate-800 ${isBest ? "bg-emerald-50/60 dark:bg-emerald-500/5" : ""}`}>
                        <td className="py-2.5 pr-4 font-semibold text-slate-800 dark:text-slate-100">
                          {r.model} {isBest && <span className="ml-1">👑</span>}
                        </td>
                        <td className="py-2.5 pr-4 font-bold text-emerald-600 dark:text-emerald-400">{(r.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-300">{(r.precision_weighted * 100).toFixed(1)}%</td>
                        <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-300">{(r.recall_weighted * 100).toFixed(1)}%</td>
                        <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-300">{(r.f1_weighted * 100).toFixed(1)}%</td>
                        <td className="py-2.5">
                          <a
                            href="#workflow"
                            className="text-xs font-semibold text-brand-600 hover:underline dark:text-brand-400"
                            onClick={(e) => e.preventDefault()}
                          >
                            model #{r.model_id}
                          </a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ImageCard src={result.charts.accuracy_comparison} title="Accuracy Comparison" alt="Accuracy bar chart" />
            <ImageCard src={result.charts.metric_radar} title="Metric Profile Radar" alt="Radar chart" />
          </div>
        </>
      )}
    </div>
  );
}
