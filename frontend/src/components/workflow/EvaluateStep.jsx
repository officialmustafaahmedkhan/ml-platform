import { useEffect, useState } from "react";
import { apiEvaluate, apiExplain, apiExport, downloadBlob } from "../../services/api";
import { ErrorBox } from "../Alert";
import Spinner from "../Spinner";
import ImageCard from "../ImageCard";

export default function EvaluateStep({ modelId, modelType, onComplete }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [permImp, setPermImp] = useState(null);

  useEffect(() => {
    apiEvaluate
      .get(modelId)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
    apiExplain
      .global(modelId)
      .then((r) => setPermImp(r.importance))
      .catch(() => setPermImp(null));
  }, [modelId]);

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Spinner label="Evaluating model…" />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="mx-auto max-w-lg">
        <ErrorBox error={error} onClose={() => setError(null)} />
      </div>
    );
  }

  const m = data.metrics;
  const cm = data.confusion_matrix;

  const download = async (kind, filename) => {
    let blob;
    if (kind === "model") blob = await apiExport.model(modelId);
    else if (kind === "pdf") blob = await apiExport.reportPdf(modelId);
    else blob = await apiExport.report(modelId);
    downloadBlob(blob, filename);
  };

  return (
    <div className="space-y-5">
      {/* Metric cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Accuracy" value={m.accuracy} color="emerald" />
        <MetricCard label="Precision (macro)" value={m.precision_macro} color="sky" />
        <MetricCard label="Recall (macro)" value={m.recall_macro} color="amber" />
        <MetricCard label="F1 (macro)" value={m.f1_macro} color="rose" />
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <MetricCard label="Precision (weighted)" value={m.precision_weighted} color="violet" />
        <MetricCard label="Recall (weighted)" value={m.recall_weighted} color="violet" />
        <MetricCard label="F1 (weighted)" value={m.f1_weighted} color="violet" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ImageCard src={data.charts.confusion_matrix} title="Confusion Matrix" alt="Confusion matrix heatmap" />
        <ImageCard
          src={data.charts.feature_importance}
          title="Feature Importance"
          alt="Feature importance bar chart"
          downloadName={`${modelType}_feature_importance.png`}
          onDownload={() => downloadBlobFromImage(data.charts.feature_importance, `${modelType}_feature_importance.png`)}
        />
      </div>
      {data.charts.tree && (
        <ImageCard src={data.charts.tree} title="Decision Tree Structure" alt="Decision tree plot" />
      )}

      {/* Prediction-oriented charts */}
      {data.charts.roc_curve && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ImageCard src={data.charts.roc_curve} title="ROC Curve" alt="ROC curve" />
          <ImageCard src={data.charts.precision_recall} title="Precision-Recall Curve" alt="Precision-recall curve" />
        </div>
      )}

      {/* Permutation importance (global, model-agnostic) */}
      {permImp && permImp.length > 0 && (
        <div className="card p-5">
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
            Global Feature Importance (permutation)
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            Average accuracy drop when each feature is shuffled. Higher = more important.
          </p>
          <div className="mt-4 space-y-2">
            {permImp.map((p) => (
              <div key={p.feature} className="flex items-center gap-3">
                <span className="w-32 shrink-0 truncate text-xs font-medium text-slate-600 dark:text-slate-300">
                  {p.feature}
                </span>
                <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                  <div
                    className="h-full rounded-full bg-brand-500 transition-all"
                    style={{ width: `${Math.min((p.importance / Math.max(...permImp.map((x) => x.importance))) * 100, 100)}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right text-xs font-bold text-slate-700 dark:text-slate-200">
                  {p.importance.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {data.charts.learning_curve && (
        <ImageCard src={data.charts.learning_curve} title="Learning Curve" alt="Learning curve" />
      )}
      {(data.charts.class_balance || data.charts.predicted_vs_actual) && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {data.charts.class_balance && (
            <ImageCard src={data.charts.class_balance} title="Class Distribution (Train)" alt="Class balance bar chart" />
          )}
          {data.charts.predicted_vs_actual && (
            <ImageCard src={data.charts.predicted_vs_actual} title="Predicted vs Actual (Test)" alt="Predicted vs actual bar chart" />
          )}
        </div>
      )}
      {(data.charts.probability_histogram || data.charts.correlation_heatmap) && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {data.charts.probability_histogram && (
            <ImageCard src={data.charts.probability_histogram} title="Confidence Distribution" alt="Confidence histogram" />
          )}
          {data.charts.correlation_heatmap && (
            <ImageCard src={data.charts.correlation_heatmap} title="Feature Correlation Heatmap" alt="Correlation heatmap" />
          )}
        </div>
      )}

      {/* Per-class table */}
      {m.per_class && (
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Per-Class Metrics</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400 dark:border-slate-700">
                  <th className="py-2 pr-4">Class</th>
                  <th className="py-2 pr-4">Precision</th>
                  <th className="py-2 pr-4">Recall</th>
                  <th className="py-2 pr-4">F1</th>
                  <th className="py-2">Support</th>
                </tr>
              </thead>
              <tbody>
                {m.per_class.map((pc) => (
                  <tr key={pc.class} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="py-2 pr-4 font-semibold text-slate-800 dark:text-slate-100">{pc.class}</td>
                    <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{pc.precision}</td>
                    <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{pc.recall}</td>
                    <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{pc.f1}</td>
                    <td className="py-2 text-slate-500 dark:text-slate-400">{pc.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Export */}
      <div className="card p-5">
        <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">Export</h3>
        <div className="flex flex-wrap gap-3">
          <button className="btn-secondary" onClick={() => download("model", `${modelType}_model.pkl`)}>
            ⬇ Download model (.pkl)
          </button>
          <button className="btn-secondary" onClick={() => download("report", `${modelType}_report.csv`)}>
            ⬇ Download report (.csv)
          </button>
          <button className="btn-primary" onClick={() => download("pdf", `${modelType}_report_${modelId}.pdf`)}>
            ⬇ Download PDF report
          </button>
        </div>
      </div>

      <div className="flex justify-end">
        <button className="btn-primary" onClick={onComplete}>
          Continue → Predict
        </button>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }) {
  const colors = {
    emerald: "text-emerald-600 dark:text-emerald-400",
    sky: "text-sky-600 dark:text-sky-400",
    amber: "text-amber-600 dark:text-amber-400",
    rose: "text-rose-600 dark:text-rose-400",
    violet: "text-brand-600 dark:text-brand-400",
  };
  return (
    <div className="card p-4 text-center">
      <div className={`text-3xl font-extrabold ${colors[color]}`}>{(value * 100).toFixed(1)}%</div>
      <div className="mt-1 text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}

function downloadBlobFromImage(dataUri, filename) {
  const a = document.createElement("a");
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
