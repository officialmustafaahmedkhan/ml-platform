import { useEffect, useMemo, useState } from "react";
import { apiDatasets } from "../services/api";
import { ErrorBox } from "../components/Alert";
import Spinner from "../components/Spinner";

const LEVEL_STYLE = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200",
  warn: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200",
  danger: "border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200",
};

const LEVEL_ICON = {
  good: "●",
  warn: "▲",
  danger: "✕",
};

function HealthRing({ value }) {
  const v = Math.max(0, Math.min(100, value || 0));
  const r = 42;
  const circ = 2 * Math.PI * r;
  const color = v >= 80 ? "text-emerald-500" : v >= 50 ? "text-amber-500" : "text-rose-500";
  return (
    <div className="relative h-32 w-32">
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r={r} fill="none" strokeWidth="10" className="stroke-slate-200 dark:stroke-slate-700" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(v / 100) * circ} ${circ}`}
          className={color}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-extrabold text-slate-900 dark:text-white">{v.toFixed(0)}</span>
        <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400">/ 100</span>
      </div>
    </div>
  );
}

function Histogram({ data }) {
  if (!data || !data.counts?.length) return <div className="text-xs text-slate-400">No data</div>;
  const max = Math.max(...data.counts);
  return (
    <div className="flex h-32 items-end gap-0.5">
      {data.counts.map((c, i) => (
        <div
          key={i}
          title={`${data.labels?.[i] ?? ""} → ${c}`}
          className="min-w-[2px] flex-1 rounded-t-sm bg-brand-400/80 dark:bg-brand-500/60"
          style={{ height: `${max ? Math.max(4, (c / max) * 100) : 4}%` }}
        />
      ))}
    </div>
  );
}

function Heatmap({ features, matrix }) {
  if (!features?.length || !matrix?.length) return <div className="text-xs text-slate-400">Not enough numeric columns.</div>;
  const cell = (v) => {
    if (v == null) return "rgb(148,163,184)";
    const a = Math.min(1, Math.abs(v));
    return v >= 0 ? `rgba(14,165,233,${0.15 + a * 0.85})` : `rgba(244,63,94,${0.15 + a * 0.85})`;
  };
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1">
        <thead>
          <tr>
            <th />
            {features.map((f) => (
              <th key={f} className="max-w-16 truncate pb-1 text-[10px] font-medium text-slate-400">
                {f}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((f, i) => (
            <tr key={f}>
              <td className="max-w-16 truncate pr-1 text-right text-[10px] font-medium text-slate-400">{f}</td>
              {matrix[i]?.map((v, j) => (
                <td
                  key={j}
                  title={`${features[i]} × ${features[j]} = ${v ?? "n/a"}`}
                  className="h-7 w-7 rounded text-center text-[10px] font-semibold text-white"
                  style={{ background: cell(v) }}
                >
                  {v == null ? "" : v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Scatter({ rows, x, y }) {
  const points = useMemo(() => {
    return rows
      .map((r) => [Number(r[x]), Number(r[y])])
      .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
  }, [rows, x, y]);
  if (points.length < 2) return <div className="text-xs text-slate-400">Not enough numeric rows for a scatter plot.</div>;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const W = 420, H = 260, PAD = 34;
  const px = (v) => PAD + ((v - xMin) / Math.max(xMax - xMin, 1e-9)) * (W - PAD * 2);
  const py = (v) => H - PAD - ((v - yMin) / Math.max(yMax - yMin, 1e-9)) * (H - PAD * 2);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.25, 0.5, 0.75].map((t) => (
        <g key={t} className="text-slate-300 dark:text-slate-700">
          <line x1={PAD} x2={W - PAD} y1={py(yMin + (yMax - yMin) * t)} y2={py(yMin + (yMax - yMin) * t)} stroke="currentColor" strokeDasharray="3 3" />
          <line x1={px(xMin + (xMax - xMin) * t)} x2={px(xMin + (xMax - xMin) * t)} y1={PAD} y2={H - PAD} stroke="currentColor" strokeDasharray="3 3" />
        </g>
      ))}
      {points.slice(0, 800).map((p, i) => (
        <circle key={i} cx={px(p[0])} cy={py(p[1])} r="2.5" className="fill-brand-500/70 dark:fill-brand-400/70" />
      ))}
      <text x={W / 2} y={H - 6} textAnchor="middle" className="fill-slate-500 text-[10px]">{x}</text>
      <text x={10} y={H / 2} textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`} className="fill-slate-500 text-[10px]">{y}</text>
    </svg>
  );
}

function BarList({ items }) {
  const max = Math.max(...items.map((i) => i.count), 1);
  return (
    <div className="space-y-1.5">
      {items.map((i) => (
        <div key={i.value} className="flex items-center gap-2 text-xs">
          <span className="w-28 truncate text-right text-slate-500 dark:text-slate-400">{i.value}</span>
          <div className="h-4 flex-1 overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
            <div className="h-full rounded bg-brand-400/80 dark:bg-brand-500/60" style={{ width: `${(i.count / max) * 100}%` }} />
          </div>
          <span className="w-10 text-slate-500 dark:text-slate-400">{i.count}</span>
        </div>
      ))}
    </div>
  );
}

const statCard = "rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900";

export default function Explore() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [dataset, setDataset] = useState(null);
  const [eda, setEda] = useState(null);
  const [head, setHead] = useState(null);
  const [scatterX, setScatterX] = useState("");
  const [scatterY, setScatterY] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiDatasets.list().then(setDatasets).catch(setError);
  }, []);

  useEffect(() => {
    if (!datasetId) {
      setDataset(null);
      setEda(null);
      setHead(null);
      return;
    }
    setLoading(true);
    setError(null);
    const id = Number(datasetId);
    Promise.all([apiDatasets.get(id), apiDatasets.eda(id), apiDatasets.head(id, 500)])
      .then(([d, e, h]) => {
        setDataset(d);
        setEda(e);
        setHead(h);
        const pair = e?.scatter_pairs?.[0];
        setScatterX(pair?.x || "");
        setScatterY(pair?.y || "");
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, [datasetId]);

  const profile = dataset?.profile || {};
  const health = profile.health_score;
  const insights = profile.insights || [];
  const numericStats = useMemo(
    () => (eda ? Object.entries(eda.column_stats || {}).filter(([, s]) => s.kind === "numeric") : []),
    [eda]
  );
  const categoricalStats = useMemo(
    () => (eda ? Object.entries(eda.column_stats || {}).filter(([, s]) => s.kind === "categorical") : []),
    [eda]
  );
  const targetCounts = Object.entries(eda?.class_counts || {}).map(([value, count]) => ({ value, count }));

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">Data Explorer</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Dataset health score, automated insights, and interactive EDA.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="label">Dataset</label>
          <select className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">Select a dataset…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} · {d.rows} rows
              </option>
            ))}
          </select>
        </div>
      </div>

      <ErrorBox error={error} onClose={() => setError(null)} />
      {loading && <Spinner label="Analyzing dataset…" />}

      {datasetId && !loading && dataset && (
        <>
          {/* Health score + insights */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className={`${statCard} flex flex-col items-center justify-center`}>
              <HealthRing value={health} />
              <div className="mt-2 text-center text-sm font-semibold text-slate-700 dark:text-slate-200">
                {health >= 80 ? "Healthy dataset" : health >= 50 ? "Needs attention" : "High risk"}
              </div>
            </div>
            <div className={`${statCard} lg:col-span-2`}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Automated insights
              </h2>
              <div className="space-y-2">
                {insights.map((ins, i) => (
                  <div key={i} className={`flex items-start gap-2 rounded-xl border p-3 text-xs ${LEVEL_STYLE[ins.level] || LEVEL_STYLE.warn}`}>
                    <span className="mt-0.5 shrink-0">{LEVEL_ICON[ins.level] || "•"}</span>
                    <div>
                      <span className="font-semibold">{ins.title}: </span>
                      {ins.detail}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Overview stats */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {[
              { label: "Rows", value: profile.rows ?? dataset.rows },
              { label: "Columns", value: profile.columns ?? dataset.columns?.length },
              { label: "Missing", value: profile.missing_pct != null ? `${profile.missing_pct}%` : "—" },
              { label: "Duplicates", value: profile.duplicate_pct != null ? `${profile.duplicate_pct}%` : "—" },
            ].map((s) => (
              <div key={s.label} className={statCard}>
                <div className="text-xs font-medium uppercase tracking-wide text-slate-400">{s.label}</div>
                <div className="mt-1 text-xl font-extrabold text-slate-900 dark:text-white">{s.value}</div>
              </div>
            ))}
          </div>

          {/* Target distribution */}
          {targetCounts.length > 0 && (
            <div className={statCard}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Target distribution{profile.target_column ? ` — ${profile.target_column}` : ""}
              </h2>
              <BarList items={targetCounts} />
            </div>
          )}

          {/* Correlation heatmap */}
          <div className={statCard}>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Correlation matrix
            </h2>
            <Heatmap features={eda?.correlation_features} matrix={eda?.correlation_matrix} />
          </div>

          {/* Scatter explorer */}
          {eda?.scatter_pairs?.length > 0 && (
            <div className={statCard}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Scatter explorer
              </h2>
              <div className="mb-3 flex flex-wrap gap-3">
                <div>
                  <label className="label">X axis</label>
                  <select className="input" value={scatterX} onChange={(e) => setScatterX(e.target.value)}>
                    {eda.numeric_columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label">Y axis</label>
                  <select className="input" value={scatterY} onChange={(e) => setScatterY(e.target.value)}>
                    {eda.numeric_columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end text-xs text-slate-400">
                  Top correlated pairs:{" "}
                  {eda.scatter_pairs.map((p) => (
                    <button
                      key={`${p.x}-${p.y}`}
                      className="ml-2 rounded-full border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:border-brand-400 hover:bg-brand-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-brand-500/10"
                      onClick={() => {
                        setScatterX(p.x);
                        setScatterY(p.y);
                      }}
                    >
                      {p.x}×{p.y} ({p.corr})
                    </button>
                  ))}
                </div>
              </div>
              {scatterX && scatterY ? (
                <Scatter rows={head?.preview || []} x={scatterX} y={scatterY} />
              ) : (
                <div className="text-xs text-slate-400">Pick two numeric columns to plot.</div>
              )}
            </div>
          )}

          {/* Numeric column stats */}
          <div className={statCard}>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Numeric columns
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-400 dark:border-slate-700">
                    <th className="py-1.5 pr-3">Column</th>
                    <th className="py-1.5 pr-3">Missing</th>
                    <th className="py-1.5 pr-3">Mean</th>
                    <th className="py-1.5 pr-3">Std</th>
                    <th className="py-1.5 pr-3">Min</th>
                    <th className="py-1.5 pr-3">Median</th>
                    <th className="py-1.5 pr-3">Max</th>
                    <th className="py-1.5 pr-3">Skew</th>
                    <th className="py-1.5">Outliers</th>
                  </tr>
                </thead>
                <tbody>
                  {numericStats.map(([name, s]) => (
                    <tr key={name} className="border-b border-slate-100 text-slate-700 dark:border-slate-800 dark:text-slate-200">
                      <td className="py-1.5 pr-3 font-semibold">{name}</td>
                      <td className="py-1.5 pr-3">{s.missing}</td>
                      <td className="py-1.5 pr-3">{s.mean ?? "—"}</td>
                      <td className="py-1.5 pr-3">{s.std ?? "—"}</td>
                      <td className="py-1.5 pr-3">{s.min ?? "—"}</td>
                      <td className="py-1.5 pr-3">{s.median ?? "—"}</td>
                      <td className="py-1.5 pr-3">{s.max ?? "—"}</td>
                      <td className="py-1.5 pr-3">{s.skew ?? "—"}</td>
                      <td className="py-1.5">{s.outliers ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Histograms */}
          {numericStats.length > 0 && (
            <div className={statCard}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Distributions
              </h2>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {numericStats.map(([name]) => (
                  <div key={name} className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
                    <div className="mb-2 truncate text-xs font-semibold text-slate-700 dark:text-slate-200">{name}</div>
                    <Histogram data={eda?.histograms?.[name]} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Categorical columns */}
          {categoricalStats.length > 0 && (
            <div className={statCard}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Categorical columns
              </h2>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {categoricalStats.map(([name, s]) => (
                  <div key={name} className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold text-slate-700 dark:text-slate-200">{name}</span>
                      <span className="shrink-0 text-[10px] text-slate-400">
                        {s.nunique} unique · {s.missing} missing
                      </span>
                    </div>
                    <BarList items={(s.categories || []).map((c) => ({ value: c.value, count: c.count }))} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Preview */}
          {head?.preview?.length > 0 && (
            <div className={statCard}>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Data preview (first {head.preview.length} rows)
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-400 dark:border-slate-700">
                      {(head.columns || []).map((c) => (
                        <th key={c} className="max-w-40 truncate py-1.5 pr-3">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {head.preview.slice(0, 8).map((r, i) => (
                      <tr key={i} className="border-b border-slate-100 text-slate-700 dark:border-slate-800 dark:text-slate-200">
                        {(head.columns || []).map((c) => (
                          <td key={c} className="max-w-40 truncate py-1.5 pr-3">{r[c] ?? ""}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {datasetId && !loading && !dataset && (
        <div className="card p-6 text-center text-sm text-slate-400">Dataset not found or no longer available.</div>
      )}
    </div>
  );
}
