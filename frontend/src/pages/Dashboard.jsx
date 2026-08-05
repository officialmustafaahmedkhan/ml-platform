import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiDashboard } from "../services/api";
import StatCard from "../components/StatCard";
import Spinner from "../components/Spinner";
import { ErrorBox } from "../components/Alert";
import { useAuth } from "../context/AuthContext";

const PIE_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#f43f5e", "#0ea5e9", "#8b5cf6"];

const ACTION_LABELS = {
  upload: "📁 Uploaded a dataset",
  preprocess: "🧹 Ran preprocessing",
  train: "🤖 Trained a model",
  predict: "🔮 Made a prediction",
  batch_predict: "📦 Batch predicted",
  compare: "⚖️ Compared models",
};

export default function Dashboard() {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiDashboard
      .get()
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center py-24">
        <Spinner label="Loading your dashboard…" />
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

  const s = data.stats;
  const pieData = Object.entries(data.model_type_distribution).map(([name, value]) => ({ name, value }));
  const trend = data.accuracy_trend.length ? data.accuracy_trend : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">
          Welcome back, {user?.name?.split(" ")[0] || "there"} 👋
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Here's what's happening in your machine learning workspace.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon="📁" label="Datasets" value={s.datasets} accent="brand" />
        <StatCard icon="🤖" label="Trained Models" value={s.models} accent="emerald" />
        <StatCard
          icon="🎯"
          label="Avg. Accuracy"
          value={s.avg_accuracy !== null ? `${(s.avg_accuracy * 100).toFixed(1)}%` : "—"}
          sub={s.models ? "across all models" : "train a model to start"}
          accent="amber"
        />
        <StatCard icon="⚡" label="Experiments" value={s.experiments} accent="rose" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="card p-5 xl:col-span-2">
          <h3 className="mb-4 text-sm font-bold text-slate-800 dark:text-slate-100">Accuracy Trend</h3>
          {trend ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={trend} margin={{ top: 5, right: 15, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
                <XAxis dataKey="index" tick={{ fontSize: 12 }} label={{ value: "Experiment", position: "insideBottom", offset: -2, fontSize: 12 }} />
                <YAxis
                  tick={{ fontSize: 12 }}
                  domain={[0, 100]}
                  tickFormatter={(v) => `${Math.round(v)}%`}
                />
                <Tooltip
                  formatter={(v) => [`${(v * 100).toFixed(1)}%`, "Accuracy"]}
                  contentStyle={{ borderRadius: 12 }}
                />
                <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState text="No experiments yet — head to the workflow to train your first model." />
          )}
        </div>

        <div className="card p-5">
          <h3 className="mb-4 text-sm font-bold text-slate-800 dark:text-slate-100">Models by Type</h3>
          {pieData.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState text="No models trained yet." />
          )}
        </div>
      </div>

      {/* Suggestions */}
      {data.suggestions?.length > 0 && (
        <div className="card p-5">
          <h3 className="mb-3 text-sm font-bold text-slate-800 dark:text-slate-100">💡 Suggested Improvements</h3>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {data.suggestions.map((sg, i) => (
              <div key={i} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
                <div className="flex items-center justify-between gap-2">
                  <span className="badge bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300">
                    {sg.category}
                  </span>
                  {sg.priority && (
                    <span className="badge bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                      {sg.priority}
                    </span>
                  )}
                </div>
                <div className="mt-2 text-sm font-semibold text-slate-800 dark:text-slate-100">
                  {sg.suggestion || sg.action}
                </div>
                <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sg.detail || sg.impact}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Lists row */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="card p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Recent Models</h3>
            <Link to="/history" className="text-xs font-semibold text-brand-600 hover:underline dark:text-brand-400">
              View all →
            </Link>
          </div>
          {data.recent_models.length ? (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {data.recent_models.map((m) => (
                <li key={m.id} className="flex items-center justify-between py-2.5">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-800 dark:text-slate-100">{m.name}</div>
                    <div className="text-xs text-slate-400">
                      {new Date(m.created_at).toLocaleString()}
                    </div>
                  </div>
                  {m.accuracy !== null && m.accuracy !== undefined ? (
                    <span className="badge bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
                      {(m.accuracy * 100).toFixed(1)}% acc
                    </span>
                  ) : (
                    <span className="badge bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                      no metric
                    </span>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState text="No models yet." />
          )}
        </div>

        <div className="card p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Recent Activity</h3>
          </div>
          {data.activity_timeline.length ? (
            <ul className="relative ml-3 border-l-2 border-slate-200 pl-5 dark:border-slate-800">
              {data.activity_timeline.map((e, i) => (
                <li key={i} className="relative pb-5 last:pb-0">
                  <span className="absolute top-1 -left-[27px] h-3 w-3 rounded-full border-2 border-white bg-brand-500 dark:border-slate-900" />
                  <div className="text-sm font-medium text-slate-800 dark:text-slate-100">
                    {ACTION_LABELS[e.action] || e.action}
                  </div>
                  <div className="text-xs text-slate-400">
                    {new Date(e.created_at).toLocaleString()}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState text="No activity yet — get started in the workflow." />
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div className="flex h-full min-h-[120px] items-center justify-center rounded-xl bg-slate-50 p-6 text-center text-sm text-slate-400 dark:bg-slate-950">
      {text}
    </div>
  );
}
