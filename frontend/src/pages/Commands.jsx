import { useEffect, useRef, useState } from "react";
import { apiCommands } from "../services/api";

const SUGGESTIONS = [
  "list my datasets",
  "list models",
  "profile dataset 1",
  "eda on dataset 1",
  "train a random forest on dataset 1 with target Outcome",
  "explain model 1",
  "label dataset 1 with 4 categories as Priority",
];

export default function Commands() {
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const run = async (cmd) => {
    const value = (cmd ?? text).trim();
    if (!value || running) return;
    setRunning(true);
    setText("");
    try {
      const res = await apiCommands.run(value);
      setHistory((h) => [...h, { text: value, res }]);
    } catch {
      setHistory((h) => [
        ...h,
        { text: value, res: { action: "unknown", summary: "Command failed — is the backend running?", cards: [], rows: null, text: null } },
      ]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-slate-50">ML Commands</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Run machine-learning tasks from plain English — no form-filling required.
        </p>
      </div>

      {/* Command input */}
      <div className="card p-4">
        <div className="flex items-center gap-3">
          <span className="select-none text-lg font-bold text-brand-600 dark:text-brand-400">›</span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Try: train a random forest on dataset 1 with target Outcome"
            className="flex-1 bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400 dark:text-slate-100"
          />
          <button className="btn-primary shrink-0" onClick={() => run()} disabled={running || !text.trim()}>
            {running ? "Running…" : "Run"}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => run(s)}
              className="rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-brand-400 hover:text-brand-600 dark:border-slate-700 dark:text-slate-300"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* History */}
      <div className="space-y-4">
        {history.length === 0 && (
          <div className="card border-dashed p-8 text-center text-sm text-slate-400">
            Your executed commands will appear here.
          </div>
        )}
        {history.map((item, i) => (
          <div key={i} className="card p-5">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-100 pb-2 text-xs font-semibold text-slate-400 dark:border-slate-800">
              <span className="text-brand-500">›</span>
              <span className="truncate text-slate-700 dark:text-slate-200">{item.text}</span>
              <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500 dark:bg-slate-800">
                {item.res.action}
              </span>
            </div>

            <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{item.res.summary}</p>

            {item.res.cards?.length > 0 && (
              <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-5">
                {item.res.cards.map((c, j) => (
                  <div key={j} className="rounded-xl border border-slate-200 p-3 text-center dark:border-slate-700">
                    <div className="text-lg font-extrabold text-brand-600 dark:text-brand-400">{c.value}</div>
                    <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">{c.label}</div>
                  </div>
                ))}
              </div>
            )}

            {item.res.rows?.length > 0 && (
              <div className="mt-3 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 text-slate-500 dark:bg-slate-800 dark:text-slate-300">
                    <tr>
                      {Object.keys(item.res.rows[0]).map((k) => (
                        <th key={k} className="px-3 py-2 font-semibold">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                    {item.res.rows.map((r, j) => (
                      <tr key={j} className="text-slate-700 dark:text-slate-200">
                        {Object.values(r).map((v, k) => (
                          <td key={k} className="px-3 py-1.5">{String(v)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {item.res.text && (
              <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-50 p-3 text-xs leading-relaxed text-slate-600 dark:bg-slate-900 dark:text-slate-300">
                {item.res.text}
              </pre>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
