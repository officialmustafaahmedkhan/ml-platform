import { useEffect, useRef, useState } from "react";
import { apiAssistant, apiDatasets, apiLlm, apiTrain } from "../services/api";
import { ErrorBox } from "../components/Alert";
import Spinner from "../components/Spinner";

export default function Assistant() {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [modelId, setModelId] = useState("");
  const [llmStatus, setLlmStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    apiDatasets.list().then(setDatasets).catch(setError);
    apiTrain.models().then(setModels).catch(() => setModels([]));
    apiLlm.status().then(setLlmStatus).catch(() => setLlmStatus(null));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const canAsk = !!datasetId && !busy && llmStatus?.enabled;
  const active = !datasetId;

  const send = async () => {
    const text = input.trim();
    if (!text || !datasetId || busy) return;
    const userMsg = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const history = messages.map(({ role, content }) => ({ role, content }));
      const res = await apiAssistant.chat({
        message: text,
        dataset_id: datasetId ? Number(datasetId) : undefined,
        model_id: modelId ? Number(modelId) : undefined,
        history,
      });
      setMessages((m) => [...m, { role: "assistant", content: res.reply }]);
    } catch (e) {
      setError(e);
      setMessages((m) => m.slice(0, -1));
    } finally {
      setBusy(false);
    }
  };

  const suggestions = [
    "Summarize this dataset",
    "What are the most important features?",
    "Any data quality issues?",
    "Which model should I try first?",
  ];

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">AI Assistant</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Ask questions about your datasets and trained models in plain language.
        </p>
      </div>

      {llmStatus && !llmStatus.enabled && (
        <div className="card p-4 text-sm text-amber-700 dark:text-amber-300">
          The LLM is disabled. Set <code className="font-mono">LLM_PROVIDER=openai</code> (or{" "}
          <code className="font-mono">ollama</code>) in the backend{" "}
          <code className="font-mono">.env</code> and restart the server.
        </div>
      )}

      {/* Context selectors */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="label">Dataset to talk about</label>
          <select className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">Select a dataset…</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} · {d.rows} rows
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Trained model (optional)</label>
          <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">None — dataset only</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.model_type}) · {(m.metrics?.accuracy * 100 || 0).toFixed(1)}%
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Chat window */}
      <div className="card flex h-[28rem] flex-col p-0">
        {active ? (
          <div className="flex flex-1 items-center justify-center p-8 text-center text-sm text-slate-400">
            Select a dataset above to start chatting about it.
          </div>
        ) : (
          <div className="flex-1 space-y-4 overflow-y-auto p-5">
            {messages.length === 0 && (
              <div className="space-y-2">
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Ask me anything about this dataset
                  {modelId ? " and model" : ""}. Try:
                </p>
                <div className="flex flex-wrap gap-2">
                  {suggestions.map((s) => (
                    <button
                      key={s}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-brand-400 hover:bg-brand-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-brand-500/10"
                      onClick={() => {
                        setInput(s);
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm ${
                    m.role === "user"
                      ? "bg-brand-500 text-white"
                      : "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-100"
                  }`}
                >
                  {m.content}
                </div>
              </div>
            ))}
            {busy && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-slate-100 px-4 py-3 dark:bg-slate-800">
                  <Spinner label="Thinking…" />
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}

        <div className="border-t border-slate-200 p-4 dark:border-slate-800">
          <ErrorBox error={error} onClose={() => setError(null)} />
          <div className="flex gap-2">
            <input
              className="input flex-1"
              placeholder="Ask about the dataset…"
              value={input}
              disabled={!datasetId || busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
            />
            <button className="btn-primary" disabled={!canAsk} onClick={send}>
              {busy ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
