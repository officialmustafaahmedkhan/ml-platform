export function Alert({ kind = "error", title, children, onClose }) {
  const styles = {
    error: "border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200",
    success: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200",
    info: "border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200",
    warning: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200",
  };
  return (
    <div className={`flex items-start justify-between gap-3 rounded-xl border p-4 text-sm ${styles[kind]}`}>
      <div>
        {title && <div className="font-semibold">{title}</div>}
        {children}
      </div>
      {onClose && (
        <button onClick={onClose} className="shrink-0 text-current opacity-60 hover:opacity-100">
          ✕
        </button>
      )}
    </div>
  );
}

export function ErrorBox({ error, onClose }) {
  if (!error) return null;
  const raw = error?.response?.data?.detail;
  let msg = error?.message || "Something went wrong";
  if (typeof raw === "string") msg = raw;
  else if (Array.isArray(raw)) msg = raw.map((e) => e.msg || JSON.stringify(e)).filter(Boolean).join("; ");
  return (
    <Alert kind="error" onClose={onClose}>
      {msg}
    </Alert>
  );
}
