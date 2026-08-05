import { useState } from "react";
import { Alert } from "./Alert";

export default function ImageCard({ src, title, alt, downloadName, onDownload }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <div className="card overflow-hidden p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">{title}</h3>
        {onDownload && (
          <button onClick={onDownload} className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-400">
            Download
          </button>
        )}
      </div>
      {src ? (
        <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-slate-50 p-2 dark:bg-slate-950">
          {!loaded && (
            <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-400">
              Rendering chart…
            </div>
          )}
          <img
            src={src}
            alt={alt || title}
            onLoad={() => setLoaded(true)}
            className={`max-w-full transition-opacity ${loaded ? "opacity-100" : "opacity-0"}`}
          />
        </div>
      ) : (
        <div className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-400 dark:bg-slate-950">
          Not available for this model type.
        </div>
      )}
    </div>
  );
}

export { Alert };
