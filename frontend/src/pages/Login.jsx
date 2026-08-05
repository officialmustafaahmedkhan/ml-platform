import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { apiAuth } from "../services/api";
import { Alert, ErrorBox } from "../components/Alert";
import Logo from "../components/Logo";
import { useTheme } from "../context/ThemeContext";

const FEATURES = [
  { title: "Multi-Model Training", text: "Decision Tree, KNN, Random Forest + hybrid ensembles" },
  { title: "Automated Preprocessing", text: "Missing values, encoding, scaling and SMOTE in one click" },
  { title: "Smart Recommendations", text: "The engine suggests the best model for your data" },
  { title: "Personal Dashboards", text: "Experiment history, accuracy trends and insights" },
];

const FLOATING_GLYPHS = [
  { top: "12%", left: "6%", size: 34, delay: "0s", opacity: 0.35, speed: "float" },
  { top: "24%", left: "86%", size: 26, delay: "-2s", opacity: 0.3, speed: "float-slow" },
  { top: "62%", left: "8%", size: 22, delay: "-4s", opacity: 0.25, speed: "float-slower" },
  { top: "78%", left: "82%", size: 30, delay: "-1s", opacity: 0.3, speed: "float-slow" },
  { top: "44%", left: "78%", size: 18, delay: "-3s", opacity: 0.35, speed: "float" },
];

function Glyph({ top, left, size, delay, opacity, speed }) {
  return (
    <svg
      className={`pointer-events-none absolute animate-${speed}`}
      style={{ top, left, width: size, height: size, opacity, animationDelay: delay }}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path d="M5 3l-3 9 3 9M19 3l3 9-3 9" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="3.5" />
      <path d="M9.5 8.5L4 4M14.5 8.5L20 4M9.5 15.5L4 20M14.5 15.5L20 20" opacity="0.6" />
    </svg>
  );
}

export default function Login() {
  const [mode, setMode] = useState("login"); // "login" | "register" | "otp"
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [otpEmail, setOtpEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [devOtp, setDevOtp] = useState("");
  const [otpMessage, setOtpMessage] = useState("");
  const [error, setError] = useState(null);
  const { login, register, verifyOtp, loading } = useAuth();
  const { dark, toggle } = useTheme();
  const navigate = useNavigate();

  const detail = (err) => {
    const d = err?.response?.data?.detail;
    return typeof d === "string" ? null : d;
  };

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      if (mode === "login") {
        await login(form.email, form.password);
        navigate("/dashboard");
      } else {
        const res = await register(form.name, form.email, form.password);
        if (res?.needs_verification) {
          setOtpEmail(res.email);
          setDevOtp(res.dev_otp || "");
          setOtp(res.dev_otp || "");
          setOtpMessage(res.message || "Enter the code we emailed you.");
          setMode("otp");
        } else {
          navigate("/dashboard");
        }
      }
    } catch (err) {
      if (mode === "login" && detail(err)?.code === "email_not_verified") {
        const d = detail(err);
        setOtpEmail(d.email);
        setDevOtp(d.dev_otp || "");
        setOtp(d.dev_otp || "");
        setOtpMessage(d.message || "Enter the verification code.");
        setMode("otp");
      } else {
        setError(err);
      }
    }
  };

  const submitOtp = async (e) => {
    e.preventDefault();
    if (!otpEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(otpEmail)) {
      setError({ message: "No valid email on file — go back and sign in again." });
      return;
    }
    if (!/^\d{4,6}$/.test(otp)) {
      setError({ message: "Enter the 4-6 digit code sent to your email." });
      return;
    }
    setError(null);
    try {
      await verifyOtp(otpEmail, otp);
      navigate("/dashboard");
    } catch (err) {
      setError(err);
    }
  };

  const resend = async () => {
    setError(null);
    try {
      const r = await apiAuth.sendOtp(otpEmail);
      if (r.dev_otp) {
        setDevOtp(r.dev_otp);
        setOtp(r.dev_otp);
      }
      setOtpMessage(r.message || "Verification code sent.");
    } catch (err) {
      setError(err);
    }
  };

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const otpInput = (code) => (e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6));

  return (
    <div className="flex min-h-screen">
      {/* Branding panel */}
      <div className="relative hidden w-1/2 flex-col justify-between overflow-hidden bg-brand-800 p-12 text-white lg:flex">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-700 via-brand-600 to-indigo-900 bg-[length:200%_200%] animate-gradient-x" />
        <div className="pointer-events-none absolute -top-24 -left-24 h-80 w-80 rounded-full bg-brand-400/40 blur-3xl animate-blob" />
        <div className="pointer-events-none absolute top-1/3 -right-24 h-96 w-96 rounded-full bg-fuchsia-500/25 blur-3xl animate-blob" style={{ animationDelay: "-5s" }} />
        <div className="pointer-events-none absolute -bottom-28 left-1/4 h-80 w-80 rounded-full bg-indigo-300/25 blur-3xl animate-blob" style={{ animationDelay: "-9s" }} />
        <div className="pointer-events-none absolute inset-0 text-white">
          {FLOATING_GLYPHS.map((g, i) => (
            <Glyph key={i} {...g} />
          ))}
        </div>

        <div className="relative z-10 animate-pop-in">
          <Logo />
        </div>

        <div className="relative z-10">
          <h1 className="text-4xl font-extrabold leading-tight animate-pop-in">
            User-Personalized Intelligent{" "}
            <span className="bg-gradient-to-r from-white via-brand-200 to-fuchsia-200 bg-clip-text text-transparent bg-[length:200%_auto] animate-gradient-x">
              Machine Learning
            </span>{" "}
            Platform
          </h1>
          <p className="mt-4 max-w-md text-brand-100 animate-pop-in" style={{ animationDelay: "0.1s" }}>
            Train, evaluate, compare and deploy machine learning models with
            hybrid ensemble learning — no code required.
          </p>
          <div className="mt-10 grid grid-cols-2 gap-4">
            {FEATURES.map((f, i) => (
              <div
                key={f.title}
                className="group rounded-2xl border border-white/10 bg-white/10 p-4 backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:bg-white/20 hover:shadow-xl animate-pop-in"
                style={{ animationDelay: `${0.15 + i * 0.12}s` }}
              >
                <div className="font-semibold transition-colors group-hover:text-white">{f.title}</div>
                <div className="mt-1 text-sm text-brand-100">{f.text}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="relative z-10 text-sm text-brand-200 animate-pop-in" style={{ animationDelay: "0.6s" }}>
          FastAPI · React · Scikit-learn
        </div>
      </div>

      {/* Form panel */}
      <div className="flex w-full items-center justify-center bg-slate-50 p-6 dark:bg-slate-950 lg:w-1/2">
        <button
          onClick={toggle}
          className="absolute top-4 right-4 flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 transition-transform hover:scale-110 hover:rotate-12 dark:border-slate-700 dark:text-slate-300"
          title="Toggle theme"
        >
          {dark ? "☀️" : "🌙"}
        </button>
        <div className="w-full max-w-sm animate-pop-in">
          <div className="mb-8 lg:hidden animate-pop-in">
            <Logo />
          </div>

          {mode === "otp" ? (
            <div key="otp" className="animate-pop-in">
              <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white">Verify your email</h2>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Enter the 6-digit code sent to <span className="font-semibold">{otpEmail}</span>
              </p>

              {devOtp && (
                <div className="mt-4 animate-pop-in">
                  <Alert kind="info">
                    <span className="font-semibold">Dev mode (no SMTP configured):</span> your code is{" "}
                    <span className="font-mono text-base font-bold">{devOtp}</span>
                  </Alert>
                </div>
              )}
              {otpMessage && !devOtp && (
                <div className="mt-4 animate-pop-in">
                  <Alert kind="info">{otpMessage}</Alert>
                </div>
              )}

              <form onSubmit={submitOtp} className="mt-6 space-y-4">
                <div>
                  <label className="label">Verification code</label>
                  <input
                    className="input font-mono text-center text-2xl tracking-[0.5em]"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    value={otp}
                    onChange={otpInput()}
                    placeholder="••••••"
                    maxLength={6}
                    minLength={4}
                    required
                  />
                </div>

                <ErrorBox error={error} onClose={() => setError(null)} />

                <button type="submit" disabled={loading || otp.length < 4} className="btn-primary w-full transition-transform hover:scale-[1.01] active:scale-[0.99]">
                  {loading ? "Verifying…" : "Verify & continue"}
                </button>
              </form>

              <div className="mt-4 flex items-center justify-between text-sm">
                <button onClick={resend} disabled={loading} className="font-semibold text-brand-600 transition hover:underline hover:underline-offset-2 dark:text-brand-400">
                  Resend code
                </button>
                <button onClick={() => setMode("login")} className="text-slate-500 transition hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200">
                  Back to login
                </button>
              </div>
            </div>
          ) : (
            <div key={mode} className="animate-pop-in">
              <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white">
                {mode === "login" ? "Welcome back" : "Create your account"}
              </h2>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                {mode === "login"
                  ? "Sign in to your personal ML workspace"
                  : "Start building models in minutes"}
              </p>

              <div className="relative mt-6 grid grid-cols-2 rounded-xl bg-slate-200/70 p-1 dark:bg-slate-800">
                <span
                  className={`absolute inset-y-1 w-[calc(50%-4px)] rounded-lg bg-white shadow transition-all duration-300 ease-out dark:bg-slate-900 ${
                    mode === "register" ? "left-[calc(50%+4px)]" : "left-1"
                  }`}
                />
                {["login", "register"].map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`relative rounded-lg py-2 text-sm font-semibold capitalize transition ${
                      mode === m
                        ? "text-slate-900 dark:text-white"
                        : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              <form onSubmit={submit} className="mt-6 space-y-4">
                {mode === "register" && (
                  <div className="animate-pop-in">
                    <label className="label">Full name</label>
                    <input className="input transition-all duration-200 hover:border-brand-300 dark:hover:border-brand-600" value={form.name} onChange={set("name")} placeholder="Ada Lovelace" required />
                  </div>
                )}
                <div>
                  <label className="label">Email</label>
                  <input className="input transition-all duration-200 hover:border-brand-300 dark:hover:border-brand-600" type="email" value={form.email} onChange={set("email")} placeholder="you@example.com" required />
                </div>
                <div>
                  <label className="label">Password</label>
                  <input className="input transition-all duration-200 hover:border-brand-300 dark:hover:border-brand-600" type="password" value={form.password} onChange={set("password")} placeholder="••••••••" required minLength={6} />
                </div>

                <ErrorBox error={error} onClose={() => setError(null)} />

                <button type="submit" disabled={loading} className="btn-primary w-full transition-transform duration-200 hover:scale-[1.01] hover:shadow-lg hover:shadow-brand-500/20 active:scale-[0.99]">
                  {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
