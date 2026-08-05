import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import Logo from "./Logo";

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: <IconGrid /> },
  { to: "/workflow", label: "ML Workflow", icon: <IconFlow /> },
  { to: "/explore", label: "Data Explorer", icon: <IconChart /> },
  { to: "/assistant", label: "AI Assistant", icon: <IconSpark /> },
  { to: "/commands", label: "ML Commands", icon: <IconCommand /> },
  { to: "/pipeline", label: "Pipeline", icon: <IconFlow /> },
  { to: "/compare", label: "Compare Models", icon: <IconCompare /> },
  { to: "/experiments", label: "Experiments", icon: <IconFlask /> },
  { to: "/history", label: "History", icon: <IconHistory /> },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const { dark, toggle } = useTheme();
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 flex-col border-r border-slate-200 bg-white px-4 py-6 dark:border-slate-800 dark:bg-slate-900 md:flex">
        <Logo />
        <nav className="mt-8 flex flex-1 flex-col gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition ${
                  isActive
                    ? "bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-800">
          <div className="flex items-center gap-3 px-1">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-100 text-sm font-bold text-brand-700 dark:bg-brand-500/20 dark:text-brand-300">
              {(user?.name || "U").charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold text-slate-800 dark:text-slate-100">
                {user?.name}
              </div>
              <div className="truncate text-xs text-slate-400">{user?.email}</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex min-h-screen flex-1 flex-col md:pl-64">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/80 px-4 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80 md:px-8">
          <div className="flex items-center gap-3 md:hidden">
            <Logo size={30} />
          </div>
          <div className="hidden text-sm text-slate-500 dark:text-slate-400 md:block">
            User-Personalized Intelligent Machine Learning Platform
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggle}
              title="Toggle dark/light mode"
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              {dark ? <IconSun /> : <IconMoon />}
            </button>
            <button
              onClick={() => {
                logout();
                navigate("/login");
              }}
              className="flex h-9 items-center gap-2 rounded-xl border border-slate-200 px-3 text-sm font-medium text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              Logout
            </button>
          </div>
        </header>
        <main className="flex-1 p-4 md:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

// --- Icons -----------------------------------------------------------------
function IconGrid() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  );
}
function IconFlow() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <circle cx="5" cy="6" r="2.5" />
      <circle cx="19" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="M7.5 6 H16.5 M13.8 15.8 L16.5 8.5 M10.2 15.8 L7.5 8.5" />
    </svg>
  );
}
function IconChart() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M4 20V10M10 20V4M16 20v-7M21 20H3" />
    </svg>
  );
}
function IconCompare() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M8 3v18M16 3v18M4 8h4M16 16h4" />
    </svg>
  );
}
function IconFlask() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M9 3h6M10 3v5.5L4.5 19a2 2 0 001.8 3h11.4a2 2 0 001.8-3L14 8.5V3M7.5 14h9" />
    </svg>
  );
}
function IconHistory() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M12 8v4l3 3M12 21a9 9 0 100-18 9 9 0 000 18z" />
    </svg>
  );
}
function IconSpark() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" />
    </svg>
  );
}
function IconCommand() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M5 4h14M7 4v4c0 2 1.5 3 3 3h4" transform="rotate(90 12 12)" />
      <path d="M17 20V8M15 6l2-2 2 2" transform="rotate(90 12 12)" />
      <path d="M5 4v4c0 2 1.5 3 3 3M17 4v4c0 2-1.5 3-3 3" />
      <path d="M5 16v4M7 16l-2 2 2 2M17 16v4M19 16l-2 2 2 2" />
    </svg>
  );
}
function IconMoon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />
    </svg>
  );
}
function IconSun() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}
