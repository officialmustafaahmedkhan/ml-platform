import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Workflow from "./pages/Workflow";
import Explore from "./pages/Explore";
import Experiments from "./pages/Experiments";
import Assistant from "./pages/Assistant";
import Compare from "./pages/Compare";
import History from "./pages/History";
import Commands from "./pages/Commands";
import Pipeline from "./pages/Pipeline";

function Protected({ children }) {
  const { token } = useAuth();
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  const { token } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={token ? <Navigate to="/dashboard" replace /> : <Login />} />
      <Route
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/workflow" element={<Workflow />} />
        <Route path="/explore" element={<Explore />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/assistant" element={<Assistant />} />
        <Route path="/commands" element={<Commands />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/history" element={<History />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
