import { createContext, useContext, useMemo, useState } from "react";
import { apiAuth } from "../services/api";

const AuthContext = createContext({
  user: null,
  token: null,
  loading: false,
  login: async () => {},
  register: async () => {},
  verifyOtp: async () => {},
  logout: () => {},
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("ml_user") || "null");
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem("ml_token"));
  const [loading, setLoading] = useState(false);

  const persist = (data) => {
    localStorage.setItem("ml_token", data.access_token);
    localStorage.setItem("ml_user", JSON.stringify(data.user));
    setToken(data.access_token);
    setUser(data.user);
  };

  const login = async (email, password) => {
    setLoading(true);
    try {
      const data = await apiAuth.login({ email, password });
      persist(data);
      return data.user;
    } finally {
      setLoading(false);
    }
  };

  const register = async (name, email, password) => {
    setLoading(true);
    try {
      // Register returns OTP info (no token yet) — the caller drives verification.
      return await apiAuth.register({ name, email, password });
    } finally {
      setLoading(false);
    }
  };

  const verifyOtp = async (email, otp) => {
    setLoading(true);
    try {
      const data = await apiAuth.verifyOtp(email, otp);
      persist(data);
      return data.user;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("ml_token");
    localStorage.removeItem("ml_user");
    setToken(null);
    setUser(null);
  };

  const value = useMemo(
    () => ({ user, token, loading, login, register, verifyOtp, logout }),
    [user, token, loading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
