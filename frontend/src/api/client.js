import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  timeout: 120000,
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem("ml_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response && err.response.status === 401) {
      localStorage.removeItem("ml_token");
      localStorage.removeItem("ml_user");
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default client;
