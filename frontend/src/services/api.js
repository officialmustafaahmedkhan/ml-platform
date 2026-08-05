import client from "../api/client";

// --------------------------------------------------------------------------
// Auth
// --------------------------------------------------------------------------
export const apiAuth = {
  register: (payload) => client.post("/api/auth/register", payload).then((r) => r.data),
  login: (payload) => client.post("/api/auth/login", payload).then((r) => r.data),
  sendOtp: (email) => client.post("/api/auth/send-otp", { email }).then((r) => r.data),
  verifyOtp: (email, otp) => client.post("/api/auth/verify-otp", { email, otp }).then((r) => r.data),
  me: () => client.get("/api/auth/me").then((r) => r.data),
};

// --------------------------------------------------------------------------
// Datasets
// --------------------------------------------------------------------------
export const apiDatasets = {
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return client.post("/api/datasets/upload", form).then((r) => r.data);
  },
  list: () => client.get("/api/datasets").then((r) => r.data),
  get: (id) => client.get(`/api/datasets/${id}`).then((r) => r.data),
  head: (id, n = 20) => client.get(`/api/datasets/${id}/head`, { params: { n } }).then((r) => r.data),
  eda: (id) => client.get(`/api/datasets/${id}/eda`).then((r) => r.data),
  remove: (id) => client.delete(`/api/datasets/${id}`),
};

// --------------------------------------------------------------------------
// LLM labeling
// --------------------------------------------------------------------------
export const apiLlm = {
  status: () => client.get("/api/llm/status").then((r) => r.data),
  label: (datasetId, payload) =>
    client.post(`/api/datasets/${datasetId}/llm-label`, payload).then((r) => r.data),
};

// --------------------------------------------------------------------------
// AI Assistant
// --------------------------------------------------------------------------
export const apiAssistant = {
  chat: (payload) => client.post("/api/assistant/chat", payload).then((r) => r.data),
};

// --------------------------------------------------------------------------
// ML Commands (natural language)
// --------------------------------------------------------------------------
export const apiCommands = {
  run: (text) => client.post("/api/commands", { text }).then((r) => r.data),
  help: () => client.get("/api/commands/help").then((r) => r.data),
};

// --------------------------------------------------------------------------
// Pipeline visualization
// --------------------------------------------------------------------------
export const apiPipeline = {
  models: () => client.get("/api/pipeline/models").then((r) => r.data),
  detail: (modelId) => client.get(`/api/pipeline/${modelId}`).then((r) => r.data),
};

// --------------------------------------------------------------------------
// Preprocessing
// --------------------------------------------------------------------------
export const apiPreprocess = {
  profile: (id, target) =>
    client.get(`/api/preprocess/profile/${id}`, { params: { target } }).then((r) => r.data),
  autoConfig: (id, { target, model_hint } = {}) =>
    client.post(`/api/preprocess/auto-config/${id}`, null, { params: { target, model_hint } }).then((r) => r.data),
  run: (payload) => client.post("/api/preprocess", payload).then((r) => r.data),
};

// --------------------------------------------------------------------------
// Training
// --------------------------------------------------------------------------
export const apiTrain = {
  registry: () => client.get("/api/train/registry").then((r) => r.data),
  train: (payload) => client.post("/api/train", payload).then((r) => r.data),
  models: () => client.get("/api/train/models").then((r) => r.data),
  model: (id) => client.get(`/api/train/models/${id}`).then((r) => r.data),
  remove: (id) => client.delete(`/api/train/models/${id}`),
};

// --------------------------------------------------------------------------
// Evaluation / comparison
// --------------------------------------------------------------------------
export const apiEvaluate = {
  get: (modelId) => client.get(`/api/evaluate/${modelId}`).then((r) => r.data),
};

export const apiCompare = {
  run: (payload) => client.post("/api/compare", payload).then((r) => r.data),
};

// --------------------------------------------------------------------------
// Prediction
// --------------------------------------------------------------------------
export const apiPredict = {
  single: (modelId, input) => client.post("/api/predict", { model_id: modelId, input }).then((r) => r.data),
  batch: (modelId, file) => {
    const form = new FormData();
    form.append("file", file);
    return client
      .post("/api/predict/batch", form, { params: { model_id: modelId } })
      .then((r) => r.data);
  },
};

// --------------------------------------------------------------------------
// Explainability (LIME-style local + permutation global)
// --------------------------------------------------------------------------
export const apiExplain = {
  local: (modelId, input) =>
    client.post(`/api/explain/local?model_id=${modelId}`, { model_id: modelId, input }).then((r) => r.data),
  global: (modelId) => client.get(`/api/explain/global/${modelId}`).then((r) => r.data),
};

// --------------------------------------------------------------------------
// Recommendation / dashboard
// --------------------------------------------------------------------------
export const apiRecommend = {
  forDataset: (id, { target, model_hint } = {}) =>
    client.get(`/api/recommend/${id}`, { params: { target, model_hint } }).then((r) => r.data),
};

export const apiDashboard = {
  get: () => client.get("/api/dashboard").then((r) => r.data),
};

// --------------------------------------------------------------------------
// Experiment workspace (activity log + notes)
// --------------------------------------------------------------------------
export const apiExperiments = {
  list: (params = {}) => client.get("/api/experiments", { params }).then((r) => r.data),
  updateNotes: (id, notes) => client.patch(`/api/experiments/${id}`, { notes }).then((r) => r.data),
};

// --------------------------------------------------------------------------
// Export / download helpers (blob-based so auth headers are attached)
// --------------------------------------------------------------------------
export const apiExport = {
  model: async (modelId) => {
    const res = await client.get(`/api/export/model/${modelId}`, { responseType: "blob" });
    return res.data;
  },
  report: async (modelId) => {
    const res = await client.get(`/api/export/report/${modelId}`, { responseType: "blob" });
    return res.data;
  },
  reportPdf: async (modelId) => {
    const res = await client.get(`/api/export/report/${modelId}/pdf`, { responseType: "blob" });
    return res.data;
  },
  batch: async (filename) => {
    const res = await client.get(`/api/export/batch/${filename}`, { responseType: "blob" });
    return res.data;
  },
};

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
