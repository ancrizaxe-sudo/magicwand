import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

export const api = axios.create({
  baseURL: API,
  timeout: 120000,
});

export async function runAnalyze(payload) {
  const { data } = await api.post("/analyze", payload);
  return data;
}
export async function getAblation() {
  const { data } = await api.get("/ablation");
  return data;
}
export async function getWeightsInfo() {
  const { data } = await api.get("/weights/info");
  return data;
}
export async function getBlockchain(limit = 25) {
  const { data } = await api.get(`/blockchain?limit=${limit}`);
  return data;
}
export async function getJobs() {
  const { data } = await api.get("/jobs");
  return data;
}
