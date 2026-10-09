/**
 * Typed client for the SpectrumSense API.
 *
 * Every type here is re-exported from api.generated.ts, which
 * `npm run types` writes from the running API's OpenAPI document
 * (backend/app/schemas.py). A field renamed in the backend therefore fails
 * `npm run build` instead of appearing as a runtime undefined. Regenerate
 * after changing the schemas; the generated file is committed so a build
 * never depends on a running backend.
 *
 * The localhost fallback is right for development and a deployment trap: a
 * production build with NEXT_PUBLIC_API_URL unset will try to reach the
 * visitor's own machine.
 */
import type { components } from "./api.generated";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type S = components["schemas"];

export type HealthOut = S["HealthOut"];
export type ModelReport = S["ModelReport"];
export type ExploreControls = S["ExploreControls"];
export type NumberControl = S["NumberControlOut"];
export type ExploreResponse = S["ExploreResponse"];
export type Decision = S["Decision"];
export type SignalView = S["SignalView"];
export type AnalyzeResponse = S["AnalyzeResponse"];
export type WindowDecision = S["WindowDecision"];
export type WindowView = S["WindowView"];
export type PrepareSettings = S["PrepareSettingsIn"];
export type SpectrogramOut = S["SpectrogramOut"];
export type ExampleInfo = S["ExampleInfo"];
export type GroupRates = S["GroupRatesOut"];

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

/** Message for an error the user can act on, by where it came from. */
export function describeError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 404) return e.message;
    if (e.status === 413) return e.message;
    if (e.status === 422) return e.message;
    if (e.status === 503)
      return "The API is still starting. Wait a moment and try again.";
    return `The API answered ${e.status}: ${e.message}`;
  }
  return "Can't reach the SpectrumSense API. Make sure the backend is running on port 8000.";
}

async function fail(res: Response): Promise<never> {
  const body = await res.json().catch(() => ({ detail: res.statusText }));
  const detail =
    typeof body.detail === "string"
      ? body.detail
      : Array.isArray(body.detail)
        ? body.detail.map((d: { msg?: string }) => d.msg).join("; ")
        : res.statusText;
  throw new ApiError(detail || "Request failed", res.status);
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!res.ok) return fail(res);
  return res.json();
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) return fail(res);
  return res.json();
}

// --- health and model ------------------------------------------------------

export function getHealth(): Promise<HealthOut> {
  return get<HealthOut>("/health");
}

export function getModel(): Promise<ModelReport> {
  return get<ModelReport>("/model");
}

// --- explore ----------------------------------------------------------------

/** Raw query values as the page received them; the API snaps and notes. */
export type ExploreQuery = Record<string, string | string[] | undefined>;

export function exploreSearch(query: ExploreQuery): string {
  const params = new URLSearchParams();
  for (const key of ["signal", "snr", "offset", "seed"]) {
    const value = query[key];
    if (value === undefined) continue;
    for (const item of Array.isArray(value) ? value : [value]) params.append(key, item);
  }
  const text = params.toString();
  return text ? `?${text}` : "";
}

export function explore(query: ExploreQuery): Promise<ExploreResponse> {
  return get<ExploreResponse>(`/explore${exploreSearch(query)}`);
}

export function exploreCsvUrl(settings: ExploreResponse["settings"]): string {
  const params = new URLSearchParams({
    signal: settings.signal,
    snr: String(settings.snr_db),
    offset: String(settings.offset),
    seed: String(settings.seed),
  });
  return `${API_BASE}/explore/csv?${params}`;
}

// --- analyze ----------------------------------------------------------------

export const DEFAULT_PREPARE: PrepareSettings = {
  offset: 0,
  bandwidth: 0,
  symbol_rate: 0,
  factor: 1,
};

export async function analyzeUpload(
  files: File[],
  settings: PrepareSettings,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  form.append("offset", String(settings.offset));
  form.append("bandwidth", String(settings.bandwidth));
  form.append("symbol_rate", String(settings.symbol_rate));
  form.append("factor", String(settings.factor));
  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) return fail(res);
  return res.json();
}

export function analyzeExample(settings: PrepareSettings): Promise<AnalyzeResponse> {
  return postJson<AnalyzeResponse>("/analyze/example", settings);
}

export function getRun(runId: string): Promise<AnalyzeResponse> {
  return get<AnalyzeResponse>(`/runs/${encodeURIComponent(runId)}`);
}

export function prepareRun(
  runId: string,
  settings: PrepareSettings,
): Promise<AnalyzeResponse> {
  return postJson<AnalyzeResponse>(`/runs/${encodeURIComponent(runId)}/prepare`, settings);
}

export function getWindow(runId: string, number: number): Promise<WindowView> {
  return get<WindowView>(`/runs/${encodeURIComponent(runId)}/windows/${number}`);
}

export function runCsvUrl(runId: string): string {
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/results.csv`;
}

// --- example ----------------------------------------------------------------

export function getExample(): Promise<ExampleInfo> {
  return get<ExampleInfo>("/example");
}

export const EXAMPLE_FILE_URL = `${API_BASE}/example/file`;
