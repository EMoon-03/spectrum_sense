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

/** Whether the API runs on this machine. Then an unreachable API means the
 *  backend isn't started; a deployed one is more likely asleep. */
export const API_IS_LOCAL = (() => {
  try {
    const host = new URL(API_BASE).hostname;
    return ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"].includes(host);
  } catch {
    return false;
  }
})();

/** How long a server-rendered page waits for the API. Render's free plan
 *  can hold a request for most of a minute while it wakes; past this the
 *  page renders its "unavailable" state and the browser waits instead.
 *  Requests from the browser have no limit: an upload can be slow, and the
 *  reader can see that it is in progress. */
const SERVER_TIMEOUT_MS = 8000;

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

/** The API gave no answer: refused, unresolvable, or slower than the
 *  server-side limit. */
export class ApiUnreachable extends Error {}

/** Whether an error means the API is down, asleep, or still starting, as
 *  opposed to an answer it gave. Pages show ApiUnavailable for these. */
export function isUnavailable(e: unknown): boolean {
  if (e instanceof ApiError) return e.status === 502 || e.status === 503 || e.status === 504;
  return e instanceof ApiUnreachable;
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
  if (API_IS_LOCAL)
    return "Can't reach the SpectrumSense API. Make sure the backend is running on port 8000.";
  return "Can't reach the SpectrumSense API. It runs on a free server that sleeps when idle, and waking it can take up to about a minute. Try again shortly.";
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

/** On the server, add the timeout to any signal the caller passed. */
function limited(init: RequestInit): RequestInit {
  if (typeof window !== "undefined") return init;
  const timeout = AbortSignal.timeout(SERVER_TIMEOUT_MS);
  return { ...init, signal: init.signal ? AbortSignal.any([init.signal, timeout]) : timeout };
}

/** Fetch and parse one response. A refused connection (fetch's TypeError)
 *  and the server-side timeout both become ApiUnreachable, so a page has
 *  one "unavailable" path; an abort the caller asked for is rethrown. */
async function request<T>(path: string, init: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, limited(init));
    if (!res.ok) return await fail(res);
    return await res.json();
  } catch (e) {
    const timedOut = e instanceof DOMException && e.name === "TimeoutError";
    if (timedOut || e instanceof TypeError) throw new ApiUnreachable(String(e), { cause: e });
    throw e;
  }
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  return request<T>(path, { cache: "no-store", ...init });
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
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
  return request<AnalyzeResponse>("/analyze", { method: "POST", body: form });
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
