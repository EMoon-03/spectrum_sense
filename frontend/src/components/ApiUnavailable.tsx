"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, useTransition } from "react";
import Notice from "@/components/Notice";
import { API_BASE, API_IS_LOCAL, type HealthOut } from "@/lib/api";

/** Ask /health this long after the last answer, allow one request this
 *  long, and stop after this long. Render's free plan usually wakes in
 *  under a minute. */
const POLL_MS = 3000;
const REQUEST_MS = 10000;
const GIVE_UP_MS = 90000;

type Phase = "waiting" | "loading" | "stopped";

/**
 * Wait for a sleeping API from the browser: ask /health every few seconds
 * and, once it answers "ok", re-render the page's server components with
 * router.refresh(), keeping this component's state. If a refresh comes back
 * without data, keep asking until the time limit.
 */
function useWake() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("waiting");
  const [elapsed, setElapsed] = useState(0);
  const [refreshing, startRefresh] = useTransition();
  const started = useRef<number | null>(null);

  useEffect(() => {
    if (API_IS_LOCAL || phase === "stopped" || refreshing) return;
    if (started.current === null) started.current = Date.now();
    const since = started.current;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;

    async function check() {
      controller = new AbortController();
      const limit = setTimeout(() => controller?.abort(), REQUEST_MS);
      let ok = false;
      try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store", signal: controller.signal });
        ok = res.ok && ((await res.json()) as HealthOut).status === "ok";
      } catch {
        // Asleep, still starting, or unreachable: ask again.
      } finally {
        clearTimeout(limit);
      }
      if (cancelled) return;
      const late = Date.now() - since >= GIVE_UP_MS;
      if (ok && !(late && phase === "loading")) {
        setPhase("loading");
        startRefresh(() => router.refresh());
      } else if (late) {
        setPhase("stopped");
      } else {
        timer = setTimeout(check, POLL_MS);
      }
    }

    // At once on arrival; after a pause if a refresh just came back empty.
    timer = setTimeout(check, phase === "loading" ? POLL_MS : 0);
    const tick = setInterval(() => setElapsed(Math.round((Date.now() - since) / 1000)), 1000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      clearInterval(tick);
      controller?.abort();
    };
  }, [phase, refreshing, router]);

  function retry() {
    started.current = Date.now();
    setElapsed(0);
    setPhase("waiting");
  }

  return { phase, elapsed, retry };
}

/**
 * What a page shows when its server could not reach the API.
 *
 * Against an API on this machine, a hint to start it. A deployed API on
 * Render's free plan sleeps when idle, so there the page says so, waits for
 * it from the browser, and loads itself once it answers; after 90 s it stops
 * and offers to try again. `inline` is the one-line form for a status line.
 *
 * The waiting text is a polite live region; the seconds counter is hidden
 * from screen readers so it isn't read out every second.
 */
export default function ApiUnavailable({ inline = false }: { inline?: boolean }) {
  const { phase, elapsed, retry } = useWake();
  const limit = Math.round(GIVE_UP_MS / 1000);

  if (API_IS_LOCAL) {
    if (inline) return <>API not reachable: start the backend at {API_BASE}</>;
    return (
      <Notice tone="bad">
        The SpectrumSense API at <span className="mono">{API_BASE}</span> isn&rsquo;t
        responding. Start the backend (<code className="mono">uvicorn app.main:app</code> in{" "}
        <code className="mono">backend/</code>), then reload.
      </Notice>
    );
  }

  if (inline) {
    return (
      <span role="status">
        {phase === "waiting" && (
          <>
            API waking up; the free server can take about a minute
            <span aria-hidden="true"> · {elapsed} s</span>
          </>
        )}
        {phase === "loading" && "API answering · loading…"}
        {phase === "stopped" && (
          <>
            API not answering after {limit} s ·{" "}
            <button type="button" onClick={retry} className="text-accent underline underline-offset-4">
              Try again
            </button>
          </>
        )}
      </span>
    );
  }

  return (
    <Notice>
      {phase === "waiting" && (
        <>
          <p>
            The SpectrumSense API isn&rsquo;t answering yet. It runs on a free
            server that sleeps when no one is using it, and waking it can take
            up to about a minute. This page will load by itself when it answers.
          </p>
          <p className="mono mt-2 text-xs text-ink-faint" aria-hidden="true">
            Waiting {elapsed} s
          </p>
        </>
      )}
      {phase === "loading" && <p>The API is answering. Loading the page…</p>}
      {phase === "stopped" && (
        <>
          <p>
            The API still isn&rsquo;t answering after {limit} seconds, so it may
            be down rather than asleep.
          </p>
          <button
            type="button"
            onClick={retry}
            className="mt-3 border border-rule-strong px-3 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent hover:text-accent"
          >
            Try again
          </button>
        </>
      )}
    </Notice>
  );
}
