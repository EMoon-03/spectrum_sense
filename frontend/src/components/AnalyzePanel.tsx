"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import DecisionBadge from "@/components/DecisionBadge";
import EvidenceCard from "@/components/EvidenceCard";
import Notice from "@/components/Notice";
import PossibleCauses from "@/components/PossibleCauses";
import PreparePanel from "@/components/PreparePanel";
import ScoreBars from "@/components/ScoreBars";
import SignalPlots from "@/components/SignalPlots";
import Spectrogram from "@/components/Spectrogram";
import WindowTimeline from "@/components/WindowTimeline";
import {
  ApiError,
  DEFAULT_PREPARE,
  EXAMPLE_FILE_URL,
  analyzeExample,
  analyzeUpload,
  describeError,
  getRun,
  getWindow,
  prepareRun,
  runCsvUrl,
  type AnalyzeResponse,
  type WindowView,
} from "@/lib/api";
import { formatCount, formatDb, formatHz, formatPct, markdownToHtml } from "@/lib/format";

const ACCEPT =
  ".csv,.cf32,.fc32,.cfile,.cs16,.sc16,.cs8,.cu8,.npy,.wav,.sigmf,.sigmf-meta,.sigmf-data";

export default function AnalyzePanel() {
  const router = useRouter();
  const params = useSearchParams();
  const runParam = params.get("run");
  const windowParam = Number(params.get("window") || 1);

  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [view, setView] = useState<WindowView | null>(null);
  const [selected, setSelected] = useState<number>(windowParam > 0 ? windowParam : 1);
  const [busy, setBusy] = useState<"analyze" | "prepare" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [windowError, setWindowError] = useState<string | null>(null);
  const restored = useRef<string | null>(null);

  const place = useCallback(
    (run: AnalyzeResponse, window: number) => {
      const next = new URLSearchParams({ run: run.run_id, window: String(window) });
      router.replace(`/analyze?${next}`);
    },
    [router],
  );

  // A reload with ?run= in the address reads the run back; a run this
  // process forgot is a 404, which the message explains.
  useEffect(() => {
    if (!runParam || result?.run_id === runParam || restored.current === runParam) return;
    restored.current = runParam;
    setBusy("analyze");
    getRun(runParam)
      .then((run) => {
        setResult(run);
        setError(null);
      })
      .catch((e) => setError(describeError(e)))
      .finally(() => setBusy(null));
  }, [runParam, result]);

  // The selected window, clamped to the run; the input may hold any number.
  const number = result ? Math.min(Math.max(selected, 1), result.windows.length) : 1;

  // Whether the window shown is the one selected, for this run and preparation.
  const windowReady =
    result !== null && view !== null && view.run_id === result.run_id && view.number === number;

  // The selected window's plots follow the selection and the preparation.
  useEffect(() => {
    if (!result) return;
    let cancelled = false;
    getWindow(result.run_id, number)
      .then((w) => {
        if (cancelled) return;
        setView(w);
        setWindowError(null);
      })
      .catch((e) => {
        if (!cancelled) setWindowError(describeError(e));
      });
    place(result, number);
    return () => {
      cancelled = true;
    };
  }, [result, number, place]);

  async function run(work: () => Promise<AnalyzeResponse>, kind: "analyze" | "prepare") {
    setBusy(kind);
    setError(null);
    try {
      const next = await work();
      setResult(next);
      if (kind === "analyze") setSelected(1);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404 && kind === "prepare") {
        setError("This analysis expired on the server. Analyze the recording again.");
      } else {
        setError(describeError(e));
      }
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="space-y-6">
      <section className="border border-rule bg-surface">
        <div className="px-5 pt-4">
          <h2 className="serif text-lg text-ink">Recording to analyze</h2>
          <p className="mt-1 text-sm leading-relaxed text-ink-muted">
            1,024 to 100,000 I/Q samples as CSV (<span className="mono">I,Q</span>), raw SDR
            (<span className="mono">.cf32 .cs16 .cs8 .cu8</span>), NumPy, two-channel WAV, or SigMF
            (one archive, or the <span className="mono">.sigmf-meta</span> and{" "}
            <span className="mono">.sigmf-data</span> pair together). Nothing is stored.
          </p>
        </div>
        <form
          className="flex flex-wrap items-end gap-3 px-5 py-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (files.length) run(() => analyzeUpload(files, DEFAULT_PREPARE), "analyze");
          }}
        >
          <div className="min-w-64 flex-1">
            <label htmlFor="upload" className="text-sm text-ink-muted">
              Your recording
            </label>
            <input
              id="upload"
              type="file"
              accept={ACCEPT}
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              className="mt-1 block w-full text-sm text-ink file:mr-3 file:border file:border-rule-strong file:bg-surface file:px-3 file:py-1.5 file:text-sm file:text-ink-muted"
            />
          </div>
          <button
            type="submit"
            disabled={!files.length || busy !== null}
            className="bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:bg-rule disabled:text-ink-faint"
          >
            {busy === "analyze" ? "Analyzing…" : "Analyze upload"}
          </button>
          <span className="text-sm text-ink-faint">or</span>
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => run(() => analyzeExample(DEFAULT_PREPARE), "analyze")}
            className="border border-rule px-4 py-2 text-sm text-ink-muted transition-colors hover:border-accent hover:text-accent disabled:text-ink-faint"
          >
            Use the bundled satellite recording (QPSK, CC BY 4.0)
          </button>
        </form>
      </section>

      {error && <Notice tone="bad">{error}</Notice>}

      {result && (
        <>
          {result.example && (
            <section className="border border-rule bg-surface px-5 py-4 text-sm leading-relaxed text-ink-muted">
              <div
                className="prose-plain"
                dangerouslySetInnerHTML={{ __html: markdownToHtml(result.example.credit_markdown) }}
              />
              <details className="mt-3">
                <summary className="cursor-pointer text-accent">What this recording holds, and why so many windows are uncertain</summary>
                <div
                  className="prose-plain mt-2"
                  dangerouslySetInnerHTML={{ __html: markdownToHtml(result.example.about_markdown) }}
                />
              </details>
              <p className="mt-3">
                <a href={EXAMPLE_FILE_URL} className="text-accent underline underline-offset-4">
                  Download the SigMF file
                </a>{" "}
                to try it elsewhere; keep the credit with it.
              </p>
            </section>
          )}

          <dl className="grid gap-px border border-rule bg-rule sm:grid-cols-4">
            {[
              ["Format", result.recording.source_format],
              ["Samples as loaded", formatCount(result.recording.sample_count)],
              ["Sample rate", result.recording.sample_rate === null || result.recording.sample_rate === undefined ? "not stated" : `${formatCount(result.recording.sample_rate)} / s`],
              ["Center frequency", formatHz(result.recording.center_frequency)],
            ].map(([label, value]) => (
              <div key={label} className="bg-surface px-5 py-3">
                <dt className="text-xs text-ink-faint">{label}</dt>
                <dd className="tabular mt-0.5 text-sm text-ink">{value}</dd>
              </div>
            ))}
          </dl>
          {result.recording.ignored_uploads > 0 && (
            <Notice>
              {result.recording.ignored_uploads} earlier file{result.recording.ignored_uploads === 1 ? " was" : "s were"} ignored; the most recently added recording was read.
            </Notice>
          )}

          <PreparePanel
            key={`${result.run_id}:${result.preparation.description}`}
            result={result}
            busy={busy === "prepare"}
            onApply={(next) => run(() => prepareRun(result.run_id, next), "prepare")}
          />

          <section className="space-y-4">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <h2 className="serif text-xl text-ink">Every window</h2>
              <a
                href={runCsvUrl(result.run_id)}
                className="border border-rule px-3 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent hover:text-accent"
              >
                Download per-window results as CSV
              </a>
            </div>

            <WindowTimeline
              windows={result.windows}
              classes={result.classes}
              selected={number}
              onSelect={setSelected}
              summary={result.timeline_summary}
            />
            {result.counts.some((row) => row.label === "uncertain" && row.count > 0) && (
              <p className="max-w-[80ch] text-sm leading-relaxed text-ink-muted">
                <span className="text-ink">Possible causes.</span> {result.causes_summary} Choose a window
                to see what was measured in it.
              </p>
            )}

            <div className="grid gap-4 md:grid-cols-[1fr_1fr]">
              <table className="border border-rule bg-surface text-sm">
                <caption className="px-5 pt-4 text-left text-sm text-ink-muted">Windows per decision</caption>
                <thead>
                  <tr className="text-xs text-ink-faint">
                    <th className="px-5 pb-1 pt-2 text-left font-normal">Decision</th>
                    <th className="px-5 pb-1 pt-2 text-right font-normal">Windows</th>
                    <th className="px-5 pb-1 pt-2 text-right font-normal">Share</th>
                  </tr>
                </thead>
                <tbody>
                  {result.counts.map((row) => (
                    <tr key={row.label} className="border-t border-rule">
                      <td className="px-5 py-1.5 text-ink">{row.label}</td>
                      <td className="tabular px-5 py-1.5 text-right text-ink">{row.count}</td>
                      <td className="tabular px-5 py-1.5 text-right text-ink-muted">
                        {formatPct(row.count / result.windows.length)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="border border-rule bg-surface px-5 py-4 text-sm">
                <p className="text-ink-muted">Jump to a loud window</p>
                <p className="mt-1 text-xs text-ink-faint">
                  The windows with the most power, relative to the median window. In a recording of bursts they are the most likely to hold a signal.
                </p>
                <ul className="mt-2 flex flex-wrap gap-2">
                  {result.loudest.map((item) => (
                    <li key={item.number}>
                      <button
                        type="button"
                        onClick={() => setSelected(item.number)}
                        className={`border px-2.5 py-1 text-xs transition-colors hover:border-accent ${number === item.number ? "border-accent text-accent" : "border-rule text-ink-muted"}`}
                      >
                        Window {item.number} <span className="tabular text-ink-faint">{formatDb(item.level_db)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
                {result.ignored_samples > 0 && (
                  <p className="mt-3 text-xs text-ink-faint">
                    The last {formatCount(result.ignored_samples)} samples, after window {result.windows.length}, were ignored because they do not fill a complete 1,024-sample window.
                  </p>
                )}
              </div>
            </div>

            <Spectrogram spectrogram={result.spectrogram} />
          </section>

          <section className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="serif text-xl text-ink">One window</h2>
              <label htmlFor="window-number" className="text-sm text-ink-muted">
                Window
              </label>
              <input
                id="window-number"
                type="number"
                min={1}
                max={result.windows.length}
                value={number}
                onChange={(e) => setSelected(Number(e.target.value) || 1)}
                className="tabular w-24 border border-rule-strong bg-surface px-3 py-1.5 text-sm text-ink focus:border-accent"
              />
              <span className="text-sm text-ink-faint">of {result.windows.length}</span>
              {view && (
                <span className="text-sm text-ink-muted">
                  · starts at sample {formatCount(view.start_sample)} · {view.level_phrase}
                </span>
              )}
            </div>

            {windowError && <Notice tone="bad">{windowError}</Notice>}
            {!windowReady && !windowError && <p className="text-sm text-ink-faint">Loading window {number}…</p>}
            {view && windowReady && (
              <>
                <div className="grid gap-4 md:grid-cols-2">
                  <DecisionBadge decision={view.decision} />
                  <ScoreBars decision={view.decision} />
                </div>
                <PossibleCauses decision={view.decision} />
                <EvidenceCard decision={view.decision} />
                <SignalPlots view={view.view} />
              </>
            )}
          </section>

          <p className="text-xs text-ink-faint">
            Run {result.run_id} · model {result.model_version} · kept on the server for about{" "}
            {Math.round(result.expires_in_seconds / 60)} more minutes, then forgotten.
          </p>
        </>
      )}
    </div>
  );
}
