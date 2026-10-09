import { Suspense } from "react";
import AnalyzePanel from "@/components/AnalyzePanel";

export default function AnalyzePage() {
  return (
    <main>
      <section className="border-b border-rule bg-surface">
        <div className="mx-auto max-w-6xl px-6 pb-8 pt-10">
          <h1 className="serif text-4xl tracking-tight text-ink">Try your own recording</h1>
          <p className="mt-2 max-w-[68ch] leading-relaxed text-ink-muted">
            Every complete 1,024-sample window is classified. A timeline shows
            the answer for each window, a spectrogram shows the whole
            recording, and any single window can be opened. The demo model has
            seen only simulated signals, so a real recording is the honest test
            of what it learned: expect &ldquo;uncertain&rdquo; often.
          </p>
        </div>
      </section>
      <div className="mx-auto max-w-6xl px-6 py-10">
        <Suspense fallback={<p className="text-sm text-ink-faint">Loading…</p>}>
          <AnalyzePanel />
        </Suspense>
      </div>
    </main>
  );
}
