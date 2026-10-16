import Link from "next/link";
import { Suspense } from "react";
import ApiUnavailable from "@/components/ApiUnavailable";
import { getHealth } from "@/lib/api";

export const dynamic = "force-dynamic";

const STATUS = "mono mt-6 text-xs text-ink-faint";

const PATHS = [
  {
    href: "/explore",
    title: "Explore a signal",
    body: "Simulate noise, BPSK, or QPSK at any SNR, offset, and seed. Read the decision, the scores, and the limits behind an “uncertain” answer. Every example has a link.",
  },
  {
    href: "/analyze",
    title: "Try your own recording",
    body: "Upload CSV, raw SDR, NumPy, WAV, or SigMF. Every window is classified; a timeline, a spectrogram, and a per-window CSV show the result. Or start with the bundled satellite recording.",
  },
  {
    href: "/model",
    title: "How the model works",
    body: "Held-out accuracy by SNR band, how often known signals are answered uncertain, and how often probe signals it never saw are flagged.",
  },
];

/** The API's status line. It streams in after the rest of the page, so a
 *  sleeping API never holds up the overview. */
async function ApiStatus() {
  const health = await getHealth().catch(() => null);
  return (
    <p className={STATUS}>
      {health?.status === "ok" ? `API ok · model ${health.model.version}` : <ApiUnavailable inline />}
    </p>
  );
}

export default function Home() {
  return (
    <main>
      <section className="border-b border-rule bg-surface">
        <div className="mx-auto max-w-6xl px-6 pb-14 pt-16">
          <p className="text-xs uppercase tracking-wide text-ink-faint">Interactive signal lab</p>
          <h1 className="serif mt-3 max-w-3xl text-5xl tracking-tight text-ink">
            A classifier that says &ldquo;uncertain&rdquo; when it should.
          </h1>
          <p className="mt-5 max-w-[68ch] text-lg leading-relaxed text-ink-muted">
            SpectrumSense simulates complex I/Q samples for noise, BPSK, and
            QPSK, extracts a few explainable features, and trains a
            logistic-regression baseline. A model like that always picks a
            class, even for signals it never saw, so every answer here first
            passes two checks and carries its reasons and limits with it.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/explore" className="bg-accent px-5 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90">
              Explore a signal
            </Link>
            <Link href="/analyze" className="border border-rule-strong px-5 py-2.5 text-sm text-ink transition-colors hover:border-accent hover:text-accent">
              Analyze a recording
            </Link>
          </div>
          <Suspense fallback={<p className={STATUS}>Checking the API…</p>}>
            <ApiStatus />
          </Suspense>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid gap-4 md:grid-cols-3">
          {PATHS.map((item) => (
            <Link key={item.href} href={item.href} className="group border border-rule bg-surface px-5 py-5 transition-colors hover:border-accent">
              <h2 className="serif text-lg text-ink group-hover:text-accent">{item.title}</h2>
              <p className="mt-2 text-sm leading-relaxed text-ink-muted">{item.body}</p>
            </Link>
          ))}
        </div>

        <section className="mt-14 grid gap-10 md:grid-cols-2">
          <div>
            <h2 className="serif border-b border-rule pb-2 text-xl text-ink">What it is</h2>
            <p className="mt-4 text-sm leading-relaxed text-ink-muted">
              A small, CPU-only learning exercise on synthetic data: three classes, a modest baseline, a few features (second- and fourth-order complex moments, amplitude variability, symbol-spaced phase differences), and a split by capture so sibling windows never land in both training and test.
            </p>
          </div>
          <div>
            <h2 className="serif border-b border-rule pb-2 text-xl text-ink">What it is not</h2>
            <p className="mt-4 text-sm leading-relaxed text-ink-muted">
              A validated RF receiver, an electronic-warfare system, or evidence about real recordings. The model expects rectangular pulses at 8 samples per symbol; the bundled real recording has root-raised-cosine pulses at 4, and most of its windows come back &ldquo;uncertain&rdquo;, which is the right answer.{" "}
              <Link href="/about" className="text-accent underline underline-offset-4">The method, in full.</Link>
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
