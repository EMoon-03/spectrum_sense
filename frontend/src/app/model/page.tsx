import Link from "next/link";
import ApiUnavailable from "@/components/ApiUnavailable";
import Notice from "@/components/Notice";
import RatesChart from "@/components/RatesChart";
import { describeError, getModel, isUnavailable } from "@/lib/api";
import { formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="bg-surface px-5 py-4">
      <dt className="text-xs text-ink-faint">{label}</dt>
      <dd className="tabular mt-1 text-2xl font-medium text-ink">{value}</dd>
      {note && <dd className="mt-1 text-xs text-ink-faint">{note}</dd>}
    </div>
  );
}

export default async function ModelPage() {
  let model;
  try {
    model = await getModel();
  } catch (e) {
    return (
      <main className="mx-auto max-w-6xl px-6 py-16">
        <div className="max-w-xl">
          {isUnavailable(e) ? <ApiUnavailable /> : <Notice tone="bad">{describeError(e)}</Notice>}
        </div>
      </main>
    );
  }
  const t = model.training;

  return (
    <main>
      <section className="border-b border-rule bg-surface">
        <div className="mx-auto max-w-6xl px-6 pb-8 pt-10">
          <h1 className="serif text-4xl tracking-tight text-ink">How the model works</h1>
          <p className="mt-2 max-w-[68ch] leading-relaxed text-ink-muted">
            A logistic regression on a few explainable features of each
            1,024-sample window, trained on simulated captures and split by
            capture so sibling windows never land on both sides. These are the
            numbers it measured about itself on held-out captures and on probe
            signals it never trained on. Nothing was tuned on either.
          </p>
          <p className="mono mt-3 text-xs text-ink-faint">model {model.version}</p>
        </div>
      </section>

      <div className="mx-auto max-w-6xl space-y-12 px-6 py-10">
        <section>
          <h2 className="serif border-b border-rule pb-2 text-xl text-ink">Held-out test captures</h2>
          <dl className="mt-5 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Test accuracy, always taking the top class" value={formatPct(model.test_accuracy)} note={`${model.test_windows} windows from ${model.test_captures} captures`} />
            <Stat label="Known windows answered uncertain" value={formatPct(model.false_reject_rate)} note="false rejects" />
            <Stat label="Accuracy on the windows it did answer" value={formatPct(model.accepted_accuracy)} />
            <Stat label="Captures used to fit / calibrate" value={`${model.fit_captures} / ${model.calibration_captures}`} note="the calibration quarter sets the distance limit" />
          </dl>
          <div className="mt-5 border border-rule bg-surface px-5 py-4">
            <h3 className="text-sm text-ink-muted">By SNR band: share answered uncertain (orange) and accuracy when answered (blue)</h3>
            <RatesChart rates={model.rates_by_snr} />
            <ul className="mt-2 grid gap-x-8 gap-y-1 text-sm text-ink sm:grid-cols-2">
              {model.rates_by_snr.map((row) => (
                <li key={row.group} className="tabular">
                  <span className="text-ink-muted">{row.group}:</span> {formatPct(row.false_reject_rate)} uncertain, {formatPct(row.accepted_accuracy)} accurate otherwise ({row.windows} windows)
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section>
          <h2 className="serif border-b border-rule pb-2 text-xl text-ink">The &ldquo;uncertain&rdquo; answer</h2>
          <p className="mt-4 max-w-[68ch] leading-relaxed text-ink-muted">
            The classifier always picks a class, even for signals it never saw,
            often with a score near 1. So two checks run first, and either
            makes the answer &ldquo;uncertain&rdquo;:
          </p>
          <ol className="mt-4 grid gap-4 md:grid-cols-2">
            <li className="border border-rule bg-surface px-5 py-4 text-sm leading-relaxed">
              <p className="font-medium text-ink">1. Unlike the training data</p>
              <p className="mt-1 text-ink-muted">
                The Mahalanobis distance from the window&rsquo;s standardized features to the nearest class is above{" "}
                <span className="tabular text-ink">{model.limits.distance_threshold.toFixed(2)}</span>, the {model.limits.distance_percentile}th percentile of distances in a held-back {formatPct(model.limits.calibration_fraction, 0)} of training captures. Each class covariance is shrunk by {model.limits.covariance_shrinkage} toward a scaled identity so it can always be inverted.
              </p>
            </li>
            <li className="border border-rule bg-surface px-5 py-4 text-sm leading-relaxed">
              <p className="font-medium text-ink">2. Scores are close</p>
              <p className="mt-1 text-ink-muted">
                The top score is below <span className="tabular text-ink">{model.limits.min_top_score.toFixed(2)}</span>. A weaker check that catches some windows between trained classes.
              </p>
            </li>
          </ol>

          <h3 className="mt-8 text-sm text-ink-muted">Probe signals the model never trained on: share of windows answered uncertain ({model.probe_windows} windows each)</h3>
          <table className="mt-3 w-full max-w-xl border border-rule bg-surface text-sm">
            <thead>
              <tr className="text-xs text-ink-faint">
                <th className="px-5 pb-1 pt-3 text-left font-normal">SNR</th>
                {model.probe_kinds.map((kind) => (
                  <th key={kind} className="px-5 pb-1 pt-3 text-right font-normal">{kind}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {model.probe_flag_rates.map((row) => (
                <tr key={row.snr_db} className="border-t border-rule">
                  <td className="tabular px-5 py-1.5 text-ink">{row.snr_db} dB</td>
                  {model.probe_kinds.map((kind) => (
                    <td key={kind} className="tabular px-5 py-1.5 text-right text-ink">{formatPct(row.rates[kind])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 max-w-[68ch] text-sm leading-relaxed text-ink-faint">
            A tone is flagged nearly always. 8PSK is flagged almost always from 4 dB up, mostly through the fourth-power line, which QPSK has and 8PSK lacks; 16-QAM in most windows from 4 dB and nearly all from 8 dB. Below 4 dB, noise hides the difference, and most 8PSK and 16-QAM windows at 0 to 2 dB are called QPSK. This is a simple heuristic tested only on synthetic probes, not validated unknown-signal detection.
          </p>
        </section>

        <section>
          <h2 className="serif border-b border-rule pb-2 text-xl text-ink">Training settings</h2>
          <dl className="mt-5 grid gap-px border border-rule bg-rule sm:grid-cols-3">
            <Stat label="Classes" value={t.classes.join(", ")} />
            <Stat label="Captures per class × windows" value={`${t.captures_per_class} × ${t.windows_per_capture}`} note={`seed ${t.seed}`} />
            <Stat label="Window" value={`${t.samples_per_window.toLocaleString("en-US")} samples`} note={`${t.samples_per_symbol} samples per symbol, rectangular pulses`} />
            <Stat label="SNR range" value={`${t.snr_db_range[0]} to ${t.snr_db_range[1]} dB`} />
            <Stat label="Carrier offset range" value={`±${t.freq_offset_range[1]}`} note="cycles per sample" />
            <Stat label="Power" value="fixed" note="total power is normalized, so power alone reveals nothing" />
          </dl>
          <p className="mt-4 max-w-[68ch] text-sm leading-relaxed text-ink-muted">
            A very high synthetic score is expected from idealized symbol timing and repeated symbols. It is not evidence of performance on real recordings; see{" "}
            <Link href="/about" className="text-accent underline underline-offset-4">the method</Link> for what it does and does not establish.
          </p>
        </section>
      </div>
    </main>
  );
}
