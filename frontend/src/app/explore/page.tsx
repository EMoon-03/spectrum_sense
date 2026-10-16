import Link from "next/link";
import ApiUnavailable from "@/components/ApiUnavailable";
import DecisionBadge from "@/components/DecisionBadge";
import EvidenceCard from "@/components/EvidenceCard";
import ExploreControls from "@/components/ExploreControls";
import Notice from "@/components/Notice";
import PossibleCauses from "@/components/PossibleCauses";
import ScoreBars from "@/components/ScoreBars";
import SignalPlots from "@/components/SignalPlots";
import { describeError, explore, exploreCsvUrl, getModel, isUnavailable } from "@/lib/api";
import { trimNumber } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ExplorePage({ searchParams }: PageProps<"/explore">) {
  const query = await searchParams;

  let result;
  let model;
  try {
    [result, model] = await Promise.all([explore(query), getModel()]);
  } catch (e) {
    return (
      <main className="mx-auto max-w-6xl px-6 py-16">
        <div className="max-w-xl">
          {isUnavailable(e) ? <ApiUnavailable /> : <Notice tone="bad">{describeError(e)}</Notice>}
        </div>
      </main>
    );
  }

  const { settings, notes, in_training_range, decision, view } = result;
  const controls = model.controls;
  const [low, high] = model.training.snr_db_range;
  const [offLow, offHigh] = model.training.freq_offset_range;
  const link = `/explore?signal=${settings.signal}&snr=${trimNumber(settings.snr_db, controls.snr.decimals)}&offset=${trimNumber(settings.offset, controls.offset.decimals)}&seed=${settings.seed}`;

  return (
    <main>
      <section className="border-b border-rule bg-surface">
        <div className="mx-auto max-w-6xl px-6 pb-8 pt-10">
          <h1 className="serif text-4xl tracking-tight text-ink">Explore a signal</h1>
          <p className="mt-2 max-w-[68ch] leading-relaxed text-ink-muted">
            Simulate one 1,024-sample window of noise, BPSK, or QPSK, read the
            demo model&rsquo;s answer with the limits behind it, and inspect the
            time, constellation, and spectrum views. The page address holds the
            settings, so a link reproduces the example exactly.
          </p>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="grid gap-8 lg:grid-cols-[18rem_1fr]">
          <aside className="space-y-4">
            <ExploreControls key={link} settings={settings} controls={controls} snrRange={[low, high]} />
            <div className="border border-rule bg-surface px-5 py-4 text-sm">
              <p className="text-ink-muted">Share this example</p>
              <p className="mono mt-1 break-all text-xs text-ink">{link}</p>
              <a
                href={exploreCsvUrl(settings)}
                className="mt-3 inline-block border border-rule px-3 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent hover:text-accent"
              >
                Download as CSV (I,Q)
              </a>
              <p className="mt-2 text-xs text-ink-faint">
                Upload it on the <Link href="/analyze" className="text-accent underline underline-offset-4">Analyze</Link> page to see the whole-recording view.
              </p>
            </div>
          </aside>

          <div className="space-y-4">
            {notes.length > 0 && (
              <Notice tone="flag">
                <p>Some values in the link were changed:</p>
                <ul className="mt-1 list-disc pl-5">
                  {notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </Notice>
            )}
            {in_training_range.snr === false && (
              <Notice tone="flag">
                This SNR is outside the demo model&rsquo;s training range of {low} to {high} dB. A decision here says little about the signal type.
              </Notice>
            )}
            {!in_training_range.offset && (
              <Notice tone="flag">
                This carrier offset is outside the training range of {offLow} to {offHigh} cycles per sample.
              </Notice>
            )}

            <p className="text-sm text-ink-muted">
              Actual simulated type: <span className="font-medium text-ink">{settings.signal}</span>
              <span className="text-ink-faint"> · {view.samples.i.length.toLocaleString("en-US")} I/Q samples · model {result.model_version}</span>
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <DecisionBadge decision={decision} />
              <ScoreBars decision={decision} />
            </div>
            <PossibleCauses decision={decision} />
            <EvidenceCard decision={decision} />

            <SignalPlots view={view} />
          </div>
        </div>
      </div>
    </main>
  );
}
