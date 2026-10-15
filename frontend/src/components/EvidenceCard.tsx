import type { Decision } from "@/lib/api";
import { formatDistance, formatPct } from "@/lib/format";

/** How many features the card lists before "All eight features". */
const SHOWN = 3;

/**
 * Where the window's distance to the nearest trained class lies, feature by
 * feature, with the value each feature would need to fit the others. Each
 * bar prints its share, so the chart reads without colour.
 *
 * The shares split the distance exactly but say where it lies, not what
 * caused it; the note under the bars says so.
 */
export default function EvidenceCard({ decision }: { decision: Decision }) {
  const evidence = decision.evidence;
  if (!evidence) return null;
  const ranked = [...evidence.features].sort((a, b) => b.distance_share - a.distance_share);
  const beyond = evidence.distance > decision.distance_threshold;

  return (
    <div className="border border-rule bg-surface px-5 py-4">
      <h3 className="text-sm text-ink-muted">Why this answer</h3>
      <p className="mt-2 text-sm leading-relaxed text-ink">{evidence.summary}</p>

      <p className="mt-4 text-xs text-ink-faint">
        Where the distance to {evidence.nearest_class} ({formatDistance(evidence.distance)}) lies
      </p>
      <ul className="mt-2 space-y-2" aria-label={`Shares of the distance to ${evidence.nearest_class}`}>
        {ranked.slice(0, SHOWN).map((feature) => (
          <li key={feature.name} className="grid grid-cols-[minmax(0,11rem)_1fr_3.5rem] items-center gap-3 text-sm">
            <span className="text-ink" title={feature.description}>
              {feature.name}
            </span>
            <span className="h-3 w-full bg-plain-soft" aria-hidden="true">
              <span
                className={`block h-3 ${beyond ? "bg-uncertain" : "bg-accent"}`}
                style={{ width: `${Math.max(feature.distance_share * 100, 0.5)}%` }}
              />
            </span>
            <span className="tabular text-right text-ink">{formatPct(feature.distance_share, 0)}</span>
          </li>
        ))}
      </ul>

      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-xs text-accent underline-offset-4 hover:underline">
          All {evidence.features.length} features
        </summary>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-ink-faint">
                <th className="py-1 pr-3 text-left font-normal">Feature</th>
                <th className="py-1 pr-3 text-right font-normal">Share</th>
                <th className="py-1 pr-3 text-right font-normal">Value</th>
                <th className="py-1 pr-3 text-right font-normal">Fits the rest</th>
                <th className="py-1 text-right font-normal">Distance if it did</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((feature) => (
                <tr key={feature.name} className="border-t border-rule">
                  <th scope="row" className="py-1 pr-3 text-left font-normal text-ink">
                    {feature.name}
                    <span className="block text-ink-faint">{feature.description}</span>
                  </th>
                  <td className="tabular py-1 pr-3 text-right text-ink">{formatPct(feature.distance_share, 0)}</td>
                  <td className="tabular py-1 pr-3 text-right text-ink">{feature.value.toFixed(2)}</td>
                  <td className="tabular py-1 pr-3 text-right text-ink-muted">{feature.expected.toFixed(2)}</td>
                  <td
                    className={`tabular py-1 text-right ${feature.distance_if_expected <= decision.distance_threshold ? "text-ink" : "text-ink-muted"}`}
                  >
                    {formatDistance(feature.distance_if_expected)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <p className="mt-3 text-xs leading-relaxed text-ink-faint">
        Shares split the distance exactly, but they say where it lies, not what
        caused it: a feature with an ordinary value can hold a large share when
        it disagrees with the others. &ldquo;Fits the rest&rdquo; is the value
        the {evidence.nearest_class} fit expects given the window&rsquo;s other
        features. The limit is {formatDistance(decision.distance_threshold)}.
      </p>
    </div>
  );
}
