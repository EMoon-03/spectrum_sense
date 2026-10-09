import type { Decision } from "@/lib/api";
import {
  DECISION_EXPLANATIONS,
  DECISION_LABELS,
  REASON_UNFAMILIAR,
  decisionKind,
  decisionTone,
  formatDistance,
  formatScore,
} from "@/lib/format";

/**
 * The answer for one window, with the reason and the two limits it was
 * checked against. The badge says which kind of answer this is; the table
 * shows the comparison, so "uncertain" is never a bare word.
 */
export default function DecisionBadge({ decision }: { decision: Decision }) {
  const kind = decisionKind(decision.label);
  const shown =
    kind === "class" ? decision.label : DECISION_LABELS[kind];
  const distanceFailed =
    decision.distance !== null &&
    decision.distance !== undefined &&
    !(decision.distance <= decision.distance_threshold);
  const scoreFailed =
    decision.reason !== null &&
    decision.reason !== undefined &&
    decision.reason !== REASON_UNFAMILIAR &&
    kind === "uncertain";

  return (
    <div className="border border-rule bg-surface">
      <div className="flex flex-wrap items-baseline justify-between gap-3 px-5 pt-4">
        <div>
          <p className="text-xs text-ink-faint">Demo model decision</p>
          <p className="serif mt-1 text-3xl tracking-tight text-ink">{shown}</p>
        </div>
        <span
          className={`inline-block border px-2 py-0.5 text-xs font-medium ${decisionTone(decision.label)}`}
          title={DECISION_EXPLANATIONS[kind]}
        >
          {DECISION_LABELS[kind]}
          {decision.reason && kind === "uncertain" ? `: ${decision.reason}` : ""}
        </span>
      </div>

      <p className="mt-3 px-5 text-sm leading-relaxed text-ink-muted">
        {decision.summary}
      </p>

      {kind !== "not-classifiable" && (
        <dl className="mt-4 grid grid-cols-2 divide-x divide-rule border-t border-rule text-sm">
          <div className="px-5 py-2.5">
            <dt className="text-xs text-ink-faint">Distance to nearest class</dt>
            <dd className={`tabular mt-0.5 ${distanceFailed ? "text-uncertain" : "text-ink"}`}>
              {formatDistance(decision.distance)}
              <span className="text-ink-faint"> / limit {formatDistance(decision.distance_threshold)}</span>
            </dd>
          </div>
          <div className="px-5 py-2.5">
            <dt className="text-xs text-ink-faint">Top score</dt>
            <dd className={`tabular mt-0.5 ${scoreFailed ? "text-uncertain" : "text-ink"}`}>
              {formatScore(decision.top_score)}
              <span className="text-ink-faint"> / minimum {formatScore(decision.min_top_score)}</span>
              {decision.leading && (
                <span className="text-ink-faint"> ({decision.leading})</span>
              )}
            </dd>
          </div>
        </dl>
      )}
    </div>
  );
}
