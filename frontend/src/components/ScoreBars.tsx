import type { Decision } from "@/lib/api";
import { formatScore } from "@/lib/format";

/** Model scores for the trained classes, each bar printing its value so
 *  the chart reads without colour or hover. */
export default function ScoreBars({ decision }: { decision: Decision }) {
  return (
    <div className="border border-rule bg-surface px-5 py-4">
      <h3 className="text-sm text-ink-muted">Scores for the trained classes</h3>
      <ul className="mt-3 space-y-2" aria-label="Model scores">
        {decision.classes.map((name, index) => {
          const score = decision.scores[index];
          const width = score === null || score === undefined ? 0 : Math.max(score * 100, 0.5);
          const leading = decision.leading === name;
          return (
            <li key={name} className="grid grid-cols-[4rem_1fr_3rem] items-center gap-3 text-sm">
              <span className={leading ? "font-medium text-ink" : "text-ink-muted"}>{name}</span>
              <span className="h-3 w-full bg-plain-soft" aria-hidden="true">
                <span
                  className="block h-3 bg-signal"
                  style={{ width: `${width}%`, opacity: leading ? 1 : 0.55 }}
                />
              </span>
              <span className="tabular text-right text-ink">{formatScore(score)}</span>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-ink-faint">
        {decision.scores_summary} Scores are model outputs, not calibrated
        confidence or proof that an unknown signal belongs to a class.
      </p>
    </div>
  );
}
