import type { Decision } from "@/lib/api";
import { decisionKind } from "@/lib/format";

/**
 * For an "uncertain" window, the properties measured in it that no training
 * window has. Each names the number it measured; none names a modulation or
 * changes the decision. When no check fires, the card says so instead of
 * guessing.
 */
export default function PossibleCauses({ decision }: { decision: Decision }) {
  if (decisionKind(decision.label) !== "uncertain") return null;
  const causes = decision.causes ?? [];

  return (
    <div className="border border-rule bg-surface px-5 py-4 text-sm">
      <h3 className="text-sm text-ink-muted">Possible causes</h3>
      {causes.length === 0 ? (
        <p className="mt-2 leading-relaxed text-ink">
          No measured cause. None of the checks for a power change within the
          window, a bare carrier, or a steady component at 0 Hz fires; the
          window&rsquo;s features lie outside what the model learned in a way
          these checks do not name. &ldquo;Why this answer&rdquo; shows where.
        </p>
      ) : (
        <ul className="mt-2 space-y-2">
          {causes.map((cause) => (
            <li key={cause.key}>
              <p className="font-medium text-ink">{cause.title}</p>
              <p className="mt-0.5 leading-relaxed text-ink-muted">{cause.detail}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
