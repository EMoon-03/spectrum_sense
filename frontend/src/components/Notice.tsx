/** A left-ruled note: `tone` picks the rule colour. Used for API errors,
 *  outside-the-training-range flags, and link notes. */
export default function Notice({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "flag" | "bad";
  children: React.ReactNode;
}) {
  const rule =
    tone === "bad"
      ? "border-bad bg-surface"
      : tone === "flag"
        ? "border-flag bg-flag-soft"
        : "border-rule-strong bg-surface";
  return (
    <div className={`border-l-2 px-4 py-3 text-sm leading-relaxed text-ink ${rule}`}>
      {children}
    </div>
  );
}
