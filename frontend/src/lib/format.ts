/**
 * Labels, tones, and number formats shared by every page, so a decision is
 * rendered the same way wherever it appears.
 */

export const UNCERTAIN = "uncertain";
export const NOT_CLASSIFIABLE = "not classifiable";
export const REASON_UNFAMILIAR = "unlike the training data";
export const REASON_CLOSE_SCORES = "scores are close";

// Chart colours (Okabe-Ito). Kept in one place with the CSS tokens.
export const BLUE = "#0072B2";
export const VERMILLION = "#D55E00";
export const ORANGE = "#E69F00";
export const GREY = "#6B7784";

export type DecisionKind = "class" | "uncertain" | "not-classifiable";

export function decisionKind(label: string): DecisionKind {
  if (label === UNCERTAIN) return "uncertain";
  if (label === NOT_CLASSIFIABLE) return "not-classifiable";
  return "class";
}

// Which kind of answer this is. Surfaced as a badge because "the model
// named a class" and "the model declined" are different kinds of claim,
// and a score bar alone does not convey it.
export const DECISION_LABELS: Record<DecisionKind, string> = {
  class: "Named a class",
  uncertain: "Uncertain",
  "not-classifiable": "Not classifiable",
};

export const DECISION_EXPLANATIONS: Record<DecisionKind, string> = {
  class:
    "The window is within the distance limit of a trained class and the top score clears the minimum. Scores are model outputs, not calibrated confidence.",
  uncertain:
    "The model declined to name a class. The reason and the limit it failed are shown with the decision.",
  "not-classifiable":
    "The window's features cannot be computed, for example because every sample is zero, so no score exists.",
};

export function decisionTone(label: string): string {
  switch (decisionKind(label)) {
    case "class":
      return "border-signal/40 bg-signal-soft text-signal";
    case "uncertain":
      return "border-uncertain/40 bg-uncertain-soft text-uncertain";
    default:
      return "border-rule bg-plain-soft text-plain";
  }
}

export function decisionColor(label: string): string {
  switch (decisionKind(label)) {
    case "class":
      return BLUE;
    case "uncertain":
      return ORANGE;
    default:
      return GREY;
  }
}

/** Marker shape per decision in the timeline, so rows never differ by
 *  colour alone. */
export function decisionMarker(label: string): "circle" | "square" | "diamond" | "triangle" | "cross" {
  switch (label) {
    case "noise":
      return "circle";
    case "BPSK":
      return "square";
    case "QPSK":
      return "diamond";
    case UNCERTAIN:
      return "triangle";
    default:
      return "cross";
  }
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(2);
}

export function formatDistance(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(2);
}

export function formatDb(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)} dB`;
}

export function formatPct(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

/** Hertz with an SI prefix, to three significant figures past 1 kHz. */
export function formatHz(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toLocaleString("en-US", { maximumFractionDigits: 6 })} GHz`;
  if (abs >= 1e6) return `${(value / 1e6).toLocaleString("en-US", { maximumFractionDigits: 4 })} MHz`;
  if (abs >= 1e3) return `${(value / 1e3).toLocaleString("en-US", { maximumFractionDigits: 3 })} kHz`;
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 2 })} Hz`;
}

/** Trim trailing zeros from a fixed number of decimals. */
export function trimNumber(value: number, decimals: number): string {
  const text = value.toFixed(decimals);
  const trimmed = text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
  return trimmed === "-0" ? "0" : trimmed;
}

// Cividis, 32 stops, from matplotlib. Grows steadily lighter with level and
// stays legible under colour-vision deficiency.
const CIVIDIS = [
  "#00224e", "#00285b", "#002e6a", "#053371", "#1c396f", "#293f6e", "#33446d", "#3c4a6c",
  "#45506c", "#4d556c", "#555b6d", "#5c616e", "#646770", "#6b6d72", "#727274", "#787877",
  "#807f78", "#888578", "#908b78", "#979177", "#a09875", "#a89e73", "#b0a571", "#b9ab6d",
  "#c2b369", "#cbb965", "#d3c05f", "#dcc859", "#e6d051", "#efd748", "#f8df3c", "#fee838",
];

function hex(color: string): [number, number, number] {
  return [
    parseInt(color.slice(1, 3), 16),
    parseInt(color.slice(3, 5), 16),
    parseInt(color.slice(5, 7), 16),
  ];
}

const CIVIDIS_RGB = CIVIDIS.map(hex);

/** RGB for a value in [0, 1], interpolated between the stops. */
export function cividis(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const position = clamped * (CIVIDIS_RGB.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, CIVIDIS_RGB.length - 1);
  const mix = position - low;
  const a = CIVIDIS_RGB[low];
  const b = CIVIDIS_RGB[high];
  return [
    Math.round(a[0] + (b[0] - a[0]) * mix),
    Math.round(a[1] + (b[1] - a[1]) * mix),
    Math.round(a[2] + (b[2] - a[2]) * mix),
  ];
}

export function cividisCss(t: number): string {
  const [r, g, b] = cividis(t);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Minimal Markdown (bold, links, paragraphs, bullet lists) to HTML, for
 *  the credit and description strings the API serves. Input is our own
 *  backend's text, not user content, but it is escaped anyway. */
export function markdownToHtml(text: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const inline = (s: string) =>
    escape(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(
        /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
      );
  const blocks = text.split(/\n\n+/);
  return blocks
    .map((block) => {
      const lines = block.split("\n");
      if (lines.every((line) => line.startsWith("- "))) {
        return `<ul>${lines.map((line) => `<li>${inline(line.slice(2))}</li>`).join("")}</ul>`;
      }
      const items: string[] = [];
      let paragraph: string[] = [];
      for (const line of lines) {
        if (line.startsWith("- ")) {
          if (paragraph.length) {
            items.push(`<p>${inline(paragraph.join(" "))}</p>`);
            paragraph = [];
          }
          items.push(`<ul><li>${inline(line.slice(2))}</li></ul>`);
        } else {
          paragraph.push(line);
        }
      }
      if (paragraph.length) items.push(`<p>${inline(paragraph.join(" "))}</p>`);
      return items.join("").replace(/<\/ul><ul>/g, "");
    })
    .join("");
}
