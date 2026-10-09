"use client";

import { useState } from "react";
import type { WindowDecision } from "@/lib/api";
import {
  NOT_CLASSIFIABLE,
  UNCERTAIN,
  decisionColor,
  decisionMarker,
  formatDistance,
  formatScore,
} from "@/lib/format";

const ROW = 26;
const LEFT = 104;
const RIGHT = 16;
const TOP = 10;
const BOTTOM = 26;

function Marker({ shape, x, y, color, selected }: { shape: string; x: number; y: number; color: string; selected: boolean }) {
  const s = selected ? 6 : 4.5;
  const stroke = selected ? "#101b26" : color;
  const common = { fill: color, stroke, strokeWidth: selected ? 2 : 1 };
  switch (shape) {
    case "square":
      return <rect x={x - s} y={y - s} width={2 * s} height={2 * s} {...common} />;
    case "diamond":
      return <polygon points={`${x},${y - s - 1} ${x + s + 1},${y} ${x},${y + s + 1} ${x - s - 1},${y}`} {...common} />;
    case "triangle":
      return <polygon points={`${x},${y - s - 1} ${x + s + 1},${y + s} ${x - s - 1},${y + s}`} {...common} />;
    case "cross":
      return (
        <g stroke={stroke} strokeWidth={2}>
          <line x1={x - s} y1={y - s} x2={x + s} y2={y + s} />
          <line x1={x - s} y1={y + s} x2={x + s} y2={y - s} />
        </g>
      );
    default:
      return <circle cx={x} cy={y} r={s} {...common} />;
  }
}

/**
 * The decision for every window: window number across, one row per
 * decision down. Rows differ by position and marker shape as well as
 * colour. Click a window to select it.
 */
export default function WindowTimeline({
  windows,
  classes,
  selected,
  onSelect,
  summary,
}: {
  windows: WindowDecision[];
  classes: string[];
  selected: number;
  onSelect: (number: number) => void;
  summary: string;
}) {
  const [hover, setHover] = useState<WindowDecision | null>(null);
  const rows = [...classes, UNCERTAIN];
  if (windows.some((w) => w.label === NOT_CLASSIFIABLE)) rows.push(NOT_CLASSIFIABLE);
  const height = TOP + rows.length * ROW + BOTTOM;
  const count = windows.length;
  const width = Math.max(640, Math.min(1400, LEFT + RIGHT + count * 12));
  const x = (n: number) => LEFT + ((n - 0.5) / count) * (width - LEFT - RIGHT);
  const y = (label: string) => TOP + (rows.indexOf(label) + 0.5) * ROW;
  const tickEvery = count <= 20 ? 1 : count <= 60 ? 5 : count <= 120 ? 10 : 20;

  return (
    <figure className="border border-rule bg-surface">
      <figcaption className="px-5 pt-4 text-sm text-ink-muted">
        Decision for each window (window number across)
      </figcaption>
      <div className="overflow-x-auto px-2 pt-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          style={{ minWidth: 560, display: "block" }}
          role="img"
          aria-label={summary}
        >
          {rows.map((label) => (
            <g key={label}>
              <line x1={LEFT} x2={width - RIGHT} y1={y(label)} y2={y(label)} stroke="#eef1f4" />
              <text x={LEFT - 8} y={y(label)} textAnchor="end" dominantBaseline="middle" fontSize={12} fill="#4f6273">
                {label}
              </text>
            </g>
          ))}
          {windows.map((w) =>
            (w.number === 1 || w.number % tickEvery === 0) ? (
              <text key={`t${w.number}`} x={x(w.number)} y={height - 8} textAnchor="middle" fontSize={11} fill="#7f8f9e">
                {w.number}
              </text>
            ) : null,
          )}
          {windows.map((w) => (
            <g
              key={w.number}
              style={{ cursor: "pointer" }}
              onClick={() => onSelect(w.number)}
              onMouseEnter={() => setHover(w)}
              onMouseLeave={() => setHover(null)}
              tabIndex={0}
              role="button"
              aria-label={`Window ${w.number}: ${w.label}`}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onSelect(w.number);
              }}
            >
              <rect x={x(w.number) - 6} y={TOP} width={12} height={rows.length * ROW} fill="transparent" />
              <Marker
                shape={decisionMarker(w.label)}
                x={x(w.number)}
                y={y(w.label)}
                color={decisionColor(w.label)}
                selected={w.number === selected}
              />
            </g>
          ))}
        </svg>
      </div>
      <p className="px-5 pb-1 pt-1 text-xs text-ink-faint" aria-live="polite">
        {hover
          ? `Window ${hover.number}: ${hover.label}${hover.reason ? ` (${hover.reason})` : ""}, distance ${formatDistance(hover.distance)}, top score ${formatScore(hover.top_score)}`
          : "Hover a marker for its scores; click to open the window below."}
      </p>
      <p className="border-t border-rule px-5 py-3 text-sm leading-relaxed text-ink">
        {summary}
      </p>
    </figure>
  );
}
