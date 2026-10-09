"use client";

import { useEffect, useRef, useState } from "react";
import type { SpectrogramOut } from "@/lib/api";
import { cividisCss, formatHz } from "@/lib/format";

const AXIS_LEFT = 56;
const AXIS_BOTTOM = 28;
const COLORBAR = 44;
const HEIGHT = 300;

function formatAxis(value: number, title: string): string {
  if (title.includes("Hz")) return formatHz(value);
  if (title.startsWith("Time")) return `${value.toFixed(value < 10 ? 2 : 1)} s`;
  if (title === "Sample") return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toFixed(3);
}

/**
 * Power at each frequency (x) over time (y), on a canvas: one cell per
 * (time step, frequency bin), Cividis from the floor to 0 dB relative to
 * the strongest cell. Hover shows the cell's level; the text summary under
 * the chart carries the same findings in words.
 */
export default function Spectrogram({ spectrogram }: { spectrogram: SpectrogramOut }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(720);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  const rows = spectrogram.levels_db.length;
  const cols = spectrogram.frequencies.length;
  const plotWidth = Math.max(width - AXIS_LEFT - COLORBAR - 16, 100);
  const plotHeight = HEIGHT - AXIS_BOTTOM - 8;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const observer = new ResizeObserver(() => setWidth(parent.clientWidth));
    observer.observe(parent);
    setWidth(parent.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = HEIGHT * ratio;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, HEIGHT);

    // Cells: time runs top (first step) to bottom.
    const image = ctx.createImageData(cols, rows);
    const floor = spectrogram.floor_db;
    for (let r = 0; r < rows; r++) {
      const row = spectrogram.levels_db[r];
      for (let c = 0; c < cols; c++) {
        const t = (row[c] - floor) / -floor;
        const [red, green, blue] = cividisCss(t).match(/\d+/g)!.map(Number);
        const index = (r * cols + c) * 4;
        image.data[index] = red;
        image.data[index + 1] = green;
        image.data[index + 2] = blue;
        image.data[index + 3] = 255;
      }
    }
    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    offscreen.getContext("2d")!.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, AXIS_LEFT, 8, plotWidth, plotHeight);

    // Axes.
    ctx.fillStyle = "#7f8f9e";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const times = spectrogram.times;
    const yTicks = Math.min(6, rows);
    for (let k = 0; k < yTicks; k++) {
      const r = Math.round((k * (rows - 1)) / Math.max(yTicks - 1, 1));
      const y = 8 + ((r + 0.5) / rows) * plotHeight;
      ctx.fillText(formatAxis(times[r], spectrogram.time_axis_title), AXIS_LEFT - 6, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const freqs = spectrogram.frequencies;
    const xTicks = 5;
    for (let k = 0; k < xTicks; k++) {
      const c = Math.round((k * (cols - 1)) / (xTicks - 1));
      const x = AXIS_LEFT + ((c + 0.5) / cols) * plotWidth;
      ctx.fillText(formatAxis(freqs[c], spectrogram.frequency_axis_title), x, HEIGHT - AXIS_BOTTOM + 6);
    }

    // Colour bar.
    const barX = AXIS_LEFT + plotWidth + 10;
    for (let y = 0; y < plotHeight; y++) {
      ctx.fillStyle = cividisCss(1 - y / plotHeight);
      ctx.fillRect(barX, 8 + y, 10, 1);
    }
    ctx.fillStyle = "#7f8f9e";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("0 dB", barX + 14, 10);
    ctx.fillText(`${floor} dB`, barX + 14, 8 + plotHeight - 2);
  }, [spectrogram, width, rows, cols, plotWidth, plotHeight]);

  function onMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const c = Math.floor(((x - AXIS_LEFT) / plotWidth) * cols);
    const r = Math.floor(((y - 8) / plotHeight) * rows);
    if (c < 0 || c >= cols || r < 0 || r >= rows) {
      setHover(null);
      return;
    }
    setHover({
      x,
      y,
      text: `${formatAxis(spectrogram.times[r], spectrogram.time_axis_title)}, ${formatAxis(spectrogram.frequencies[c], spectrogram.frequency_axis_title)}: ${spectrogram.levels_db[r][c].toFixed(1)} dB`,
    });
  }

  return (
    <figure className="border border-rule bg-surface">
      <figcaption className="px-5 pt-4 text-sm text-ink-muted">
        Spectrogram: {spectrogram.frequency_axis_title.toLowerCase()} across,{" "}
        {spectrogram.time_axis_title.toLowerCase()} down, level in dB relative to the strongest cell
      </figcaption>
      <div className="relative px-2 pt-2">
        <canvas
          ref={canvasRef}
          style={{ width, height: HEIGHT, display: "block" }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          role="img"
          aria-label={spectrogram.summary}
        />
        {hover && (
          <div
            className="pointer-events-none absolute border border-rule bg-surface px-2 py-1 text-xs text-ink"
            style={{ left: Math.min(hover.x + 12, width - 220), top: hover.y + 12 }}
          >
            {hover.text}
          </div>
        )}
      </div>
      <p className="border-t border-rule px-5 py-3 text-sm leading-relaxed text-ink">
        {spectrogram.summary}
      </p>
    </figure>
  );
}
