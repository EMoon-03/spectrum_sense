"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SignalView } from "@/lib/api";
import { BLUE, VERMILLION } from "@/lib/format";

/** Two significant figures, so ticks on a window of amplitude 0.007 do not
 *  all read "0.0". */
const tick = (v: number) => (v === 0 ? "0" : Number(v.toPrecision(2)).toString());

const TOOLTIP_STYLE = {
  border: "1px solid #dde3e9",
  borderRadius: 0,
  fontSize: 12,
  padding: "6px 8px",
  boxShadow: "none",
  background: "#fff",
};

function Figure({
  title,
  summary,
  children,
}: {
  title: string;
  summary: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="border border-rule bg-surface">
      <figcaption className="px-5 pt-4 text-sm text-ink-muted">{title}</figcaption>
      <div className="h-56 px-2 pt-2">{children}</div>
      <p className="border-t border-rule px-5 py-3 text-sm leading-relaxed text-ink">
        {summary}
      </p>
    </figure>
  );
}

/**
 * Time, constellation, and spectrum views of one window, each with the
 * API's text summary directly beneath it. I is solid blue, Q is dashed
 * vermillion, so the two never differ by colour alone.
 */
export default function SignalPlots({ view }: { view: SignalView }) {
  const shown = view.time_view_samples;
  const time = Array.from({ length: shown }, (_, n) => ({
    n,
    i: view.samples.i[n],
    q: view.samples.q[n],
  }));
  const points = view.samples.i
    .map((i, index) => ({ i, q: view.samples.q[index] }))
    .filter((_, index) => index % view.constellation_step === 0);
  const spectrum = view.spectrum.frequencies.map((f, index) => ({
    f,
    db: view.spectrum.levels_db[index],
  }));

  return (
    <div className="grid gap-4">
      <Figure title="I (solid) and Q (dashed) over time" summary={view.summaries.time}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={time} margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
            <CartesianGrid stroke="#eef1f4" vertical={false} />
            <XAxis
              dataKey="n"
              tick={{ fontSize: 11, fill: "#7f8f9e" }}
              tickLine={false}
              axisLine={{ stroke: "#c0cad4" }}
              label={{ value: "Sample", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#7f8f9e" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#7f8f9e" }}
              tickLine={false}
              axisLine={false}
              domain={["auto", "auto"]}
              tickFormatter={tick}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => (typeof value === "number" ? Number(value.toPrecision(4)).toString() : String(value))}
              labelFormatter={(label) => `Sample ${label}`}
            />
            <Line type="linear" dataKey="i" name="I" stroke={BLUE} strokeWidth={1.75} dot={false} isAnimationActive={false} />
            <Line type="linear" dataKey="q" name="Q" stroke={VERMILLION} strokeWidth={1.75} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Figure>

      <div className="grid gap-4 md:grid-cols-2">
        <Figure title="I/Q plane" summary={view.summaries.constellation}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
              <CartesianGrid stroke="#eef1f4" />
              <XAxis
                type="number"
                dataKey="i"
                name="I"
                tick={{ fontSize: 11, fill: "#7f8f9e" }}
                tickLine={false}
                axisLine={{ stroke: "#c0cad4" }}
                domain={["auto", "auto"]}
                tickFormatter={tick}
              />
              <YAxis
                type="number"
                dataKey="q"
                name="Q"
                tick={{ fontSize: 11, fill: "#7f8f9e" }}
                tickLine={false}
                axisLine={false}
                domain={["auto", "auto"]}
                tickFormatter={tick}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                cursor={{ stroke: "#c0cad4" }}
                formatter={(value) => (typeof value === "number" ? Number(value.toPrecision(4)).toString() : String(value))}
              />
              <Scatter
                data={points}
                fill="rgba(0, 114, 178, 0.35)"
                stroke={BLUE}
                strokeWidth={1}
                isAnimationActive={false}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </Figure>

        <Figure title="FFT magnitude" summary={view.summaries.spectrum}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={spectrum} margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
              <CartesianGrid stroke="#eef1f4" vertical={false} />
              <XAxis
                dataKey="f"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fontSize: 11, fill: "#7f8f9e" }}
                tickLine={false}
                axisLine={{ stroke: "#c0cad4" }}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{ value: "Cycles per sample", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#7f8f9e" }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#7f8f9e" }}
                tickLine={false}
                axisLine={false}
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => v.toFixed(0)}
                label={{ value: "dB", angle: -90, position: "insideLeft", offset: 24, fontSize: 11, fill: "#7f8f9e" }}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value) => (typeof value === "number" ? `${value.toFixed(1)} dB` : String(value))}
                labelFormatter={(label) => `${Number(label).toFixed(4)} cycles/sample`}
              />
              <Line type="linear" dataKey="db" name="Magnitude" stroke={BLUE} strokeWidth={1.25} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Figure>
      </div>
    </div>
  );
}
