"use client";

import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GroupRates } from "@/lib/api";
import { BLUE, ORANGE } from "@/lib/format";

/** Held-out windows by SNR band: how often the model declined, and how
 *  accurate it was when it answered. Printed values, not just bars. */
export default function RatesChart({ rates }: { rates: GroupRates[] }) {
  const data = rates.map((row) => ({
    group: row.group === "Noise (no SNR)" ? "Noise" : row.group.replace("BPSK and QPSK at ", "").replace(" SNR", ""),
    uncertain: Math.round(row.false_reject_rate * 1000) / 10,
    accuracy: row.accepted_accuracy === null || row.accepted_accuracy === undefined ? null : Math.round(row.accepted_accuracy * 1000) / 10,
    windows: row.windows,
  }));
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 16, right: 16, bottom: 4, left: -12 }} barGap={2}>
          <CartesianGrid stroke="#eef1f4" vertical={false} />
          <XAxis dataKey="group" tick={{ fontSize: 11, fill: "#7f8f9e" }} tickLine={false} axisLine={{ stroke: "#c0cad4" }} interval={0} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#7f8f9e" }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} />
          <Tooltip
            contentStyle={{ border: "1px solid #dde3e9", borderRadius: 0, fontSize: 12, padding: "6px 8px", boxShadow: "none", background: "#fff" }}
            formatter={(value, name) => [`${value}%`, name === "uncertain" ? "Answered uncertain" : "Accuracy when answered"]}
            labelFormatter={(label, payload) => `${label} (${payload?.[0]?.payload?.windows ?? ""} windows)`}
          />
          <Bar dataKey="uncertain" name="uncertain" fill={ORANGE} isAnimationActive={false}>
            <LabelList dataKey="uncertain" position="top" fontSize={11} fill="#4f6273" formatter={(v) => `${v}%`} />
          </Bar>
          <Bar dataKey="accuracy" name="accuracy" fill={BLUE} isAnimationActive={false}>
            <LabelList dataKey="accuracy" position="top" fontSize={11} fill="#4f6273" formatter={(v) => (v === null ? "" : `${v}%`)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
