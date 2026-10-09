"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { ExploreControls as Controls, ExploreResponse, NumberControl } from "@/lib/api";
import { trimNumber } from "@/lib/format";

type Settings = ExploreResponse["settings"];

function toQuery(settings: Settings, controls: Controls): string {
  const params = new URLSearchParams({
    signal: settings.signal,
    snr: trimNumber(settings.snr_db, controls.snr.decimals),
    offset: trimNumber(settings.offset, controls.offset.decimals),
    seed: String(settings.seed),
  });
  return `/explore?${params}`;
}

function Slider({
  control,
  value,
  onChange,
  disabled,
  help,
}: {
  control: NumberControl;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  help: string;
}) {
  const id = `control-${control.key}`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm text-ink-muted">
          {control.name.charAt(0).toUpperCase() + control.name.slice(1)}
        </label>
        <span className="tabular text-sm text-ink">
          {trimNumber(value, control.decimals)}
          {control.unit}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={control.minimum}
        max={control.maximum}
        step={control.step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full"
        aria-describedby={`${id}-help`}
      />
      <p id={`${id}-help`} className="mt-1 text-xs text-ink-faint">
        {help}
      </p>
    </div>
  );
}

/**
 * The four Explore settings. Every change pushes a new address, so the
 * page address is the state: it can be bookmarked or shared, and Back and
 * Forward move between examples. The sliders change the address only when
 * released, so dragging does not flood the history.
 */
export default function ExploreControls({
  settings,
  controls,
  snrRange,
}: {
  settings: Settings;
  controls: Controls;
  snrRange: [number, number];
}) {
  const router = useRouter();
  // The page remounts this form (by key) whenever the address changes, so
  // the draft always starts from the settings the API snapped to.
  const [draft, setDraft] = useState<Settings>(settings);

  function commit(next: Settings) {
    setDraft(next);
    router.push(toQuery(next, controls));
  }

  const isNoise = draft.signal === "noise";

  return (
    <form
      className="grid gap-5 border border-rule bg-surface px-5 py-5"
      onSubmit={(e) => {
        e.preventDefault();
        commit(draft);
      }}
    >
      <div>
        <label htmlFor="control-signal" className="text-sm text-ink-muted">
          Signal type
        </label>
        <select
          id="control-signal"
          value={draft.signal}
          onChange={(e) => commit({ ...draft, signal: e.target.value })}
          className="mt-1 w-full border border-rule-strong bg-surface px-3 py-2 text-sm text-ink focus:border-accent"
        >
          {controls.signals.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-ink-faint">
          Noise, or a modulated carrier at 8 samples per symbol with rectangular pulses.
        </p>
      </div>

      <Slider
        control={controls.snr}
        value={draft.snr_db}
        disabled={isNoise}
        onChange={(snr_db) => setDraft({ ...draft, snr_db })}
        help={
          isNoise
            ? "Noise has no SNR."
            : `Signal-to-noise ratio. The model trained on ${snrRange[0]} to ${snrRange[1]} dB; outside that range a decision says little.`
        }
      />
      <Slider
        control={controls.offset}
        value={draft.offset}
        onChange={(offset) => setDraft({ ...draft, offset })}
        help="Carrier offset in cycles per sample. The features remove a steady offset, but large values leave the training range."
      />

      <div>
        <label htmlFor="control-seed" className="text-sm text-ink-muted">
          Example seed
        </label>
        <input
          id="control-seed"
          type="number"
          min={controls.seed.minimum}
          max={controls.seed.maximum}
          step={controls.seed.step}
          value={draft.seed}
          onChange={(e) => setDraft({ ...draft, seed: Number(e.target.value) })}
          className="tabular mt-1 w-full border border-rule-strong bg-surface px-3 py-2 text-sm text-ink focus:border-accent"
        />
        <p className="mt-1 text-xs text-ink-faint">
          Same seed and settings, same samples: a link reproduces the example exactly.
        </p>
      </div>

      <button
        type="submit"
        className="bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90"
      >
        Simulate
      </button>
    </form>
  );
}
