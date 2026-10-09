"use client";

import { useState } from "react";
import { DEFAULT_PREPARE, type AnalyzeResponse, type PrepareSettings } from "@/lib/api";
import Notice from "@/components/Notice";

function Field({
  id,
  label,
  help,
  value,
  onChange,
  step,
  min,
  max,
}: {
  id: string;
  label: string;
  help: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <div>
      <label htmlFor={id} className="text-sm text-ink-muted">
        {label}
      </label>
      <input
        id={id}
        type="number"
        value={value}
        step={step ?? "any"}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))}
        className="tabular mt-1 w-full border border-rule-strong bg-surface px-3 py-2 text-sm text-ink focus:border-accent"
        aria-describedby={`${id}-help`}
      />
      <p id={`${id}-help`} className="mt-1 text-xs text-ink-faint">
        {help}
      </p>
    </div>
  );
}

/**
 * The three optional preparation steps (shift, filter, resample), applied
 * in that order before any analysis. Frequencies are in hertz when the
 * recording states its sample rate and in cycles per sample otherwise.
 */
export default function PreparePanel({
  result,
  busy,
  onApply,
}: {
  result: AnalyzeResponse;
  busy: boolean;
  onApply: (settings: PrepareSettings) => void;
}) {
  // Remounted (by key) whenever the run or its preparation changes.
  const [draft, setDraft] = useState<PrepareSettings>(result.preparation.settings);

  const rate = result.recording.sample_rate;
  const unit = rate === null || rate === undefined ? "cycles per sample" : "Hz";
  const prepared = result.preparation.description !== "none";

  return (
    <form
      className="border border-rule bg-surface"
      onSubmit={(e) => {
        e.preventDefault();
        onApply(draft);
      }}
    >
      <div className="px-5 pt-4">
        <h2 className="serif text-lg text-ink">Prepare the recording (optional)</h2>
        <p className="mt-1 text-sm leading-relaxed text-ink-muted">
          Shift the signal to 0 {unit === "Hz" ? "Hz" : ""}, keep a band around it, and resample to the
          model&rsquo;s 8 samples per symbol. Steps run in that order, and the
          per-window results then describe the prepared recording.
        </p>
      </div>

      <div className="grid gap-4 px-5 py-4 sm:grid-cols-2">
        <Field
          id="prepare-offset"
          label={`Carrier offset to remove (${unit})`}
          help="The signal's frequency, read from the spectrogram, moves to 0. A shift alone changes no answer."
          value={draft.offset}
          onChange={(offset) => setDraft({ ...draft, offset })}
        />
        <Field
          id="prepare-bandwidth"
          label={`Bandwidth to keep (${unit})`}
          help="A linear-phase low-pass filter around 0. 0 keeps every frequency."
          value={draft.bandwidth}
          onChange={(bandwidth) => setDraft({ ...draft, bandwidth })}
          min={0}
        />
        {rate !== null && rate !== undefined ? (
          <Field
            id="prepare-symbol-rate"
            label="Symbol rate (symbols per second)"
            help="Resamples to 8 samples per symbol. 0 keeps the timing."
            value={draft.symbol_rate}
            onChange={(symbol_rate) => setDraft({ ...draft, symbol_rate, factor: 1 })}
            min={0}
          />
        ) : (
          <Field
            id="prepare-factor"
            label="Resampling factor"
            help="New rate divided by old rate, 0.01 to 8. The file states no sample rate, so a symbol rate cannot be used."
            value={draft.factor}
            onChange={(factor) => setDraft({ ...draft, factor, symbol_rate: 0 })}
            min={0.01}
            max={8}
          />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-rule px-5 py-3">
        <button
          type="submit"
          disabled={busy}
          className="bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:bg-rule disabled:text-ink-faint"
        >
          {busy ? "Preparing…" : "Apply"}
        </button>
        <button
          type="button"
          disabled={busy || !prepared}
          onClick={() => {
            setDraft(DEFAULT_PREPARE);
            onApply(DEFAULT_PREPARE);
          }}
          className="border border-rule px-4 py-2 text-sm text-ink-muted transition-colors hover:border-accent hover:text-accent disabled:text-ink-faint"
        >
          Reset the preparation
        </button>
        <span className="text-sm text-ink-muted">{result.preparation.summary}</span>
      </div>

      {prepared && (
        <div className="border-t border-rule px-5 py-3 text-sm leading-relaxed text-ink">
          <ul className="list-disc pl-5">
            {result.preparation.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </div>
      )}
      {result.preparation.band_note && (
        <div className="px-5 pb-4">
          <Notice tone="flag">{result.preparation.band_note}</Notice>
        </div>
      )}
    </form>
  );
}
