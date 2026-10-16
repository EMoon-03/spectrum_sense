"""Analyze a recording and serialize the result.

One function does the work (analyze), one re-prepares a cached run with other
settings (reprepare), and the rest turn a Run into response models. The HTTP
routers only handle transport.
"""

from __future__ import annotations

import math

import numpy as np

import real_example
from causes import TITLES as CAUSE_TITLES, causes_summary, possible_causes
from iq_csv import MAX_SAMPLES
from iq_formats import Recording, load_uploads, select_uploads
from prepare import (
    PrepareSettings,
    level_phrase,
    loudest_windows,
    prepare_recording,
    result_summary,
    window_levels,
)
from signal_config import SAMPLES_PER_WINDOW, UNCERTAIN
from spectrogram import FLOOR_DB, compute_spectrogram
from spectrum_sense import (
    DemoModel,
    classify_recording,
    decision_counts,
    decision_order,
    extract_features,
    results_csv,
)
from summaries import spectrogram_summary, timeline_summary

from . import schemas
from .model import MODEL_VERSION
from .runs import Run, new_run_id, store
from .views import decision_for, display_order, reorder, signal_view


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _analyze_recording(
    demo: DemoModel, recording: Recording, settings: PrepareSettings
) -> tuple:
    prepared = prepare_recording(recording, settings)
    samples = prepared.recording.samples
    truncated = MAX_SAMPLES if any("the app's limit" in step for step in prepared.steps) else None
    return (
        prepared,
        truncated,
        classify_recording(demo.model, samples),
        compute_spectrogram(samples),
        loudest_windows(samples),
    )


def load_example() -> tuple[tuple[str, bytes], ...]:
    return ((real_example.EXAMPLE_PATH.name, real_example.EXAMPLE_PATH.read_bytes()),)


def analyze(
    demo: DemoModel,
    files: tuple[tuple[str, bytes], ...],
    settings: PrepareSettings,
    *,
    source: str,
) -> Run:
    """Read the uploads (the last file, with its SigMF partner if any), prepare
    and classify the recording, and keep it as a run. ValueError describes a
    bad file or bad settings in plain language, never repeating file text."""
    selected, ignored = select_uploads(files)
    if not selected:
        raise ValueError("Send one recording file, or a .sigmf-meta and .sigmf-data pair.")
    recording = load_uploads(selected)
    prepared, truncated, result, spectrogram, loudest = _analyze_recording(
        demo, recording, settings
    )
    run = Run(
        run_id=new_run_id(),
        source=source,
        recording=recording,
        ignored_uploads=ignored,
        settings=settings,
        prepared=prepared,
        truncated_to=truncated,
        result=result,
        spectrogram=spectrogram,
        loudest=loudest,
        expires_at=0.0,
    )
    return store.put(run)


def reprepare(demo: DemoModel, run: Run, settings: PrepareSettings) -> Run:
    """Prepare the run's loaded recording again with other settings; the run
    keeps its id so links to it stay valid."""
    prepared, truncated, result, spectrogram, loudest = _analyze_recording(
        demo, run.recording, settings
    )
    run.settings = settings
    run.prepared = prepared
    run.truncated_to = truncated
    run.result = result
    run.spectrogram = spectrogram
    run.loudest = loudest
    return store.put(run)


def example_info() -> schemas.ExampleInfo:
    return schemas.ExampleInfo(
        file_name=real_example.EXAMPLE_PATH.name,
        title=real_example.TITLE,
        author=real_example.AUTHOR,
        doi=real_example.DOI,
        doi_url=real_example.DOI_URL,
        license=real_example.LICENSE,
        license_url=real_example.LICENSE_URL,
        blog_url=real_example.BLOG_URL,
        sample_rate=real_example.SAMPLE_RATE,
        sample_count=real_example.SAMPLE_COUNT,
        first_sample=real_example.FIRST_SAMPLE,
        last_sample=real_example.LAST_SAMPLE,
        credit_markdown=real_example.CREDIT,
        about_markdown=real_example.ABOUT,
    )


def _features_or_none(window: np.ndarray) -> np.ndarray | None:
    try:
        return extract_features(window)
    except ValueError:
        return None


def window_causes(samples: np.ndarray, starts) -> list[list[str]]:
    """Each window's possible-cause keys (see causes.py)."""
    found = []
    for start in starts:
        window = samples[start : start + SAMPLES_PER_WINDOW]
        found.append([cause.key for cause in possible_causes(window, _features_or_none(window))])
    return found


def to_response(demo: DemoModel, run: Run) -> schemas.AnalyzeResponse:
    result, prepared, spectrogram = run.result, run.prepared, run.spectrogram
    recording = prepared.recording
    causes = window_causes(recording.samples, result.starts)
    uncertain = [keys for label, keys in zip(result.labels, causes) if label == UNCERTAIN]
    rate, center = recording.sample_rate, recording.center_frequency
    order = decision_order(result.classes)
    frequencies, frequency_title = spectrogram.frequency_axis(rate)
    times, time_title = spectrogram.time_axis(rate)
    windows = [
        schemas.WindowDecision(
            number=index + 1,
            start_sample=int(start),
            label=result.labels[index],
            reason=result.reasons[index],
            distance=_finite(result.distances[index]),
            top_score=_finite(np.nanmax(result.scores[index]))
            if not np.isnan(result.scores[index]).all()
            else None,
            scores=[_finite(value) for value in reorder(result.classes, result.scores[index])],
            causes=causes[index],
        )
        for index, start in enumerate(result.starts)
    ]
    return schemas.AnalyzeResponse(
        run_id=run.run_id,
        expires_in_seconds=store.remaining_seconds(run),
        model_version=MODEL_VERSION,
        source=run.source,
        example=example_info() if run.source == "example" else None,
        recording=schemas.RecordingOut(
            source_format=run.recording.source_format,
            sample_count=len(run.recording.samples),
            sample_rate=run.recording.sample_rate,
            center_frequency=run.recording.center_frequency,
            ignored_uploads=run.ignored_uploads,
        ),
        preparation=schemas.PreparationOut(
            settings=schemas.PrepareSettingsIn(
                offset=run.settings.offset,
                bandwidth=run.settings.bandwidth,
                symbol_rate=run.settings.symbol_rate,
                factor=run.settings.factor,
            ),
            steps=list(prepared.steps),
            description=prepared.description,
            ratio=list(prepared.ratio),
            samples_per_symbol=prepared.samples_per_symbol,
            filled_share=prepared.filled_share,
            band_note=prepared.band_note,
            summary=result_summary(prepared),
            truncated_to=run.truncated_to,
        ),
        classes=display_order(result.classes),
        windows=windows,
        counts=[schemas.DecisionCount(label=name, count=count) for name, count in decision_counts(result)],
        ignored_samples=result.ignored_samples,
        timeline_summary=timeline_summary(result.labels, order),
        cause_counts=[
            schemas.CauseCount(key=key, title=title, count=sum(key in keys for keys in uncertain))
            for key, title in CAUSE_TITLES.items()
        ],
        causes_summary=causes_summary(result.labels, causes),
        distance_threshold=demo.model.distance_threshold,
        min_top_score=demo.model.min_top_score,
        spectrogram=schemas.SpectrogramOut(
            frequencies=np.round(np.asarray(frequencies, dtype=float), 6).tolist(),
            frequency_axis_title=frequency_title,
            times=np.round(np.asarray(times, dtype=float), 4).tolist(),
            time_axis_title=time_title,
            levels_db=np.round(spectrogram.levels, 2).tolist(),
            floor_db=FLOOR_DB,
            segment_length=spectrogram.segment_length,
            ignored_samples=spectrogram.ignored_samples,
            summary=spectrogram_summary(spectrogram, rate, center),
        ),
        loudest=[
            schemas.LoudWindow(number=index + 1, level_db=level, phrase=level_phrase(level))
            for index, level in run.loudest
        ],
    )


def window_view(demo: DemoModel, run: Run, number: int) -> schemas.WindowView:
    samples = run.prepared.recording.samples
    count = len(samples) // SAMPLES_PER_WINDOW
    if not 1 <= number <= count:
        raise LookupError(f"This run has windows 1 to {count:,}.")
    start = (number - 1) * SAMPLES_PER_WINDOW
    window = samples[start : start + SAMPLES_PER_WINDOW]
    level = float(window_levels(samples)[number - 1])
    recording = run.prepared.recording
    return schemas.WindowView(
        run_id=run.run_id,
        number=number,
        start_sample=start,
        level_db=level,
        level_phrase=level_phrase(level),
        decision=decision_for(demo.model, window),
        view=signal_view(window, recording.sample_rate, recording.center_frequency),
    )


def csv_for(run: Run) -> str:
    return results_csv(
        run.result,
        (("preparation", run.prepared.description), ("model_version", MODEL_VERSION)),
    )
