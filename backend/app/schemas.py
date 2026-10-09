"""Response and request models.

These are the API's contract. The frontend's TypeScript types are generated
from the OpenAPI document these produce (frontend: `npm run types`), so a
field renamed here fails the frontend build instead of appearing as a
runtime undefined.

NaN never crosses the boundary: a score or distance that cannot be computed
is None (JSON null), and the field's comment says when that happens.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- health and model ------------------------------------------------------


class ModelStatus(BaseModel):
    trained: bool
    version: str = Field(description="Fingerprint of the training settings and model code")


class HealthOut(BaseModel):
    status: Literal["ok", "starting"]
    service: str
    model: ModelStatus


class NumberControlOut(BaseModel):
    """A numeric Explore control, so a client renders the same range, step,
    and default the lab does and never sends a value the API would snap."""

    key: str
    name: str
    minimum: float
    maximum: float
    step: float
    default: float
    decimals: int
    unit: str


class ExploreControls(BaseModel):
    signals: list[str]
    default_signal: str
    snr: NumberControlOut
    offset: NumberControlOut
    seed: NumberControlOut


class TrainingSettings(BaseModel):
    seed: int
    captures_per_class: int
    windows_per_capture: int
    classes: list[str]
    samples_per_symbol: int
    samples_per_window: int
    snr_db_range: list[float]
    freq_offset_range: list[float]


class UncertainLimits(BaseModel):
    """The two checks behind an "uncertain" answer and how they were chosen."""

    distance_threshold: float
    min_top_score: float
    distance_percentile: float
    calibration_fraction: float
    covariance_shrinkage: float


class GroupRatesOut(BaseModel):
    group: str
    windows: int
    false_reject_rate: float
    # None when every window of the group was answered uncertain.
    accepted_accuracy: Optional[float] = None


class ProbeRates(BaseModel):
    snr_db: float
    # Share of windows answered "uncertain", by probe type (8PSK, 16-QAM, tone).
    rates: dict[str, float]


class ModelReport(BaseModel):
    version: str
    training: TrainingSettings
    limits: UncertainLimits
    controls: ExploreControls
    fit_captures: int
    calibration_captures: int
    test_captures: int
    test_windows: int
    test_accuracy: float
    false_reject_rate: float
    accepted_accuracy: Optional[float] = None
    rates_by_snr: list[GroupRatesOut]
    probe_kinds: list[str]
    probe_windows: int
    probe_flag_rates: list[ProbeRates]


# --- decisions and views ----------------------------------------------------


class Decision(BaseModel):
    """One window's answer. `label` is a trained class, "uncertain", or
    "not classifiable". A client should show `reason` whenever it is set,
    and the two limits are included so it can show the comparison too."""

    label: str
    reason: Optional[str] = None
    leading: Optional[str] = Field(None, description="Highest-scoring class, even when uncertain")
    top_score: Optional[float] = None
    distance: Optional[float] = None
    distance_threshold: float
    min_top_score: float
    classes: list[str]
    scores: list[Optional[float]] = Field(description="One per class, in `classes` order")
    summary: str = Field(description="The decision in one or two plain sentences")
    scores_summary: str


class Samples(BaseModel):
    i: list[float]
    q: list[float]


class Spectrum(BaseModel):
    frequencies: list[float] = Field(description="Cycles per sample, zero in the middle")
    levels_db: list[float]


class ViewSummaries(BaseModel):
    """Text equivalents of each chart, for screen readers and for anyone who
    wants the numbers without reading them off a plot."""

    time: str
    constellation: str
    spectrum: str


class SignalView(BaseModel):
    samples: Samples
    time_view_samples: int = Field(description="How many samples the time plot shows")
    constellation_step: int = Field(description="Plot every n-th sample in the I/Q plane")
    spectrum: Spectrum
    summaries: ViewSummaries


# --- explore ----------------------------------------------------------------


class ExploreSettingsOut(BaseModel):
    signal: str
    snr_db: float
    offset: float
    seed: int


class TrainingRangeCheck(BaseModel):
    # None for noise, which has no SNR.
    snr: Optional[bool] = None
    offset: bool


class ExploreResponse(BaseModel):
    settings: ExploreSettingsOut
    notes: list[str] = Field(description="Each query value that was ignored or changed")
    in_training_range: TrainingRangeCheck
    model_version: str
    decision: Decision
    view: SignalView


# --- analyze ----------------------------------------------------------------


class PrepareSettingsIn(BaseModel):
    """Optional preparation; the defaults change nothing. Frequencies are in
    hertz when the recording states its sample rate and in cycles per sample
    otherwise."""

    offset: float = 0.0
    bandwidth: float = 0.0
    symbol_rate: float = 0.0
    factor: float = 1.0


class RecordingOut(BaseModel):
    source_format: str
    sample_count: int
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    ignored_uploads: int = Field(0, description="Earlier files ignored when several were sent")


class PreparationOut(BaseModel):
    settings: PrepareSettingsIn
    steps: list[str]
    description: str = Field(description="Every step in one line, or \"none\"")
    ratio: list[int] = Field(description="Resampling (up, down); (1, 1) without resampling")
    samples_per_symbol: Optional[float] = None
    filled_share: float
    band_note: str
    summary: str
    truncated_to: Optional[int] = Field(
        None, description="Set when only the first n prepared samples were kept"
    )


class WindowDecision(BaseModel):
    number: int = Field(description="Window number, from 1")
    start_sample: int
    label: str
    reason: str
    # None where the window is not classifiable.
    distance: Optional[float] = None
    top_score: Optional[float] = None
    scores: list[Optional[float]]


class DecisionCount(BaseModel):
    label: str
    count: int


class SpectrogramOut(BaseModel):
    frequencies: list[float]
    frequency_axis_title: str
    times: list[float]
    time_axis_title: str
    levels_db: list[list[float]] = Field(description="Rows are time steps, columns frequencies")
    floor_db: float
    segment_length: int
    ignored_samples: int
    summary: str


class LoudWindow(BaseModel):
    number: int
    level_db: float = Field(description="Relative to the median window")
    phrase: str


class ExampleInfo(BaseModel):
    file_name: str
    title: str
    author: str
    doi: str
    doi_url: str
    license: str
    license_url: str
    blog_url: str
    sample_rate: float
    sample_count: int
    first_sample: int
    last_sample: int
    credit_markdown: str
    about_markdown: str


class AnalyzeResponse(BaseModel):
    run_id: str
    expires_in_seconds: int
    model_version: str
    source: Literal["upload", "example"]
    example: Optional[ExampleInfo] = None
    recording: RecordingOut
    preparation: PreparationOut
    classes: list[str]
    windows: list[WindowDecision]
    counts: list[DecisionCount]
    ignored_samples: int = Field(description="Trailing samples after the last complete window")
    timeline_summary: str
    distance_threshold: float
    min_top_score: float
    spectrogram: SpectrogramOut
    loudest: list[LoudWindow]


class WindowView(BaseModel):
    run_id: str
    number: int
    start_sample: int
    level_db: float
    level_phrase: str
    decision: Decision
    view: SignalView


class ErrorOut(BaseModel):
    detail: str
