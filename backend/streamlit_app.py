"""Local interactive demo for the SpectrumSense synthetic RF baseline."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from explore_url import (
    OFFSET,
    SEED,
    SNR,
    URL_KEYS,
    ExploreSettings,
    explore_params,
    explore_settings,
    parse_explore_params,
)
from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import (
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    Recording,
    load_uploads,
    select_uploads,
)
from prepare import (
    MAX_FACTOR,
    MIN_BANDWIDTH,
    MIN_FACTOR,
    Prepared,
    PrepareSettings,
    level_phrase,
    loudest_windows,
    prepare_recording,
    result_summary,
)
from real_example import ABOUT as EXAMPLE_ABOUT
from real_example import CREDIT as EXAMPLE_CREDIT
from real_example import EXAMPLE_PATH, LICENSE, SAMPLE_COUNT, SAMPLE_RATE
from signal_config import (
    CLASSES,
    DISTANCE_PERCENTILE,
    FREQ_OFFSET_RANGE,
    NOT_CLASSIFIABLE,
    PROBE_KINDS,
    REASON_CLOSE_SCORES,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
    UNCERTAIN,
)
from spectrogram import FLOOR_DB, Spectrogram, compute_spectrogram
from spectrum_sense import (
    DemoModel,
    FileClassification,
    SignalClassifier,
    build_demo_model,
    classify_features,
    classify_recording,
    decision_counts,
    decision_order,
    extract_features,
    generate_capture,
    results_csv,
)
from summaries import (
    constellation_summary,
    decision_summary,
    format_against_limit,
    format_frequency,
    format_sample_rate,
    scores_summary,
    spectrogram_summary,
    spectrum_db,
    spectrum_summary,
    time_summary,
    timeline_summary,
)


st.set_page_config(page_title="SpectrumSense Signal Lab", page_icon="📡", layout="wide")

# The browser-side check counts a megabyte as 1,000,000 bytes (1,048,576 on
# Windows). Round up so the widget never rejects a file that a reader accepts.
UPLOAD_WIDGET_MB = math.ceil(MAX_FILE_BYTES / 1_000_000)
TIME_VIEW_SAMPLES = 128
CONSTELLATION_STEP = 2
EXAMPLE_WINDOWS = 2
# Okabe-Ito blue and vermillion stay distinct with color-vision deficiency. On
# Streamlit's light (#FFFFFF) and dark (#0E1117) backgrounds their contrast is
# 5.2:1 and 3.6:1 (blue) and 3.9:1 and 4.9:1 (vermillion), above the 3:1 that
# WCAG asks of graphics. The gray reaches 3.9:1 and 4.8:1.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAY = "#808080"
# Timeline markers differ in shape as well as position, so color is never needed.
MARKER_SYMBOLS = {
    **dict(zip(CLASSES, ("circle", "square", "diamond"))),
    UNCERTAIN: "x",
    NOT_CLASSIFIABLE: "triangle-up",
}
MARKER_COLORS = {UNCERTAIN: VERMILLION, NOT_CLASSIFIABLE: GRAY}
# Cividis grows steadily lighter from dark blue to yellow (OKLab lightness 0.26
# to 0.92), so levels read by lightness alone, with any color vision, and in
# light or dark themes.
SPECTROGRAM_COLORSCALE = "Cividis"
UNCERTAIN_NOTE = (
    '"Uncertain" comes from a simple distance-and-score heuristic that was tested only '
    f"on synthetic probe signals ({', '.join(PROBE_KINDS[:-1])}, and an unmodulated "
    f"{PROBE_KINDS[-1]}). It is not validated unknown-signal detection: it can miss "
    "unfamiliar signals, and very-low-SNR signals can still be called noise."
)
# Session State keys of the Explore controls, by ExploreSettings field.
CONTROL_KEYS = {
    "signal": "explore_signal",
    "snr_db": "explore_snr",
    "offset": "explore_offset",
    "seed": "explore_seed",
}
# The settings that the URL describes, and notes about a link's unusable values.
URL_SETTINGS = "explore_url_settings"
URL_NOTES = "explore_url_notes"
# After Back or Forward, Streamlit reruns the app with the query string that the
# app last wrote, not the one in the address bar, so the app cannot see the change.
SHARE_HINT = (
    "**To share this exact example,** copy the address from your browser's address bar. "
    "The Back and Forward buttons change the address but not the example, so after "
    "using them, change a control to update the address, or reload the page to show "
    "the example in the address."
)
# Every ASCII punctuation character; Markdown shows one as is after a backslash.
MARKDOWN_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")
# The upload tab analyzes either an uploaded file or the bundled real recording.
SOURCE_KEY = "recording_source"
UPLOAD_KEY = "recording_upload"
SOURCE_UPLOAD = "A file you upload"
SOURCE_EXAMPLE = f"A real satellite recording (QPSK, {LICENSE})"
UPLOAD_TIP = (
    "Tip: the example from **Download an example CSV** above always loads, so you can "
    "compare your file with it."
)
# Session State keys of the preparation controls, with the defaults that change nothing.
PREPARE_DEFAULTS = {
    "prepare_offset": 0.0,
    "prepare_bandwidth": 0.0,
    "prepare_symbol_rate": 0.0,
    "prepare_factor": 1.0,
}
# The recording that the preparation controls were set for; a new one resets them.
PREPARED_FOR = "prepared_for"
WINDOW_KEY = "window_number"
# The number of windows the slider had; with a different number, it starts at window 1.
WINDOW_COUNT = "window_count"
LOUD_KEY = "loud_window"
PREPARE_LABEL = "Prepare the recording (optional)"
RESET_LABEL = "Reset the preparation"
PREPARE_INTRO = (
    f"The demo model learned simulated signals close to 0 Hz, with {SAMPLES_PER_SYMBOL} "
    "samples per symbol and noise that fills the whole recorded band. These optional "
    "steps, applied in this order, can bring a recording closer to that:\n\n"
    "1. **Carrier offset to remove** shifts the recording so that the signal's frequency, "
    "which you can read from the spectrogram below, moves to 0 Hz. On its own, a shift "
    "changes no answer, because the model's features already remove a steady frequency "
    "offset; it matters before the next two steps, which keep only the band around 0 Hz.\n"
    "2. **Bandwidth to keep** runs a low-pass filter that keeps only this band around "
    "0 Hz and removes the rest.\n"
    f"3. **Symbol rate** resamples the recording to {SAMPLES_PER_SYMBOL} samples per symbol "
    "when the file states its sample rate; otherwise, a **resampling factor** multiplies "
    "the number of samples.\n\n"
    "Every result below, including the per-window CSV, then uses the prepared recording. "
    "Filtering and resampling to a higher rate leave part of the band without noise, which "
    'the model never saw, so it often answers "uncertain"; resampling to a lower rate '
    "keeps the band full."
)


@st.cache_resource(show_spinner="Training the synthetic baseline…")
def get_demo_model() -> DemoModel:
    """Train, calibrate, and measure once per app process (a fraction of a
    second); no user data enter the training set."""
    return build_demo_model()


@st.cache_data(max_entries=2, show_spinner="Reading the recording…", scope="session")
def load_upload(files: tuple[tuple[str, bytes], ...]) -> tuple[Recording, str]:
    """Read (name, data) uploads or the bundled recording once per session, not
    on every widget change, and return it with a digest of the files that
    names it in analyze_recording's cache.

    No other cached function takes the uploads, so each run hashes them once.
    Streamlit does not cache raised errors, so an invalid file is rechecked and
    its ValueError reaches the caller on every run.
    """
    digest = hashlib.sha256()
    for name, data in files:
        for part in (name.encode("utf-8", "surrogatepass"), data):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return load_uploads(files), digest.hexdigest()


@st.cache_data(
    max_entries=4, show_spinner="Preparing and classifying the recording…", scope="session"
)
def analyze_recording(
    _recording: Recording, key: str, settings: PrepareSettings, _model: SignalClassifier
) -> tuple[Prepared, FileClassification, Spectrogram, list[tuple[int, float]]]:
    """Prepare the recording that load_upload named with key, decide every
    window, compute the spectrogram, and rank the loudest windows, once per
    recording and settings. The model never changes in a process.
    """
    prepared = prepare_recording(_recording, settings)
    samples = prepared.recording.samples
    return (
        prepared,
        classify_recording(_model, samples),
        compute_spectrogram(samples),
        loudest_windows(samples),
    )


def as_plain_markdown(text: str) -> str:
    """Markdown that shows text as it is: characters that cannot be sent to the
    browser (lone surrogates) become escape codes, and every punctuation
    character that Markdown or Streamlit could read as a link, formatting, or
    an icon is escaped. Error messages go through this before st.error."""
    text = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return MARKDOWN_PUNCTUATION.sub(r"\\\1", text)


@st.cache_data(show_spinner=False)
def example_csv() -> str:
    """A fixed simulated BPSK recording that the upload tab always accepts."""
    capture, _, _ = generate_capture("BPSK", np.random.default_rng(2024), windows=EXAMPLE_WINDOWS)
    return to_csv(capture.ravel())


def load_settings_from_url() -> None:
    """Read the Explore settings from the URL on a session's first run only, and
    give each control without a value its setting. Later runs trust the
    controls, so a URL that the browser has not updated yet cannot reset them."""
    if URL_SETTINGS not in st.session_state:
        params = {key: values for key in URL_KEYS if (values := st.query_params.get_all(key))}
        settings, notes = parse_explore_params(params)
        st.session_state[URL_SETTINGS] = settings
        st.session_state[URL_NOTES] = notes
    settings = st.session_state[URL_SETTINGS]
    for field, key in CONTROL_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = getattr(settings, field)


def save_settings_to_url(settings: ExploreSettings) -> None:
    """Write changed settings to the URL, which updates the browser's address bar
    without a rerun. Other query parameters are kept, and a link is not rewritten
    just because it was opened."""
    if settings == st.session_state[URL_SETTINGS]:
        return
    st.query_params.update(explore_params(settings))
    st.session_state[URL_SETTINGS] = settings
    st.session_state[URL_NOTES] = []


def signal_plots(
    samples: np.ndarray,
    section: str,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
) -> None:
    """Hoverable, zoomable time, constellation, and frequency views, each with a
    text summary directly underneath. I and Q differ in line style, not only color.

    Summaries are body text, not captions: Streamlit fades captions to a contrast
    of 3.7:1 on its light theme, below the 4.5:1 that WCAG asks of text.
    """
    shown = samples[:TIME_VIEW_SAMPLES]
    time_axis = np.arange(len(shown))
    time_plot = go.Figure()
    time_plot.add_trace(
        go.Scatter(x=time_axis, y=shown.real, name="I", line=dict(color=BLUE, width=2))
    )
    time_plot.add_trace(
        go.Scatter(
            x=time_axis, y=shown.imag, name="Q", line=dict(color=VERMILLION, width=2, dash="dash")
        )
    )
    time_plot.update_layout(
        title="I (solid) and Q (dashed) over time", xaxis_title="Sample",
        yaxis_title="Amplitude", height=300,
    )

    points = samples[::CONSTELLATION_STEP]
    # A translucent fill (BLUE at 35% opacity) shows where points pile up; the
    # solid outline keeps every point visible on light and dark backgrounds.
    constellation = go.Figure(
        go.Scattergl(
            x=points.real, y=points.imag, mode="markers",
            marker=dict(size=6, color="rgba(0, 114, 178, 0.35)", line=dict(color=BLUE, width=1)),
        )
    )
    constellation.update_layout(
        title="I/Q plane", xaxis_title="I", yaxis_title="Q", height=300,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    frequencies, levels = spectrum_db(samples)
    fft_plot = go.Figure(
        go.Scatter(x=frequencies, y=levels, mode="lines", line=dict(color=BLUE, width=1.5))
    )
    fft_plot.update_layout(
        title="FFT magnitude", xaxis_title="Cycles per sample", yaxis_title="Magnitude (dB)", height=300
    )

    st.plotly_chart(time_plot, key=f"{section}-time")
    st.markdown(time_summary(samples, TIME_VIEW_SAMPLES))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(constellation, key=f"{section}-constellation")
        st.markdown(constellation_summary(samples, CONSTELLATION_STEP))
    with right:
        st.plotly_chart(fft_plot, key=f"{section}-fft")
        st.markdown(spectrum_summary(samples, sample_rate, center_frequency))


def show_prediction(model: SignalClassifier, samples: np.ndarray, section: str) -> None:
    """The decision (a class or "uncertain" with its reason) and a score chart."""
    all_scores, (decision,) = classify_features(model, extract_features(samples))
    scores = all_scores[0]
    shown = "Uncertain" if decision.label == UNCERTAIN else decision.label
    st.metric("Demo model prediction", shown)
    st.write(decision_summary(decision, model.distance_threshold, model.min_top_score))
    st.markdown(UNCERTAIN_NOTE)
    score_plot = go.Figure(
        go.Bar(
            x=model.classes, y=scores, marker_color=BLUE,
            text=[f"{score:.2f}" for score in scores], textposition="outside", cliponaxis=False,
        )
    )
    score_plot.update_layout(
        title="Model scores for the three trained classes",
        # Headroom above 1 keeps the value printed on a full-height bar visible.
        yaxis=dict(range=[0, 1.15], tickvals=[0, 0.25, 0.5, 0.75, 1], title="Score"),
        xaxis_title="Class",
        height=280,
    )
    st.plotly_chart(score_plot, key=f"{section}-scores")
    st.markdown(
        f"{scores_summary(model.classes, scores)} Scores are model outputs, not calibrated "
        "confidence or proof that an unknown signal belongs to a class."
    )


def window_hover(result: FileClassification, index: int, min_top_score: float) -> str:
    start = int(result.starts[index])
    where = f"Window {index + 1:,}: samples {start:,} to {start + SAMPLES_PER_WINDOW - 1:,}"
    label = result.labels[index]
    if label == NOT_CLASSIFIABLE:
        return f"{where}<br>{label}: {result.reasons[index]}"
    best = int(np.argmax(result.scores[index]))
    score = float(result.scores[index, best])
    shown = f"{score:.2f}"
    if result.reasons[index] == REASON_CLOSE_SCORES:
        shown, _ = format_against_limit(score, min_top_score)  # Never "0.60" when below 0.60.
    detail = f"leading class {result.classes[best]}, score {shown}"
    if label == UNCERTAIN:
        detail = f"{result.reasons[index]}; {detail}"
    return f"{where}<br>{label} ({detail})"


def show_spectrogram(
    spectrogram: Spectrogram, sample_rate: float | None, center_frequency: float | None
) -> None:
    """A waterfall of the whole recording (frequency across, time running down)
    with a labeled color bar, cell values on hover, and a text summary underneath."""
    frequencies, frequency_title = spectrogram.frequency_axis(sample_rate)
    times, time_title = spectrogram.time_axis(sample_rate)
    frequency_hover = (
        "%{x:.4f} cycles per sample" if sample_rate is None else "%{x:.3s}Hz from the band center"
    )
    time_hover = "sample %{y:,.0f}" if time_title == "Sample" else "%{y:.3s}s"
    heatmap = go.Figure(
        go.Heatmap(
            x=frequencies, y=times, z=spectrogram.levels.astype(np.float32),
            colorscale=SPECTROGRAM_COLORSCALE, zmin=FLOOR_DB, zmax=0,
            colorbar=dict(
                title=dict(text="Level (dB, 0 = strongest)", side="right"),
                tickvals=np.linspace(FLOOR_DB, 0, 5),
            ),
            hovertemplate=(
                f"Frequency: {frequency_hover}<br>Time: {time_hover}<br>"
                "Level: %{z:.1f} dB<extra></extra>"
            ),
        )
    )
    heatmap.update_layout(
        title="Spectrogram of the whole recording (time runs down)", height=440,
        xaxis_title=frequency_title, yaxis=dict(title=time_title, autorange="reversed"),
    )
    st.write(
        "Each row is a moment in the recording, from top to bottom, and each column is "
        "a frequency. Lighter cells hold more power; hover over a cell to read its level."
    )
    st.plotly_chart(heatmap, key="uploaded-spectrogram")
    st.markdown(spectrogram_summary(spectrogram, sample_rate, center_frequency))


def show_file_results(result: FileClassification, min_top_score: float, preparation: str) -> None:
    """A timeline of the decision for every window with its text summary, a
    count table, a note on ignored trailing samples, and a CSV download that
    also describes how the recording was prepared."""
    order = decision_order(result.classes)
    numbers = np.arange(1, len(result.labels) + 1)
    timeline = go.Figure()
    for name in order:
        rows = [index for index, label in enumerate(result.labels) if label == name]
        if rows:
            timeline.add_trace(
                go.Scatter(
                    x=numbers[rows], y=[name] * len(rows), mode="markers", name=name,
                    marker=dict(
                        symbol=MARKER_SYMBOLS.get(name, "circle"), size=10,
                        color=MARKER_COLORS.get(name, BLUE),
                    ),
                    hovertext=[window_hover(result, index, min_top_score) for index in rows],
                    hoverinfo="text",
                )
            )
    timeline.update_layout(
        title="Decision for each window", height=300, showlegend=False,
        xaxis=dict(title="Window", tickformat="d"),
        yaxis=dict(
            title="Decision", type="category", categoryorder="array", categoryarray=list(order)
        ),
    )
    st.plotly_chart(timeline, key="uploaded-timeline")
    st.markdown(timeline_summary(result.labels, order))

    counts = decision_counts(result)
    total = len(result.labels)
    st.dataframe(
        {
            "Decision": [name for name, _ in counts],
            "Windows": [count for _, count in counts],
            "Share of windows": [f"{count / total:.1%}" for _, count in counts],
        },
        hide_index=True,
    )
    if result.ignored_samples:
        st.markdown(
            f"The last {result.ignored_samples:,} samples, after window {total:,}, were "
            f"ignored because they do not fill a complete {SAMPLES_PER_WINDOW:,}-sample window."
        )
    st.download_button(
        "Download per-window results as CSV",
        data=results_csv(result, (("preparation", preparation),)),
        file_name="spectrum_sense_windows.csv",
        mime="text/csv",
        help=(
            "One row per window: its number, first sample, decision, reason, distance "
            "from the nearest trained class, the score for each class, and how the "
            "recording was prepared. Windows and samples count the prepared recording."
        ),
    )


def choose_uploaded_file() -> None:
    """Adding a file switches the upload tab to that file, so the newest choice is analyzed."""
    if st.session_state.get(UPLOAD_KEY):
        st.session_state[SOURCE_KEY] = SOURCE_UPLOAD


def reset_preparation() -> None:
    """Put every preparation control back to its default (also a button callback)."""
    st.session_state.update(PREPARE_DEFAULTS)


def frequency_step(sample_rate: float) -> float:
    """A round step near a hundredth of the sample rate: 1 Hz at 500 samples per second."""
    return 10.0 ** math.floor(math.log10(sample_rate / 100))


def preparation_controls(recording: Recording, key: str) -> PrepareSettings:
    """The preparation inputs, in hertz when the recording states its sample
    rate and in cycles per sample otherwise, and a reset button. A different
    recording starts from the defaults and from its first window."""
    if st.session_state.get(PREPARED_FOR) != key:
        reset_preparation()
        st.session_state[WINDOW_KEY] = 1
        st.session_state[PREPARED_FOR] = key
    # Streamlit forgets the value of an input that a run does not show.
    for name, default in PREPARE_DEFAULTS.items():
        st.session_state.setdefault(name, default)
    rate = recording.sample_rate
    if rate is None:
        unit, limit, step, number_format = "cycles per sample", 0.5, 0.001, "%.4f"
        widths = f"{MIN_BANDWIDTH:g} up to, but not including, 1 cycle per sample"
    else:
        unit, limit, step, number_format = "Hz", rate / 2, frequency_step(rate), "%.2f"
        widths = (
            f"{format_frequency(MIN_BANDWIDTH * rate)} up to, but not including, the sample "
            f"rate of {format_frequency(rate)}"
        )
    st.markdown(PREPARE_INTRO)
    offset_column, bandwidth_column, timing_column = st.columns(3)
    offset = offset_column.number_input(
        f"Carrier offset to remove ({unit})", -limit, limit, step=step, format=number_format,
        key="prepare_offset",
        help=(
            "The signal's frequency relative to the center of the recorded band, as the "
            "spectrogram below shows it. The recording is shifted so that this frequency "
            "moves to 0 Hz, where a filter or a lower sample rate keeps it; 0 leaves the "
            "frequency unchanged."
        ),
    )
    bandwidth = bandwidth_column.number_input(
        f"Bandwidth to keep ({unit})", 0.0, 2 * limit, step=step, format=number_format,
        key="prepare_bandwidth",
        help=(
            "The width of the band kept around 0 Hz, for example a little more than the "
            f"symbol rate. It can be from {widths}; 0 keeps every frequency."
        ),
    )
    if rate is None:
        factor = timing_column.number_input(
            "Resampling factor", MIN_FACTOR, MAX_FACTOR, step=0.1, format="%.3f",
            key="prepare_factor",
            help=(
                "Multiplies the number of samples: 0.4 turns 20 samples per symbol into "
                f"{SAMPLES_PER_SYMBOL}, and 2 turns 4 into {SAMPLES_PER_SYMBOL}. 1 keeps the "
                "recorded timing."
            ),
        )
        symbol_rate = 0.0
    else:
        symbol_rate = timing_column.number_input(
            "Symbol rate (symbols per second)", 0.0, rate * MAX_FACTOR / SAMPLES_PER_SYMBOL,
            step=step, format="%.2f", key="prepare_symbol_rate",
            help=(
                "How many symbols per second the transmitter sends; look it up for the "
                "signal you recorded (the bundled satellite recording uses 125). The "
                f"recording is resampled to {SAMPLES_PER_SYMBOL} samples per symbol; 0 keeps "
                "the recorded timing."
            ),
        )
        factor = 1.0
    st.button(
        RESET_LABEL, on_click=reset_preparation,
        help=(
            "Sets every preparation control back to its default, so the recording is "
            "analyzed as loaded."
        ),
    )
    return PrepareSettings(float(offset), float(bandwidth), float(symbol_rate), float(factor))


def jump_to_loud_window() -> None:
    """Move the window slider to the loud window just chosen, then clear the
    choice so that the list never names a window other than the one shown."""
    number = st.session_state.get(LOUD_KEY)
    if number is not None:
        st.session_state[WINDOW_KEY] = number
        st.session_state[LOUD_KEY] = None


def choose_window(available: int, loud: list[tuple[int, float]]) -> int:
    """A list of the loudest windows that moves the slider, and the slider."""
    levels = {index + 1: level for index, level in loud}
    if levels:
        st.selectbox(
            "Jump to a loud window", list(levels), index=None, key=LOUD_KEY,
            format_func=lambda number: f"Window {number:,}: {level_phrase(levels[number])}",
            placeholder=f"Choose one of the {len(levels)} loudest windows",
            on_change=jump_to_loud_window,
            help=(
                "The windows with the most power, loudest first. In a recording of bursts, "
                "they are the most likely to hold a signal rather than noise alone. Choosing "
                "one moves the slider below to it."
            ),
        )
    # Resampling changes which samples each window holds, so start again at window 1.
    if (
        st.session_state.get(WINDOW_COUNT) != available
        or not 1 <= st.session_state.get(WINDOW_KEY, 1) <= available
    ):
        st.session_state[WINDOW_KEY] = 1
        st.session_state[WINDOW_COUNT] = available
    return st.slider(
        "Window to inspect", 1, available, key=WINDOW_KEY,
        help=(
            f"Each window is {SAMPLES_PER_WINDOW:,} consecutive samples; "
            "choose one to plot and score."
        ),
    )


def show_recording(
    model: SignalClassifier, files: tuple[tuple[str, bytes], ...], error_tip: str = ""
) -> None:
    """Read (name, data) files, then show the file's details and the
    preparation controls, then every window's decision, the spectrogram, and
    one window in detail for the prepared recording. Uploads and the bundled
    recording take exactly these steps."""
    try:
        loaded, key = load_upload(files)
    except ValueError as exc:
        message = as_plain_markdown(str(exc))
        st.error(f"{message}\n\n{error_tip}" if error_tip else message)
        return
    details = [f"**Format:** {loaded.source_format}", f"**Samples:** {len(loaded.samples):,}"]
    if loaded.sample_rate is not None:
        details.append(f"**Sample rate:** {format_sample_rate(loaded.sample_rate)}")
    if loaded.center_frequency is not None:
        details.append(f"**Center frequency:** {format_frequency(loaded.center_frequency)}")
    st.markdown(" · ".join(details))

    problem = None
    with st.expander(PREPARE_LABEL):
        settings = preparation_controls(loaded, key)
        try:
            prepared, file_result, spectrogram, loud = analyze_recording(
                loaded, key, settings, model
            )
        except ValueError as exc:
            problem = as_plain_markdown(str(exc))
        else:
            st.markdown(f"**Result:** {result_summary(prepared)}")
    if problem is not None:
        st.error(
            f"These preparation settings cannot be used. {problem}\n\nChange them under "
            f'"{PREPARE_LABEL}", or choose **{RESET_LABEL}** there.'
        )
        return
    if prepared.steps:
        st.markdown(f"**Prepared recording:** {prepared.description}")
        if prepared.band_note:
            st.info(prepared.band_note)
    st.info(
        "The predictions are exploratory. The demo model assumes its own synthetic "
        f"timing of {SAMPLES_PER_SYMBOL} samples per symbol, and the file's sample "
        "rate does not change what the model expects; to match that timing, resample the "
        f'recording under "{PREPARE_LABEL}". The model cannot identify arbitrary '
        "real-world radio traffic."
    )
    recording = prepared.recording
    st.subheader("Every window")
    show_file_results(file_result, model.min_top_score, prepared.description)

    st.subheader("Frequency content over time")
    show_spectrogram(spectrogram, recording.sample_rate, recording.center_frequency)

    st.subheader("One window in detail")
    available = len(recording.samples) // SAMPLES_PER_WINDOW
    window_number = choose_window(available, loud) if available > 1 else 1
    start = (window_number - 1) * SAMPLES_PER_WINDOW
    window = recording.samples[start : start + SAMPLES_PER_WINDOW]
    st.caption(f"Showing window {window_number} of {available} ({SAMPLES_PER_WINDOW} samples)")
    try:
        show_prediction(model, window, "uploaded")
    except ValueError as exc:
        st.error(as_plain_markdown(str(exc)))
    signal_plots(window, "uploaded", recording.sample_rate, recording.center_frequency)


demo = get_demo_model()
model = demo.model
st.title("SpectrumSense Signal Lab")
st.write("Explore a radio signal, then see how a small machine-learning baseline describes it.")
st.warning(
    "Research demo: the model learned idealized simulated noise, BPSK, and QPSK. "
    "Results on other signals or real recordings have not been validated."
)

lab_tab, upload_tab, about_tab = st.tabs(["Explore a signal", "Try your own recording", "How the model works"])

with lab_tab:
    st.header("Create a simulated recording")
    load_settings_from_url()
    controls, explanation = st.columns([2, 3])
    # The URL sets the controls through Session State, so they take no default.
    with controls:
        label = st.selectbox(
            "Signal type", CLASSES, key=CONTROL_KEYS["signal"],
            help="Noise alone, or random bits sent as two phases (BPSK) or four phases (QPSK).",
        )
        snr = st.slider(
            "Signal-to-noise ratio (dB)", SNR.minimum, SNR.maximum, step=SNR.step,
            key=CONTROL_KEYS["snr_db"], disabled=(label == "noise"),
            help="How strong the signal is compared with the noise; it does not apply to noise alone.",
        )
        offset = st.slider(
            "Carrier offset (cycles/sample)", OFFSET.minimum, OFFSET.maximum, step=OFFSET.step,
            format="%.3f", key=CONTROL_KEYS["offset"],
            help="A frequency mismatch between transmitter and receiver that makes the I/Q points rotate.",
        )
        seed = st.number_input(
            "Example seed", min_value=SEED.minimum, max_value=SEED.maximum, step=SEED.step,
            key=CONTROL_KEYS["seed"],
            help="The same seed always gives the same recording; change it for new random bits and noise.",
        )
    settings = explore_settings(label, snr, offset, seed)
    save_settings_to_url(settings)
    with explanation:
        st.markdown(
            "**What the controls mean**\n\n"
            "- **Signal type:** noise alone or bits encoded by phase shifts (BPSK/QPSK).\n"
            "- **SNR:** lower values bury the signal in more noise.\n"
            "- **Carrier offset:** shifts and rotates the received signal.\n"
            "- **Example seed:** change it to generate another random message and noise pattern."
        )
        st.markdown(SHARE_HINT)
        if st.session_state[URL_NOTES]:
            st.info(
                "Part of this link could not be used as given:\n\n"
                + "\n".join(f"- {note}" for note in st.session_state[URL_NOTES])
            )
        if label != "noise" and not (SNR_DB_RANGE[0] <= settings.snr_db <= SNR_DB_RANGE[1]):
            st.info(
                "This SNR is outside the demo model's training range. Watch how the answer "
                "and scores change."
            )
        if not (FREQ_OFFSET_RANGE[0] <= settings.offset <= FREQ_OFFSET_RANGE[1]):
            st.info("This carrier offset is outside the demo model's training range.")

    recording, _, _ = generate_capture(
        settings.signal, np.random.default_rng(settings.seed), windows=1,
        snr_db_override=settings.snr_db, freq_offset_override=settings.offset,
    )
    selected_samples = recording[0]
    st.markdown(
        f"Actual simulated type: **{settings.signal}** · {len(selected_samples)} I/Q samples"
    )
    show_prediction(model, selected_samples, "lab")
    signal_plots(selected_samples, "lab")
    st.download_button(
        "Download this I/Q example as CSV",
        data=to_csv(selected_samples),
        file_name="spectrum_sense_example.csv",
        mime="text/csv",
        help='Save these samples as a CSV file that you can upload in the "Try your own recording" tab.',
    )

with upload_tab:
    st.header("Explore your own I/Q recording")
    st.write(
        f"Upload a recording with {SAMPLES_PER_WINDOW:,} to {MAX_SAMPLES:,} I/Q samples: "
        "a CSV, a raw SDR sample file, a NumPy array, a two-channel I/Q WAV file, or a "
        "SigMF recording. "
        f"Every complete {SAMPLES_PER_WINDOW:,}-sample window is classified, and you can "
        "inspect any one of them. No recording at hand? Choose the real satellite "
        "recording that comes with the app, or download the example CSV below and "
        "upload it, or use the CSV from the first tab."
    )
    source = st.radio(
        "Recording to analyze", (SOURCE_UPLOAD, SOURCE_EXAMPLE), key=SOURCE_KEY,
        help=(
            "Analyze a file you choose below, or a "
            f"{SAMPLE_COUNT / SAMPLE_RATE:.0f}-second real satellite recording that comes "
            "with the app. Adding a file switches to your file."
        ),
    )
    st.download_button(
        "Download an example CSV",
        data=example_csv(),
        file_name="spectrum_sense_upload_example.csv",
        mime="text/csv",
        help=(
            f"A simulated BPSK recording of {EXAMPLE_WINDOWS * SAMPLES_PER_WINDOW:,} samples "
            f"({EXAMPLE_WINDOWS} windows) that is the same every time."
        ),
    )
    with st.expander("Supported file formats"):
        st.markdown(
            f"- **CSV:** UTF-8 text with the header `I,Q` and one sample per row "
            f"(at most {format_size(MAX_UPLOAD_BYTES)}).\n"
            "- **Raw SDR samples:** interleaved I and Q values, little-endian. "
            "`.cf32`, `.fc32`, or `.cfile` hold 32-bit floats (GNU Radio file sink); "
            "`.cs16` or `.sc16` hold 16-bit integers; `.cs8` holds 8-bit integers "
            "(HackRF); `.cu8` holds unsigned 8-bit values (`rtl_sdr`). Integer "
            "samples are scaled to about -1 to 1.\n"
            "- **NumPy `.npy`:** a one-dimensional complex array, or a real array "
            "with two columns (I and Q).\n"
            "- **WAV:** a `.wav` file with two channels, I (left) and Q (right), the "
            "way KiwiSDR's `kiwirecorder` and PC programs such as SDR# and HDSDR save "
            "I/Q recordings. Samples may be 8-, 16-, 24-, or 32-bit integers or 32- or "
            "64-bit floats. The sample rate comes from the header, which stores whole "
            "hertz, so it may be rounded. The center frequency is shown when the file "
            "has an `auxi` chunk that states it. A one-channel WAV file is ordinary "
            "audio, not I/Q.\n"
            "- **SigMF:** a `.sigmf` archive, or a `.sigmf-meta` file uploaded "
            "together with its `.sigmf-data` file. Data types `cf32_le`, `ci16_le`, "
            "`ci8`, and `cu8` are supported."
        )
    # The uploader is always shown, so its files stay while the bundled recording is analyzed.
    uploads = st.file_uploader(
        "Choose a recording (or a .sigmf-meta and .sigmf-data pair)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        max_upload_size=UPLOAD_WIDGET_MB,
        key=UPLOAD_KEY,
        on_change=choose_uploaded_file,
        help=(
            "Choose one recording file, or both files of a SigMF recording; "
            "the formats and limits are listed above. Adding a file also selects "
            f'"{SOURCE_UPLOAD}" above.'
        ),
    )
    if source == SOURCE_EXAMPLE:
        st.subheader("About the real satellite recording")
        st.markdown(EXAMPLE_CREDIT)
        st.markdown(EXAMPLE_ABOUT)
        if uploads:
            st.info(
                f'Your uploaded file is kept but not analyzed while "{SOURCE_EXAMPLE}" is '
                f'selected. Choose "{SOURCE_UPLOAD}" above to analyze it.'
            )
        try:
            example = EXAMPLE_PATH.read_bytes()
        except OSError:
            st.error(
                "The bundled recording could not be read. This copy of the app may be "
                "missing its examples folder."
            )
        else:
            show_recording(model, ((EXAMPLE_PATH.name, example),))
    elif uploads:
        # The uploader adds each new choice to its list, so read the latest one.
        chosen, ignored = select_uploads([(upload.name, upload) for upload in uploads])
        files = tuple((name, upload.getvalue()) for name, upload in chosen)
        if ignored:
            st.info(
                f"Only the most recently added recording is read, so {ignored:,} earlier "
                f"file{' was' if ignored == 1 else 's were'} ignored. You can remove files "
                "you no longer need from the list above."
            )
        show_recording(model, files, UPLOAD_TIP)

with about_tab:
    st.header("What you are testing")
    a, b, c = st.columns(3)
    a.metric("Training captures", demo.fit_captures)
    b.metric("Calibration captures", demo.calibration_captures)
    c.metric("Held-out captures", demo.test_captures)
    st.metric(
        "Synthetic test accuracy", f"{demo.test_accuracy:.1%}",
        help=(
            "Share of held-out windows whose top-scoring class is correct, without "
            "ever answering uncertain."
        ),
    )
    st.write(
        f"The baseline was trained on {', '.join(CLASSES)} with {SAMPLES_PER_SYMBOL} "
        "repeated samples per symbol. Separate recording sessions were held out "
        "for testing. Features describe amplitude, phase moments, and correlations; "
        "logistic regression assigns a class."
    )
    st.write(
        "That accuracy measures this simplified generator only. A practical receiver "
        "would need realistic channels, pulse shaping, independent recordings, a "
        "validated unknown-signal option, and fresh evaluation before making useful claims."
    )

    st.subheader('When the model answers "uncertain"')
    st.markdown(
        f"- **Unlike the training data:** the window's features are more than "
        f"{model.distance_threshold:.2f} from every trained class. This Mahalanobis "
        "distance counts how many typical spreads of a class's training features the "
        "window lies from that class's mean, using a slightly shrunk covariance. "
        f"The limit is the {DISTANCE_PERCENTILE:g}th percentile of the distances "
        f"of {demo.calibration_captures} calibration captures, which were training captures "
        "held out of fitting.\n"
        f"- **Scores are close:** the top score is below {model.min_top_score:.2f}. This "
        "weaker check catches some windows that fall between trained classes."
    )
    st.write(
        f"On {demo.test_windows:,} held-out windows of noise, BPSK, and QPSK, "
        f"{demo.false_reject_rate:.1%} were answered uncertain; the other windows were "
        f"{demo.accepted_accuracy:.1%} accurate. Both rates depend on SNR: uncertain "
        "answers and errors are most common near the bottom of the training range. At "
        f"{SNR_DB_RANGE[0]:g} dB, most BPSK examples in the Explore tab are answered "
        "uncertain, and some of the others are wrong."
    )
    st.dataframe(
        {
            "Held-out windows": [rates.group for rates in demo.rates_by_snr],
            "Windows": [rates.windows for rates in demo.rates_by_snr],
            "Answered uncertain": [
                f"{rates.false_reject_rate:.1%}" for rates in demo.rates_by_snr
            ],
            "Accuracy of the other windows": [
                "none answered" if math.isnan(rates.accepted_accuracy)
                else f"{rates.accepted_accuracy:.1%}"
                for rates in demo.rates_by_snr
            ],
        },
        hide_index=True,
    )
    snrs = sorted(demo.probe_flag_rates)
    st.dataframe(
        {
            "Probe signal (never trained on)": list(PROBE_KINDS),
            **{
                f"Answered uncertain at {snr:g} dB SNR": [
                    f"{demo.probe_flag_rates[snr][kind]:.0%}" for kind in PROBE_KINDS
                ]
                for snr in snrs
            },
        },
        hide_index=True,
    )
    st.markdown(
        f"Each rate uses {demo.probe_windows} simulated windows per probe type and SNR "
        "from a fixed seed. Neither the probes nor the held-out captures were used to "
        f"choose the limits. {UNCERTAIN_NOTE}"
    )
    st.markdown(
        "**Try this:** Lower SNR, shift the carrier beyond the training range, "
        "and change the seed. Record when the prediction changes or becomes uncertain, "
        "then improve the dataset and baseline in `spectrum_sense.py`."
    )
