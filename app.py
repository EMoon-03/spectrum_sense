"""Local interactive demo for the SpectrumSense synthetic RF baseline."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, Recording, load_uploads
from signal_config import (
    CLASSES,
    DISTANCE_PERCENTILE,
    FREQ_OFFSET_RANGE,
    NOT_CLASSIFIABLE,
    PROBE_KINDS,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
    UNCERTAIN,
)
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
    format_frequency,
    format_sample_rate,
    scores_summary,
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
UNCERTAIN_NOTE = (
    '"Uncertain" comes from a simple distance-and-score heuristic that was tested only '
    f"on synthetic probe signals ({', '.join(PROBE_KINDS[:-1])}, and an unmodulated "
    f"{PROBE_KINDS[-1]}). It is not validated unknown-signal detection: it can miss "
    "unfamiliar signals, and very-low-SNR signals can still be called noise."
)


@st.cache_resource(show_spinner="Training the synthetic baseline…")
def get_demo_model() -> DemoModel:
    """Train, calibrate, and measure once per app process (a fraction of a
    second); no user data enter the training set."""
    return build_demo_model()


@st.cache_data(max_entries=2, show_spinner=False, scope="session")
def parse_upload(files: tuple[tuple[str, bytes], ...]) -> Recording:
    """Parse (name, data) uploads once per session, not on every widget change.

    Streamlit does not cache raised errors, so an invalid file is rechecked and
    its ValueError reaches the caller on every run.
    """
    return load_uploads(files)


@st.cache_data(max_entries=2, show_spinner="Classifying every window…", scope="session")
def classify_upload(
    files: tuple[tuple[str, bytes], ...], _samples: np.ndarray, _model: SignalClassifier
) -> FileClassification:
    """Decide every window once per upload, not on every widget change.

    The cache key is the uploaded (name, data) pairs. The samples are parsed
    from them and the model never changes in a process, so neither is hashed.
    """
    return classify_recording(_model, _samples)


@st.cache_data(show_spinner=False)
def example_csv() -> str:
    """A fixed simulated BPSK recording that the upload tab always accepts."""
    capture, _, _ = generate_capture("BPSK", np.random.default_rng(2024), windows=EXAMPLE_WINDOWS)
    return to_csv(capture.ravel())


def signal_plots(
    samples: np.ndarray,
    section: str,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
) -> None:
    """Hoverable, zoomable time, constellation, and frequency views, each with a
    text summary directly underneath. I and Q differ in line style, not only color."""
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
    st.caption(time_summary(samples, TIME_VIEW_SAMPLES))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(constellation, key=f"{section}-constellation")
        st.caption(constellation_summary(samples, CONSTELLATION_STEP))
    with right:
        st.plotly_chart(fft_plot, key=f"{section}-fft")
        st.caption(spectrum_summary(samples, sample_rate, center_frequency))


def show_prediction(model: SignalClassifier, samples: np.ndarray, section: str) -> None:
    """The decision (a class or "uncertain" with its reason) and a score chart."""
    all_scores, (decision,) = classify_features(model, extract_features(samples))
    scores = all_scores[0]
    shown = "Uncertain" if decision.label == UNCERTAIN else decision.label
    st.metric("Demo model prediction", shown)
    st.write(decision_summary(decision, model.distance_threshold, model.min_top_score))
    st.caption(UNCERTAIN_NOTE)
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
    st.caption(
        f"{scores_summary(model.classes, scores)} Scores are model outputs, not calibrated "
        "confidence or proof that an unknown signal belongs to a class."
    )


def window_hover(result: FileClassification, index: int) -> str:
    start = int(result.starts[index])
    where = f"Window {index + 1:,}: samples {start:,} to {start + SAMPLES_PER_WINDOW - 1:,}"
    label = result.labels[index]
    if label == NOT_CLASSIFIABLE:
        return f"{where}<br>{label}: {result.reasons[index]}"
    best = int(np.argmax(result.scores[index]))
    detail = f"leading class {result.classes[best]}, score {result.scores[index, best]:.2f}"
    if label == UNCERTAIN:
        detail = f"{result.reasons[index]}; {detail}"
    return f"{where}<br>{label} ({detail})"


def show_file_results(result: FileClassification) -> None:
    """A timeline of the decision for every window with its text summary, a
    count table, a note on ignored trailing samples, and a CSV download."""
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
                    hovertext=[window_hover(result, index) for index in rows], hoverinfo="text",
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
    st.caption(timeline_summary(result.labels, order))

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
        st.caption(
            f"The last {result.ignored_samples:,} samples, after window {total:,}, were "
            f"ignored because they do not fill a complete {SAMPLES_PER_WINDOW:,}-sample window."
        )
    st.download_button(
        "Download per-window results as CSV",
        data=results_csv(result),
        file_name="spectrum_sense_windows.csv",
        mime="text/csv",
        help=(
            "One row per window: its number, first sample, decision, reason, distance "
            "from the nearest trained class, and the score for each class."
        ),
    )


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
    controls, explanation = st.columns([2, 3])
    with controls:
        label = st.selectbox(
            "Signal type", CLASSES, index=1,
            help="Noise alone, or random bits sent as two phases (BPSK) or four phases (QPSK).",
        )
        snr = st.slider(
            "Signal-to-noise ratio (dB)", -12.0, 20.0, 6.0, 1.0, disabled=(label == "noise"),
            help="How strong the signal is compared with the noise; it does not apply to noise alone.",
        )
        offset = st.slider(
            "Carrier offset (cycles/sample)", -0.030, 0.030, 0.000, 0.001,
            help="A frequency mismatch between transmitter and receiver that makes the I/Q points rotate.",
        )
        seed = st.number_input(
            "Example seed", min_value=0, max_value=1_000_000, value=42, step=1,
            help="The same seed always gives the same recording; change it for new random bits and noise.",
        )
    with explanation:
        st.markdown(
            "**What the controls mean**\n\n"
            "- **Signal type:** noise alone or bits encoded by phase shifts (BPSK/QPSK).\n"
            "- **SNR:** lower values bury the signal in more noise.\n"
            "- **Carrier offset:** shifts and rotates the received signal.\n"
            "- **Example seed:** change it to generate another random message and noise pattern."
        )
        if label != "noise" and not (SNR_DB_RANGE[0] <= snr <= SNR_DB_RANGE[1]):
            st.info(
                "This SNR is outside the demo model's training range. Watch how the answer "
                "and scores change."
            )
        if not (FREQ_OFFSET_RANGE[0] <= offset <= FREQ_OFFSET_RANGE[1]):
            st.info("This carrier offset is outside the demo model's training range.")

    recording, _, _ = generate_capture(
        label, np.random.default_rng(int(seed)), windows=1,
        snr_db_override=snr, freq_offset_override=offset,
    )
    selected_samples = recording[0]
    st.caption(f"Actual simulated type: **{label}** · {len(selected_samples)} I/Q samples")
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
        "a CSV, a raw SDR sample file, a NumPy array, or a SigMF recording. "
        f"Every complete {SAMPLES_PER_WINDOW:,}-sample window is classified, and you can "
        "inspect any one of them. To start, download the example CSV below and upload it, "
        "or use the CSV from the first tab."
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
            "- **SigMF:** a `.sigmf` archive, or a `.sigmf-meta` file uploaded "
            "together with its `.sigmf-data` file. Data types `cf32_le`, `ci16_le`, "
            "`ci8`, and `cu8` are supported."
        )
    uploads = st.file_uploader(
        "Choose a recording (or a .sigmf-meta and .sigmf-data pair)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        max_upload_size=UPLOAD_WIDGET_MB,
        help=(
            "Choose one recording file, or both files of a SigMF recording; "
            "the formats and limits are listed above."
        ),
    )
    if uploads:
        files = tuple((upload.name, upload.getvalue()) for upload in uploads)
        try:
            loaded = parse_upload(files)
        except ValueError as exc:
            st.error(
                f"{exc}\n\nTip: the example from **Download an example CSV** above always "
                "loads, so you can compare your file with it."
            )
        else:
            values = loaded.samples
            details = [f"**Format:** {loaded.source_format}", f"**Samples:** {len(values):,}"]
            if loaded.sample_rate is not None:
                details.append(f"**Sample rate:** {format_sample_rate(loaded.sample_rate)}")
            if loaded.center_frequency is not None:
                details.append(
                    f"**Center frequency:** {format_frequency(loaded.center_frequency)}"
                )
            st.markdown(" · ".join(details))
            st.info(
                "The predictions are exploratory. The demo model assumes its own synthetic "
                f"timing of {SAMPLES_PER_SYMBOL} samples per symbol, and the file's sample "
                "rate does not change what the model expects. It cannot identify arbitrary "
                "real-world radio traffic."
            )
            st.subheader("Every window")
            show_file_results(classify_upload(files, values, model))

            st.subheader("One window in detail")
            available = len(values) // SAMPLES_PER_WINDOW
            if available > 1:
                window_number = st.slider(
                    "Window to inspect", 1, available, 1,
                    help=(
                        f"Each window is {SAMPLES_PER_WINDOW:,} consecutive samples; "
                        "choose one to plot and score."
                    ),
                )
            else:
                window_number = 1
            start = (window_number - 1) * SAMPLES_PER_WINDOW
            window = values[start : start + SAMPLES_PER_WINDOW]
            st.caption(f"Showing window {window_number} of {available} ({SAMPLES_PER_WINDOW} samples)")
            try:
                show_prediction(model, window, "uploaded")
            except ValueError as exc:
                st.error(str(exc))
            signal_plots(window, "uploaded", loaded.sample_rate, loaded.center_frequency)

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
        f"{demo.accepted_accuracy:.1%} accurate."
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
    st.caption(
        f"Each rate uses {demo.probe_windows} simulated windows per probe type and SNR "
        "from a fixed seed. Neither the probes nor the held-out captures were used to "
        f"choose the limits. {UNCERTAIN_NOTE}"
    )
    st.markdown(
        "**Try this:** Lower SNR, shift the carrier beyond the training range, "
        "and change the seed. Record when the prediction changes or becomes uncertain, "
        "then improve the dataset and baseline in `spectrum_sense.py`."
    )
