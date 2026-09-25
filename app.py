"""Local interactive demo for the SpectrumSense synthetic RF baseline."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, Recording, load_uploads
from signal_config import (
    CLASSES,
    FREQ_OFFSET_RANGE,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
)
from spectrum_sense import extract_features, generate_capture, make_dataset


st.set_page_config(page_title="SpectrumSense Signal Lab", page_icon="📡", layout="wide")

# The browser-side check counts a megabyte as 1,000,000 bytes (1,048,576 on
# Windows). Round up so the widget never rejects a file that a reader accepts.
UPLOAD_WIDGET_MB = math.ceil(MAX_FILE_BYTES / 1_000_000)


@st.cache_resource(show_spinner="Training the synthetic baseline…")
def get_demo_model():
    """Train once per local app process; no user data enter the training set."""
    rng = np.random.default_rng(42)
    windows, labels, groups, _, _ = make_dataset(rng, captures_per_class=60, windows=4)
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
            windows, labels, groups
        )
    )
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    train_features = np.vstack([extract_features(windows[i]) for i in train_idx])
    test_features = np.vstack([extract_features(windows[i]) for i in test_idx])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train_features, labels[train_idx])
    accuracy = float(accuracy_score(labels[test_idx], model.predict(test_features)))
    return model, accuracy, len(set(groups[train_idx])), len(set(groups[test_idx]))


@st.cache_data(max_entries=2, show_spinner=False, scope="session")
def parse_upload(files: tuple[tuple[str, bytes], ...]) -> Recording:
    """Parse (name, data) uploads once per session, not on every widget change.

    Streamlit does not cache raised errors, so an invalid file is rechecked and
    its ValueError reaches the caller on every run.
    """
    return load_uploads(files)


SI_SCALES = ((1e9, "G", "billion"), (1e6, "M", "million"), (1e3, "k", "thousand"))


def format_frequency(hertz: float) -> str:
    """Readable frequency such as 433.92 MHz."""
    for scale, prefix, _ in SI_SCALES:
        if abs(hertz) >= scale:
            return f"{hertz / scale:.6g} {prefix}Hz"
    return f"{hertz:.6g} Hz"


def format_sample_rate(rate: float) -> str:
    """Readable rate such as 2.4 million samples per second."""
    for scale, _, word in SI_SCALES:
        if rate >= scale:
            return f"{rate / scale:.6g} {word} samples per second"
    return f"{rate:.6g} samples per second"


def signal_plots(samples: np.ndarray, section: str) -> None:
    """Hoverable, zoomable time, constellation, and frequency views."""
    time_axis = np.arange(128)
    time_plot = go.Figure()
    time_plot.add_trace(go.Scatter(x=time_axis, y=samples.real[:128], name="I"))
    time_plot.add_trace(go.Scatter(x=time_axis, y=samples.imag[:128], name="Q"))
    time_plot.update_layout(title="I and Q over time", xaxis_title="Sample", yaxis_title="Amplitude", height=300)

    points = samples[::2]
    constellation = go.Figure(
        go.Scattergl(x=points.real, y=points.imag, mode="markers", marker=dict(size=5, opacity=0.5))
    )
    constellation.update_layout(
        title="I/Q plane", xaxis_title="I", yaxis_title="Q", height=300,
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    frequencies = np.fft.fftshift(np.fft.fftfreq(len(samples)))
    # Scale before the FFT, then restore the original magnitude in log space.
    # Keep the noise floor positive even for zero or subnormal input samples.
    fft_scale = max(1.0, np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))
    fft_samples = np.empty_like(samples)
    fft_samples.real = samples.real / fft_scale
    fft_samples.imag = samples.imag / fft_scale
    magnitude = np.abs(np.fft.fftshift(np.fft.fft(fft_samples)))
    spectrum = 20 * (np.log10(magnitude + 1e-8 / fft_scale) + np.log10(fft_scale))
    fft_plot = go.Figure(go.Scatter(x=frequencies, y=spectrum, mode="lines"))
    fft_plot.update_layout(
        title="FFT magnitude", xaxis_title="Cycles per sample", yaxis_title="Magnitude (dB)", height=300
    )

    st.plotly_chart(time_plot, key=f"{section}-time")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(constellation, key=f"{section}-constellation")
    with right:
        st.plotly_chart(fft_plot, key=f"{section}-fft")


def show_prediction(model, samples: np.ndarray, section: str) -> str:
    features = extract_features(samples).reshape(1, -1)
    predicted = str(model.predict(features)[0])
    scores = model.predict_proba(features)[0]
    st.metric("Demo model prediction", predicted)
    score_plot = go.Figure(
        go.Bar(x=model.classes_, y=scores, marker_color=["#466AE8", "#466AE8", "#466AE8"])
    )
    score_plot.update_layout(
        title="Model scores for the three trained classes",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Class",
        height=280,
    )
    st.plotly_chart(score_plot, key=f"{section}-scores")
    st.caption("Scores are model outputs, not calibrated confidence or proof that an unknown signal belongs to a class.")
    return predicted


model, test_accuracy, train_captures, test_captures = get_demo_model()
st.title("📡 SpectrumSense Signal Lab")
st.write("Explore a radio signal, then see how a small machine-learning baseline describes it.")
st.warning(
    "Research demo: the model learned idealized simulated noise, BPSK, and QPSK. "
    "Results on other signals or real recordings have not been validated."
)

lab_tab, upload_tab, about_tab = st.tabs(["Explore a signal", "Try your own recording", "How the model works"])

with lab_tab:
    st.subheader("Create a simulated recording")
    controls, explanation = st.columns([2, 3])
    with controls:
        label = st.selectbox("Signal type", CLASSES, index=1)
        snr = st.slider("Signal-to-noise ratio (dB)", -12.0, 20.0, 6.0, 1.0, disabled=(label == "noise"))
        offset = st.slider("Carrier offset (cycles/sample)", -0.030, 0.030, 0.000, 0.001)
        seed = st.number_input("Example seed", min_value=0, max_value=1_000_000, value=42, step=1)
    with explanation:
        st.markdown(
            "**What the controls mean**\n\n"
            "- **Signal type:** noise alone or bits encoded by phase shifts (BPSK/QPSK).\n"
            "- **SNR:** lower values bury the signal in more noise.\n"
            "- **Carrier offset:** shifts and rotates the received signal.\n"
            "- **Example seed:** change it to generate another random message and noise pattern."
        )
        if label != "noise" and not (SNR_DB_RANGE[0] <= snr <= SNR_DB_RANGE[1]):
            st.info("This SNR is outside the demo model's training range. Watch how the scores change.")
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
    )

with upload_tab:
    st.subheader("Explore your own I/Q recording")
    st.write(
        f"Upload a recording with {SAMPLES_PER_WINDOW:,} to {MAX_SAMPLES:,} I/Q samples: "
        "a CSV, a raw SDR sample file, a NumPy array, or a SigMF recording. "
        "You can try the CSV downloaded from the first tab. "
        f"Every {SAMPLES_PER_WINDOW:,} samples are displayed as a separate window."
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
    )
    if uploads:
        try:
            loaded = parse_upload(tuple((upload.name, upload.getvalue()) for upload in uploads))
        except ValueError as exc:
            st.error(str(exc))
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
            available = len(values) // SAMPLES_PER_WINDOW
            if available > 1:
                window_number = st.slider("Window to inspect", 1, available, 1)
            else:
                window_number = 1
            start = (window_number - 1) * SAMPLES_PER_WINDOW
            window = values[start : start + SAMPLES_PER_WINDOW]
            st.caption(f"Showing window {window_number} of {available} ({SAMPLES_PER_WINDOW} samples)")
            st.info(
                "The prediction is exploratory. The demo model assumes its own synthetic "
                f"timing of {SAMPLES_PER_SYMBOL} samples per symbol, and the file's sample "
                "rate does not change what the model expects. It cannot identify arbitrary "
                "real-world radio traffic."
            )
            try:
                show_prediction(model, window, "uploaded")
            except ValueError as exc:
                st.error(str(exc))
            signal_plots(window, "uploaded")

with about_tab:
    st.subheader("What you are testing")
    a, b, c = st.columns(3)
    a.metric("Training captures", train_captures)
    b.metric("Held-out captures", test_captures)
    c.metric("Synthetic test accuracy", f"{test_accuracy:.1%}")
    st.write(
        f"The baseline was trained on {', '.join(CLASSES)} with {SAMPLES_PER_SYMBOL} "
        "repeated samples per symbol. Separate recording sessions were held out "
        "for testing. Features describe amplitude, phase moments, and correlations; "
        "logistic regression assigns a class."
    )
    st.write(
        "That accuracy measures this simplified generator only. A practical receiver "
        "would need realistic channels, pulse shaping, independent recordings, an "
        "unknown-signal option, and fresh evaluation before making useful claims."
    )
    st.markdown(
        "**Try this:** Lower SNR, shift the carrier beyond the training range, "
        "and change the seed. Record when the prediction changes, then improve the "
        "dataset and baseline in `spectrum_sense.py`."
    )
