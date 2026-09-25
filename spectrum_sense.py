"""Generate simple synthetic I/Q captures and evaluate an RF/ML baseline.

Each capture is independent; several windows from one capture form one group.
All figures and metrics are written under --output-dir (default: results/).
The app also uses this module's "uncertain" check and whole-file classification.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from sklearn.covariance import ShrunkCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from signal_config import (
    CALIBRATION_FRACTION,
    CLASSES,
    COVARIANCE_SHRINKAGE,
    DISTANCE_PERCENTILE,
    FREQ_OFFSET_RANGE,
    MIN_TOP_SCORE,
    NOT_CLASSIFIABLE,
    PROBE_KINDS,
    REASON_CLOSE_SCORES,
    REASON_UNFAMILIAR,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
    UNCERTAIN,
)


def complex_gaussian(rng: np.random.Generator, size: int) -> np.ndarray:
    """Unit-power circularly symmetric complex Gaussian noise."""
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)


def generate_capture(
    label: str,
    rng: np.random.Generator,
    windows: int,
    *,
    snr_db_override: float | None = None,
    freq_offset_override: float | None = None,
) -> tuple[np.ndarray, float | None, float]:
    """Make one continuous capture and return windows, SNR, and frequency offset.

    The transmitter is intentionally simplified: repeated ideal symbols instead
    of pulse-shaped symbols. Extend that model after validating the baseline.
    """
    n = windows * SAMPLES_PER_WINDOW
    offset = float(
        rng.uniform(*FREQ_OFFSET_RANGE)
        if freq_offset_override is None
        else freq_offset_override
    )
    snr_db = (
        None
        if label == "noise"
        else float(rng.uniform(*SNR_DB_RANGE) if snr_db_override is None else snr_db_override)
    )

    if label == "noise":
        samples = complex_gaussian(rng, n)
    else:
        n_symbols = n // SAMPLES_PER_SYMBOL + 2
        i = rng.choice((-1.0, 1.0), size=n_symbols)
        if label == "BPSK":
            symbols = i.astype(complex)
        elif label == "QPSK":
            q = rng.choice((-1.0, 1.0), size=n_symbols)
            symbols = (i + 1j * q) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown class: {label}")
        samples = _transmit(symbols, rng, n, offset, snr_db)

    # Keep power from trivially revealing the label/SNR.
    samples = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
    return samples.reshape(windows, SAMPLES_PER_WINDOW), snr_db, offset


def _transmit(
    symbols: np.ndarray, rng: np.random.Generator, n: int, offset: float, snr_db: float
) -> np.ndarray:
    """Repeat unit-power symbols, apply random timing and carrier phase, the
    carrier offset, and noise at snr_db; return n samples before normalization."""
    start = int(rng.integers(0, SAMPLES_PER_SYMBOL))
    signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)[start : start + n]
    phase = float(rng.uniform(0, 2 * np.pi))
    signal = signal * np.exp(1j * (phase + 2 * np.pi * offset * np.arange(n)))
    noise = complex_gaussian(rng, n)
    # Unit signal power. SNR sets the noise power relative to the signal.
    return signal + noise * 10 ** (-snr_db / 20)


def generate_probe_capture(
    kind: str,
    rng: np.random.Generator,
    windows: int,
    *,
    snr_db: float,
    freq_offset_override: float | None = None,
) -> np.ndarray:
    """Windows of a signal type the model never saw: 8PSK, 16-QAM, or an
    unmodulated tone, with the same impairments and power normalization as
    generate_capture. Probes only test the "uncertain" answer; never train on them.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"Unknown probe type: {kind}")
    n = windows * SAMPLES_PER_WINDOW
    offset = float(
        rng.uniform(*FREQ_OFFSET_RANGE)
        if freq_offset_override is None
        else freq_offset_override
    )
    n_symbols = n // SAMPLES_PER_SYMBOL + 2
    if kind == "8PSK":
        symbols = np.exp(2j * np.pi * rng.integers(0, 8, size=n_symbols) / 8)
    elif kind == "16-QAM":
        levels = np.array((-3.0, -1.0, 1.0, 3.0))
        # The 4 x 4 grid has average power 10; dividing by sqrt(10) gives unit power.
        symbols = (rng.choice(levels, n_symbols) + 1j * rng.choice(levels, n_symbols)) / np.sqrt(10)
    else:
        symbols = np.ones(n_symbols, dtype=complex)
    samples = _transmit(symbols, rng, n, offset, float(snr_db))
    samples = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
    return samples.reshape(windows, SAMPLES_PER_WINDOW)


def extract_features(window: np.ndarray) -> np.ndarray:
    """Amplitude- and phase-invariant features of a finite, nonzero I/Q window.

    Invalid windows raise ValueError; the caller's samples are left unchanged.
    """
    window = np.asarray(window, dtype=np.complex128)
    if window.ndim != 1 or window.size <= SAMPLES_PER_SYMBOL:
        raise ValueError(
            f"Provide a one-dimensional window of at least {SAMPLES_PER_SYMBOL + 1} samples."
        )
    if not np.isfinite(window).all():
        raise ValueError("The I/Q window must contain only finite values.")

    # Scale components separately: complex magnitude can overflow near float64's
    # maximum, and complex division can overflow for subnormal scale factors.
    scale = max(np.max(np.abs(window.real)), np.max(np.abs(window.imag)))
    if scale == 0:
        raise ValueError(
            "The I/Q window contains only zeros. Choose a window with nonzero samples."
        )
    normalized = np.empty_like(window)
    normalized.real = window.real / scale
    normalized.imag = window.imag / scale
    normalized /= np.sqrt(np.mean(np.abs(normalized) ** 2))
    window = normalized

    n = len(window)
    # Estimate carrier rotation from adjacent samples, then remove it. For pure
    # noise this estimate is arbitrary, which is fine for this baseline.
    lag_one = np.mean(window[1:] * np.conj(window[:-1]))
    freq_est = np.angle(lag_one) / (2 * np.pi)
    x = window * np.exp(-2j * np.pi * freq_est * np.arange(n))
    power = np.mean(np.abs(x) ** 2)
    amplitude = np.abs(x)
    # Symbol-spaced phase differences distinguish binary from quadrature
    # phase transitions while discarding a constant carrier phase.
    symbol_difference = window[SAMPLES_PER_SYMBOL:] * np.conj(
        window[:-SAMPLES_PER_SYMBOL]
    )
    features = np.array(
        [
            np.abs(np.mean(x**2)) / power,
            np.abs(np.mean(x**4)) / power**2,
            float(np.std(amplitude) / np.mean(amplitude)),
            float(np.mean(amplitude**4) / power**2),
            np.abs(lag_one) / power,
            np.abs(np.mean(x[SAMPLES_PER_SYMBOL:] * np.conj(x[:-SAMPLES_PER_SYMBOL])))
            / power,
            np.abs(np.mean(symbol_difference**2)) / power**2,
            np.abs(np.mean(symbol_difference**4)) / power**4,
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        raise ValueError("Could not compute finite features for this I/Q window.")
    return features


def make_dataset(rng: np.random.Generator, captures_per_class: int, windows: int):
    all_windows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    snrs: list[float] = []
    manifest: list[dict] = []
    capture_id = 0
    for label in CLASSES:
        for _ in range(captures_per_class):
            capture_windows, snr_db, offset = generate_capture(label, rng, windows)
            all_windows.extend(capture_windows)
            labels.extend([label] * windows)
            groups.extend([capture_id] * windows)
            snrs.extend([np.nan if snr_db is None else snr_db] * windows)
            manifest.append(
                {
                    "capture_id": capture_id,
                    "label": label,
                    "snr_db": "" if snr_db is None else f"{snr_db:.2f}",
                    "frequency_offset_cycles_per_sample": f"{offset:.5f}",
                    "windows": windows,
                }
            )
            capture_id += 1
    return np.asarray(all_windows), np.asarray(labels), np.asarray(groups), np.asarray(snrs), manifest


@dataclass(frozen=True, eq=False)
class NoveltyModel:
    """Per-class means and inverse covariances of standardized features."""

    classes: tuple[str, ...]
    means: np.ndarray  # (classes, features)
    precisions: np.ndarray  # (classes, features, features)


def fit_novelty(
    features: np.ndarray, labels: np.ndarray, shrinkage: float = COVARIANCE_SHRINKAGE
) -> NoveltyModel:
    """Fit each class's mean and covariance, shrunk slightly toward a scaled
    identity so that it can always be inverted."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    classes = tuple(str(name) for name in np.unique(labels))
    fits = []
    for name in classes:
        rows = features[labels == name]
        if len(rows) < 2:
            raise ValueError(f"The class {name} needs at least two training windows.")
        fits.append(ShrunkCovariance(shrinkage=shrinkage).fit(rows))
    return NoveltyModel(
        classes,
        np.array([fit.location_ for fit in fits]),
        np.array([fit.precision_ for fit in fits]),
    )


def novelty_distance(novelty: NoveltyModel, features: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from each row of standardized features to the
    nearest class; large values mean the window is unlike the training data."""
    features = np.atleast_2d(np.asarray(features, dtype=float))
    offsets = features[:, None, :] - novelty.means[None, :, :]
    squared = np.einsum("nkd,kde,nke->nk", offsets, novelty.precisions, offsets)
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


@dataclass(frozen=True)
class Decision:
    """The answer for one window: a class, or UNCERTAIN with a reason."""

    label: str
    reason: str | None  # None when a class was chosen
    leading: str  # the class with the highest score, even when uncertain
    top_score: float
    distance: float


def decide(
    classes: Sequence[str],
    scores: Sequence[float],
    distance: float,
    *,
    distance_threshold: float,
    min_top_score: float = MIN_TOP_SCORE,
) -> Decision:
    """Name the top-scoring class unless the window is far from every class
    (checked first) or its top score is low; then answer UNCERTAIN."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or len(scores) != len(classes):
        raise ValueError("Provide one score for each class.")
    best = int(np.argmax(scores))
    leading, top_score, distance = str(classes[best]), float(scores[best]), float(distance)
    # A NaN distance fails this comparison, so it also counts as unfamiliar.
    if not distance <= distance_threshold:
        return Decision(UNCERTAIN, REASON_UNFAMILIAR, leading, top_score, distance)
    if top_score < min_top_score:
        return Decision(UNCERTAIN, REASON_CLOSE_SCORES, leading, top_score, distance)
    return Decision(leading, None, leading, top_score, distance)


@dataclass(frozen=True, eq=False)
class SignalClassifier:
    """The logistic-regression pipeline plus the checks behind "uncertain"."""

    pipeline: Pipeline
    novelty: NoveltyModel
    distance_threshold: float
    min_top_score: float = MIN_TOP_SCORE

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.pipeline.classes_)


def classify_features(
    model: SignalClassifier, features: np.ndarray
) -> tuple[np.ndarray, list[Decision]]:
    """Scores (a row per window, in model.classes order) and a decision per
    window, using one predict_proba call for all rows."""
    features = np.atleast_2d(np.asarray(features, dtype=float))
    scores = model.pipeline.predict_proba(features)
    distances = novelty_distance(model.novelty, model.pipeline[0].transform(features))
    decisions = [
        decide(
            model.classes, row, distance,
            distance_threshold=model.distance_threshold, min_top_score=model.min_top_score,
        )
        for row, distance in zip(scores, distances)
    ]
    return scores, decisions


def train_classifier(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, seed: int
) -> tuple[SignalClassifier, np.ndarray, np.ndarray]:
    """Fit on most training captures and set the distance threshold on the rest.

    No capture (group) is split between the two parts. Returns the model and
    the row indices used for fitting and for calibration.
    """
    features = np.asarray(features, dtype=float)
    labels, groups = np.asarray(labels), np.asarray(groups)
    fit_rows, calibration_rows = next(
        GroupShuffleSplit(n_splits=1, test_size=CALIBRATION_FRACTION, random_state=seed).split(
            features, labels, groups
        )
    )
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(features[fit_rows], labels[fit_rows])
    scaler = pipeline[0]
    novelty = fit_novelty(scaler.transform(features[fit_rows]), labels[fit_rows])
    calibration = novelty_distance(novelty, scaler.transform(features[calibration_rows]))
    threshold = float(np.percentile(calibration, DISTANCE_PERCENTILE))
    return SignalClassifier(pipeline, novelty, threshold), fit_rows, calibration_rows


def uncertain_rates(decisions: Sequence[Decision], truth: Sequence[str]) -> tuple[float, float]:
    """Share of known-class windows answered UNCERTAIN (false rejects) and the
    accuracy of the others (NaN if every window was uncertain)."""
    if len(decisions) == 0 or len(decisions) != len(truth):
        raise ValueError("Provide one true label for each decision.")
    answered = [
        (item.label, str(label)) for item, label in zip(decisions, truth) if item.label != UNCERTAIN
    ]
    rejected = (len(decisions) - len(answered)) / len(decisions)
    accuracy = (
        float(np.mean([guess == label for guess, label in answered])) if answered else float("nan")
    )
    return rejected, accuracy


def probe_flag_rates(
    model: SignalClassifier,
    rng: np.random.Generator,
    *,
    snr_db: float,
    captures: int,
    windows: int,
) -> dict[str, float]:
    """Share of probe windows (signal types never trained on) answered UNCERTAIN."""
    rates = {}
    for kind in PROBE_KINDS:
        probe = np.vstack(
            [generate_probe_capture(kind, rng, windows, snr_db=snr_db) for _ in range(captures)]
        )
        _, decisions = classify_features(model, np.vstack([extract_features(w) for w in probe]))
        rates[kind] = float(np.mean([item.label == UNCERTAIN for item in decisions]))
    return rates


@dataclass(frozen=True)
class GroupRates:
    """How the held-out windows of one group were answered."""

    group: str  # an SNR band of BPSK and QPSK windows, or noise, which has no SNR
    windows: int
    false_reject_rate: float  # answered "uncertain"
    accepted_accuracy: float  # of the other windows; NaN if every window was uncertain


def rates_by_snr(
    decisions: Sequence[Decision], truth: Sequence[str], snrs: Sequence[float], bands: int = 4
) -> tuple[GroupRates, ...]:
    """uncertain_rates for signal windows in equal SNR bands across SNR_DB_RANGE,
    then for noise windows, whose SNR is NaN. Empty groups are left out."""
    snrs, truth = np.asarray(snrs, dtype=float), np.asarray(truth)
    if not len(decisions) == len(truth) == len(snrs):
        raise ValueError("Provide one true label and one SNR for each decision.")
    edges = np.linspace(*SNR_DB_RANGE, bands + 1)
    signals = " and ".join(name for name in CLASSES if name != "noise")
    masks = [
        (f"{signals} at {lower:g} to {upper:g} dB SNR", (snrs >= lower) & (snrs < upper))
        for lower, upper in zip(edges[:-1], edges[1:])
    ]
    masks[-1] = (masks[-1][0], (snrs >= edges[-2]) & (snrs <= edges[-1]))
    masks.append(("Noise (no SNR)", np.isnan(snrs)))
    rates = []
    for group, mask in masks:
        if mask.any():
            chosen = [item for item, keep in zip(decisions, mask) if keep]
            rejected, accuracy = uncertain_rates(chosen, truth[mask])
            rates.append(GroupRates(group, int(mask.sum()), rejected, accuracy))
    return tuple(rates)


@dataclass(frozen=True, eq=False)
class DemoModel:
    """The app's model and the measurements shown with it."""

    model: SignalClassifier
    fit_captures: int
    calibration_captures: int
    test_captures: int
    test_windows: int
    test_accuracy: float  # always taking the top class
    false_reject_rate: float  # held-out known windows answered "uncertain"
    accepted_accuracy: float  # accuracy on the other held-out windows
    rates_by_snr: tuple[GroupRates, ...]  # the same two rates for groups of held-out windows
    probe_windows: int  # per probe type and SNR
    probe_flag_rates: dict[float, dict[str, float]]  # by SNR in dB, then probe type


def build_demo_model(
    seed: int = 42,
    captures_per_class: int = 100,
    windows: int = 4,
    *,
    probe_snrs_db: Sequence[float] = (5.0, 10.0),
    probe_captures: int = 25,
) -> DemoModel:
    """Train and calibrate on synthetic training captures, then measure the
    model on held-out captures and on a fixed set of probe signals.

    Held-out captures and probes are for reporting only; no setting is tuned on them.
    """
    rng = np.random.default_rng(seed)
    all_windows, labels, groups, snrs, _ = make_dataset(rng, captures_per_class, windows)
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            all_windows, labels, groups
        )
    )
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    features = np.vstack([extract_features(window) for window in all_windows])
    model, fit_rows, calibration_rows = train_classifier(
        features[train_idx], labels[train_idx], groups[train_idx], seed=seed
    )
    scores, decisions = classify_features(model, features[test_idx])
    top_class = np.asarray(model.classes)[np.argmax(scores, axis=1)]
    false_rejects, accepted_accuracy = uncertain_rates(decisions, labels[test_idx])
    probe_rng = np.random.default_rng(seed + 1)
    probe_rates = {
        float(snr_db): probe_flag_rates(
            model, probe_rng, snr_db=snr_db, captures=probe_captures, windows=windows
        )
        for snr_db in probe_snrs_db
    }
    return DemoModel(
        model=model,
        fit_captures=len(set(groups[train_idx][fit_rows])),
        calibration_captures=len(set(groups[train_idx][calibration_rows])),
        test_captures=len(set(groups[test_idx])),
        test_windows=len(test_idx),
        test_accuracy=float(np.mean(top_class == labels[test_idx])),
        false_reject_rate=false_rejects,
        accepted_accuracy=accepted_accuracy,
        rates_by_snr=rates_by_snr(decisions, labels[test_idx], snrs[test_idx]),
        probe_windows=probe_captures * windows,
        probe_flag_rates=probe_rates,
    )


@dataclass(frozen=True, eq=False)
class FileClassification:
    """Decisions for every complete window of one recording."""

    classes: tuple[str, ...]  # score column order
    starts: np.ndarray  # first sample of each window
    labels: tuple[str, ...]  # a class, UNCERTAIN, or NOT_CLASSIFIABLE
    reasons: tuple[str, ...]  # why a window is uncertain or not classifiable; "" otherwise
    scores: np.ndarray  # (windows, classes); NaN where not classifiable
    distances: np.ndarray  # NaN where not classifiable
    ignored_samples: int  # samples after the last complete window


def classify_recording(
    model: SignalClassifier, samples: np.ndarray, window_size: int = SAMPLES_PER_WINDOW
) -> FileClassification:
    """Decide every complete window. Windows whose features cannot be computed
    (for example all zeros) are NOT_CLASSIFIABLE instead of stopping the scan."""
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim != 1:
        raise ValueError("Provide a one-dimensional array of I/Q samples.")
    count = len(samples) // window_size
    if count == 0:
        raise ValueError(f"The recording needs at least {window_size:,} samples.")
    starts = np.arange(count) * window_size
    rows, valid, reasons = [], [], [""] * count
    for index, start in enumerate(starts):
        try:
            rows.append(extract_features(samples[start : start + window_size]))
            valid.append(index)
        except ValueError as exc:
            reasons[index] = str(exc)
    labels = [NOT_CLASSIFIABLE] * count
    scores = np.full((count, len(model.classes)), np.nan)
    distances = np.full(count, np.nan)
    if rows:
        valid_scores, decisions = classify_features(model, np.vstack(rows))
        scores[valid] = valid_scores
        for index, item in zip(valid, decisions):
            labels[index] = item.label
            reasons[index] = item.reason or ""
            distances[index] = item.distance
    return FileClassification(
        model.classes, starts, tuple(labels), tuple(reasons), scores, distances,
        len(samples) - count * window_size,
    )


def _display_classes(classes: Sequence[str]) -> tuple[str, ...]:
    """Classes in CLASSES order (noise, BPSK, QPSK), then any others."""
    return (
        *(name for name in CLASSES if name in classes),
        *(name for name in classes if name not in CLASSES),
    )


def decision_order(classes: Sequence[str] = CLASSES) -> tuple[str, ...]:
    """Display order: the classes, then UNCERTAIN and NOT_CLASSIFIABLE."""
    return (*_display_classes(classes), UNCERTAIN, NOT_CLASSIFIABLE)


def decision_counts(result: FileClassification) -> list[tuple[str, int]]:
    """Windows per decision in display order. NOT_CLASSIFIABLE is listed only
    when it occurs; the classes and UNCERTAIN are always listed."""
    counts = [(name, result.labels.count(name)) for name in decision_order(result.classes)]
    return [(name, count) for name, count in counts if count or name != NOT_CLASSIFIABLE]


def results_csv(result: FileClassification) -> str:
    """Per-window results: window number, first sample, decision, reason, the
    distance to the nearest class, and a score per class (blank if not classifiable)."""
    names = _display_classes(result.classes)
    columns = [result.classes.index(name) for name in names]
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ("window", "start_sample", "decision", "reason", "distance",
         *(f"score_{name}" for name in names))
    )
    for index, start in enumerate(result.starts):
        numbers = (result.distances[index], *result.scores[index, columns])
        writer.writerow(
            (index + 1, int(start), result.labels[index], result.reasons[index],
             *("" if np.isnan(value) else f"{value:.6g}" for value in numbers))
        )
    return output.getvalue()


def plot_examples(windows: np.ndarray, labels: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt  # Imported here so the app does not load matplotlib.

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(13, 9), constrained_layout=True)
    for row, label in enumerate(CLASSES):
        x = windows[np.flatnonzero(labels == label)[0]]
        axes[row, 0].plot(x.real[:100], lw=0.8, label="I")
        axes[row, 0].plot(x.imag[:100], lw=0.8, label="Q")
        axes[row, 0].set(ylabel=label, title="First 100 I/Q samples", xlabel="sample")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].scatter(x.real[::3], x.imag[::3], s=3, alpha=0.35)
        axes[row, 1].set(title="Received I/Q plane", xlabel="I", ylabel="Q")
        fft = np.fft.fftshift(np.fft.fft(x))
        freq = np.fft.fftshift(np.fft.fftfreq(len(x)))
        axes[row, 2].plot(freq, 20 * np.log10(np.abs(fft) + 1e-8), lw=0.8)
        axes[row, 2].set(title="FFT magnitude", xlabel="cycles / sample", ylabel="dB")
    fig.suptitle("Synthetic RF samples (examples from independent captures)")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-per-class", type=int, default=60)
    parser.add_argument("--windows-per-capture", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.captures_per_class < 10 or args.windows_per_capture < 2:
        parser.error("Use at least 10 captures per class and 2 windows per capture")

    # The command line only writes image files, so it never needs a display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    windows, labels, groups, snrs, manifest = make_dataset(
        rng, args.captures_per_class, args.windows_per_capture
    )
    plot_examples(windows, labels, args.output_dir / "example_signals.png")

    # Split *recordings*, never individual windows from a recording.
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed).split(
            windows, labels, groups
        )
    )
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(test_groups)

    features = np.vstack([extract_features(w) for w in windows])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[train_idx], labels[train_idx])
    predictions = model.predict(features[test_idx])
    report = classification_report(labels[test_idx], predictions, labels=CLASSES, zero_division=0)
    matrix = confusion_matrix(labels[test_idx], predictions, labels=CLASSES)
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(cmap="Blues", values_format="d")
    plt.title("Test captures only")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    test_snrs = snrs[test_idx]
    snr_lines = []
    for lower, upper in ((-2, 2), (2, 6), (6, 10), (10, 14)):
        mask = (test_snrs >= lower) & (test_snrs < upper)
        if mask.any():
            accuracy = np.mean(labels[test_idx][mask] == predictions[mask])
            snr_lines.append(f"{lower:>3} to {upper:>2} dB: {accuracy:.3f} ({mask.sum()} samples)")

    for item in manifest:
        item["split"] = "train" if item["capture_id"] in train_groups else "test"
    with (args.output_dir / "capture_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    metrics = (
        f"Synthetic-only baseline; seed={args.seed}\n"
        f"Captures: train={len(train_groups)}, test={len(test_groups)}; "
        f"windows per capture={args.windows_per_capture}\n"
        f"SNR: {SNR_DB_RANGE} dB; carrier offsets: {FREQ_OFFSET_RANGE} cycles/sample\n\n"
        + report
        + "\nAccuracy on signal-only test windows by SNR:\n"
        + "\n".join(snr_lines)
        + "\n\nNoise false-alarm rate: "
        + f"{np.mean(predictions[labels[test_idx] == 'noise'] != 'noise'):.3f}\n"
    )
    (args.output_dir / "metrics.txt").write_text(metrics)
    print(metrics)
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
