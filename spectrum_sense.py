"""Generate simple synthetic I/Q captures and evaluate an RF/ML baseline.

Each capture is independent; several windows from one capture form one group.
All figures and metrics are written under --output-dir (default: results/).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ("noise", "BPSK", "QPSK")
SAMPLES_PER_SYMBOL = 8
SAMPLES_PER_WINDOW = 1024
SNR_DB_RANGE = (-2.0, 14.0)
FREQ_OFFSET_RANGE = (-0.008, 0.008)  # cycles per sample, relative to baseband


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

        start = int(rng.integers(0, SAMPLES_PER_SYMBOL))
        signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)[start : start + n]
        phase = float(rng.uniform(0, 2 * np.pi))
        signal = signal * np.exp(1j * (phase + 2 * np.pi * offset * np.arange(n)))
        noise = complex_gaussian(rng, n)
        # Unit signal power. SNR sets the noise power relative to the signal.
        samples = signal + noise * 10 ** (-snr_db / 20)

    # Keep power from trivially revealing the label/SNR.
    samples = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
    return samples.reshape(windows, SAMPLES_PER_WINDOW), snr_db, offset


def extract_features(window: np.ndarray) -> np.ndarray:
    """Phase-invariant moments and short-lag correlation of an I/Q window."""
    n = len(window)
    # Estimate carrier rotation from adjacent samples, then remove it. For pure
    # noise this estimate is arbitrary, which is fine for this baseline.
    lag_one = np.mean(window[1:] * np.conj(window[:-1]))
    freq_est = np.angle(lag_one) / (2 * np.pi)
    x = window * np.exp(-2j * np.pi * freq_est * np.arange(n))
    power = np.mean(np.abs(x) ** 2) + 1e-12
    amplitude = np.abs(x)
    # Symbol-spaced phase differences distinguish binary from quadrature
    # phase transitions while discarding a constant carrier phase.
    symbol_difference = window[SAMPLES_PER_SYMBOL:] * np.conj(
        window[:-SAMPLES_PER_SYMBOL]
    )
    return np.array(
        [
            np.abs(np.mean(x**2)) / power,
            np.abs(np.mean(x**4)) / power**2,
            float(np.std(amplitude) / (np.mean(amplitude) + 1e-12)),
            float(np.mean(amplitude**4) / power**2),
            np.abs(lag_one) / power,
            np.abs(np.mean(x[SAMPLES_PER_SYMBOL:] * np.conj(x[:-SAMPLES_PER_SYMBOL])))
            / power,
            np.abs(np.mean(symbol_difference**2)) / power**2,
            np.abs(np.mean(symbol_difference**4)) / power**4,
        ],
        dtype=float,
    )


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


def plot_examples(windows: np.ndarray, labels: np.ndarray, output: Path) -> None:
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
