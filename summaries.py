"""Plain-language text alternatives for the app's charts.

Each summary is one or two sentences with the numbers a sighted reader takes
from a chart. Summaries of the samples never name a modulation; only summaries
of the model's output name its classes. Only NumPy and the standard library
are used.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from signal_config import REASON_UNFAMILIAR, UNCERTAIN


_SI_SCALES = ((1e9, "G", "billion"), (1e6, "M", "million"), (1e3, "k", "thousand"))
# Spectrum level added to every bin so silent bins stay finite in dB.
SPECTRUM_FLOOR = 1e-8
# The timeline summary lists at most this many runs of equal decisions.
MAX_LISTED_RUNS = 6


def _round_significant(value: float, digits: int) -> float:
    """Round before choosing a unit, so 999,999.9 Hz becomes 1 MHz, not 1e+03 kHz."""
    return float(f"{value:.{digits}g}")


def format_frequency(hertz: float, digits: int = 6) -> str:
    """Readable frequency such as 433.92 MHz."""
    hertz = _round_significant(hertz, digits)
    for scale, prefix, _ in _SI_SCALES:
        if abs(hertz) >= scale:
            return f"{hertz / scale:.{digits}g} {prefix}Hz"
    return f"{hertz:.{digits}g} Hz"


def format_sample_rate(rate: float) -> str:
    """Readable rate such as 2.4 million samples per second."""
    rate = _round_significant(rate, 6)
    for scale, _, word in _SI_SCALES:
        if rate >= scale:
            return f"{rate / scale:.6g} {word} samples per second"
    return f"{rate:.6g} samples per second"


def _as_samples(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Provide a non-empty, one-dimensional array of I/Q samples.")
    if not np.isfinite(array).all():
        raise ValueError("The I/Q samples must contain only finite values.")
    return array


def _peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _amplitude(scaled: float, scale: float) -> str:
    """Three significant digits of scaled * scale, without forming a product that
    could overflow: 0.707, 12.3, or 2.54e+308."""
    if scaled == 0 or scale == 0:
        return "0"
    exponent = math.log10(scaled) + math.log10(scale)
    if -3 <= exponent < 3:
        return f"{_round_significant(scaled * scale, 3):g}"
    power = math.floor(exponent)
    mantissa = round(10 ** (exponent - power), 2)
    if mantissa >= 10:
        mantissa, power = mantissa / 10, power + 1
    return f"{mantissa:.2f}e{power:+03d}"


def _rms(values: np.ndarray) -> str:
    scale = _peak(values)
    if scale == 0:
        return "0"
    return _amplitude(float(np.sqrt(np.mean((values / scale) ** 2))), scale)


def scores_summary(classes: Sequence[str], scores: Sequence[float]) -> str:
    """Scores from highest to lowest: "Scores: BPSK 0.91, QPSK 0.07, noise 0.02."."""
    pairs = list(zip(classes, (float(score) for score in scores), strict=True))
    ranked = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    return "Scores: " + ", ".join(f"{name} {score:.2f}" for name, score in ranked) + "."


def time_summary(samples: np.ndarray, shown: int) -> str:
    """Describe the time plot: which samples it shows and the RMS of I and Q."""
    samples = _as_samples(samples)
    if shown < 1:
        raise ValueError("The time plot must show at least one sample.")
    span = samples[:shown]
    if len(span) == len(samples):
        coverage = f"The time plot shows all {len(span):,} samples."
    else:
        coverage = f"The time plot shows the first {len(span):,} of {len(samples):,} samples."
    return (
        f"{coverage} Over that span, the RMS amplitude is {_rms(span.real)} for I "
        f"and {_rms(span.imag)} for Q."
    )


def constellation_summary(samples: np.ndarray, step: int = 1) -> str:
    """Describe the I/Q plane: how many points it shows and how far from the
    origin they lie. It does not say which modulation the points resemble."""
    samples = _as_samples(samples)
    if step < 1:
        raise ValueError("The step between plotted samples must be at least 1.")
    points = samples[::step]
    shown = f"The I/Q plane shows {len(points):,} points"
    if step > 1:
        shown += f" (one of every {step} samples)"
    scale = max(_peak(points.real), _peak(points.imag))
    if scale == 0:
        return f"{shown}, all at the origin because every sample is zero."
    # Scaled distances are at most sqrt(2), so they never overflow.
    distance = np.hypot(points.real / scale, points.imag / scale)
    mean = float(np.mean(distance))
    variation = float(np.std(distance)) / mean
    return (
        f"{shown}. Their distance from the origin averages {_amplitude(mean, scale)}, "
        f"with a coefficient of variation of {variation:.2f} "
        "(0 would mean every point lies on one circle)."
    )


def spectrum_db(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies in cycles per sample (zero in the middle) and FFT magnitudes
    in dB, exactly as the app plots them.

    The FFT runs on a copy scaled to at most 1 and the level is restored in log
    space, so extreme amplitudes never overflow. SPECTRUM_FLOOR keeps zero or
    subnormal input finite.
    """
    samples = _as_samples(samples)
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(samples)))
    scale = max(1.0, _peak(samples.real), _peak(samples.imag))
    scaled = np.empty_like(samples)
    scaled.real = samples.real / scale
    scaled.imag = samples.imag / scale
    magnitude = np.abs(np.fft.fftshift(np.fft.fft(scaled)))
    levels = 20 * (np.log10(magnitude + SPECTRUM_FLOOR / scale) + np.log10(scale))
    return frequencies, levels


def spectrum_summary(
    samples: np.ndarray,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
) -> str:
    """Describe the plotted spectrum: where its strongest bin lies and how far
    that bin stands above the median bin.

    A sample rate adds the offset in hertz; a center frequency adds the
    approximate radio frequency of that bin.
    """
    samples = _as_samples(samples)
    frequencies, levels = spectrum_db(samples)
    bins = f"{len(levels):,} frequency bins"
    if not samples.any():
        return f"Spectrum of {bins}: flat, because every sample is zero."
    peak = int(np.argmax(levels))
    above = float(levels[peak] - np.median(levels))
    if above < 0.05:
        return f"Spectrum of {bins}: flat; no bin is more than 0.1 dB above the median bin."

    cycles = float(frequencies[peak])
    where = f"{round(cycles, 3) + 0.0:.3f} cycles per sample"  # + 0.0 turns -0.0 into 0.0.
    if sample_rate is not None:
        offset = cycles * sample_rate
        if offset == 0:
            relation = "the center of the recorded band"
        else:
            side = "above" if offset > 0 else "below"
            relation = f"{format_frequency(abs(offset), 3)} {side} the center of the recorded band"
        if center_frequency is not None and math.isfinite(center_frequency + offset):
            relation += f", about {format_frequency(center_frequency + offset)}"
        where += f" ({relation})"
    return (
        f"Spectrum of {bins}: the strongest bin is at {where}. "
        f"It is {above:.1f} dB above the median bin."
    )


def decision_summary(decision, distance_threshold: float, min_top_score: float) -> str:
    """Explain a spectrum_sense.Decision: the answer, the reason for an
    uncertain one, and the leading class with its score."""
    distance = f"{decision.distance:.1f}"
    if decision.label != UNCERTAIN:
        return (
            f"The model names {decision.label}: the window is {distance} from the nearest "
            f"trained class (limit {distance_threshold:.1f}) and the top score is "
            f"{decision.top_score:.2f} (minimum {min_top_score:.2f})."
        )
    if decision.reason == REASON_UNFAMILIAR:
        why = (
            f"the window is {decision.reason}: it is {distance} from the nearest trained "
            f"class, beyond the limit of {distance_threshold:.1f}"
        )
    else:
        why = (
            f"the {decision.reason}: the top score is {decision.top_score:.2f}, below "
            f"the minimum of {min_top_score:.2f}"
        )
    return (
        f"Uncertain because {why}. The leading candidate is {decision.leading} "
        f"with a score of {decision.top_score:.2f}."
    )


def _windows(count: int) -> str:
    return f"{count:,} window" + ("" if count == 1 else "s")


def _span(first: int, last: int) -> str:
    return f"window {first:,}" if first == last else f"windows {first:,} to {last:,}"


def timeline_summary(labels: Sequence[str], order: Sequence[str]) -> str:
    """Describe the decision timeline: how many windows got each decision
    (listed in the given order) and where the decision changes."""
    labels = [str(label) for label in labels]
    if not labels:
        raise ValueError("The timeline needs at least one window.")
    runs: list[list] = []  # [decision, first window, last window], numbered from 1
    for number, label in enumerate(labels, start=1):
        if runs and runs[-1][0] == label:
            runs[-1][2] = number
        else:
            runs.append([label, number, number])
    names = [*(name for name in order if name in labels), *sorted(set(labels) - set(order))]
    counts = ", ".join(f"{labels.count(name):,} {name}" for name in names)
    text = f"The timeline shows {_windows(len(labels))}: {counts}."
    if len(labels) == 1:
        return text
    if len(runs) == 1:
        return text + " All windows have the same decision."
    if len(runs) <= MAX_LISTED_RUNS:
        listed = "; ".join(f"{_span(first, last)}: {label}" for label, first, last in runs)
        return f"{text} In order, {listed}."
    label, first, last = max(runs, key=lambda run: run[2] - run[1])  # The first longest run.
    return (
        f"{text} The decision changes {len(runs) - 1:,} times; the longest run is "
        f"{_span(first, last)} ({label})."
    )
