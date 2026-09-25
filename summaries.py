"""Plain-language text alternatives for the app's charts.

Each summary is a few short sentences with the numbers a sighted reader takes
from a chart. Summaries of the samples never name a modulation; only summaries
of the model's output name its classes. Only NumPy, the standard library, and
the NumPy-only spectrogram module are used.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from signal_config import REASON_UNFAMILIAR, UNCERTAIN
from spectrogram import FLOOR_DB, Spectrogram


_SI_SCALES = ((1e9, "G", "billion"), (1e6, "M", "million"), (1e3, "k", "thousand"))
# Spectrum level added to every bin so silent bins stay finite in dB.
SPECTRUM_FLOOR = 1e-8
# The timeline summary lists at most this many runs of equal decisions.
MAX_LISTED_RUNS = 6
# The spectrogram summary calls the overall level steady when every time step
# is within this many dB of the loudest one (3 dB is half the power).
STEADY_LEVEL_DB = 3.0
# A time step has a clear peak when its strongest bin is at least this many dB
# above the step's median bin. Noise alone rarely reaches it.
CLEAR_PEAK_DB = 12.0
# The strongest frequency counts as moving when it changes by more than this
# share of the band. BPSK and QPSK spread over about a quarter of the band, so
# their strongest bin wanders by less than this from one time step to the next.
PEAK_MOVE_SHARE = 1 / 8
# The most decimal places used to tell a value from the limit it is compared with.
MAX_DECIMALS = 12


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


def format_duration(seconds: float) -> str:
    """Readable time such as 21.3 milliseconds."""
    seconds = _round_significant(seconds, 3)
    if seconds == 0:
        return "0 seconds"
    for scale, unit in ((1.0, "second"), (1e-3, "millisecond"), (1e-6, "microsecond")):
        if abs(seconds) >= scale:
            break
    else:
        scale, unit = 1e-9, "nanosecond"
    number = f"{seconds / scale:,g}"
    return f"{number} {unit}" + ("" if number == "1" else "s")


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


def _frequency_phrase(
    cycles: float, sample_rate: float | None, center_frequency: float | None
) -> str:
    """A frequency in cycles per sample, plus its offset in hertz when the
    sample rate is known and its radio frequency when the center is known."""
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
    return where


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

    where = _frequency_phrase(float(frequencies[peak]), sample_rate, center_frequency)
    return (
        f"Spectrum of {bins}: the strongest bin is at {where}. "
        f"It is {above:.1f} dB above the median bin."
    )


def _decibels_below(ratio: float) -> float:
    """How far a power ratio of at most 1 lies below 1, in dB, at most -FLOOR_DB."""
    return -10 * math.log10(max(ratio, 10 ** (FLOOR_DB / 10))) + 0.0  # Never -0.0.


def _amount(decibels: float) -> str:
    return f"at least {-FLOOR_DB:g} dB" if decibels >= -FLOOR_DB else f"{decibels:.1f} dB"


def _time_step_position(spectrogram: Spectrogram, row: int, sample_rate: float | None) -> str:
    center = float(spectrogram.centers[row])
    if sample_rate is not None and math.isfinite(center / sample_rate):
        return format_duration(center / sample_rate)
    return f"sample {math.floor(center + 0.5):,}"


def _lower_median(values: Sequence[int]) -> int:
    """The median of whole numbers, taking the lower middle value, so it is a bin."""
    return sorted(values)[(len(values) - 1) // 2]


def _peak_track(spectrogram: Spectrogram, sample_rate: float | None) -> str:
    """Describe how the strongest frequency of each time step changes: whether
    it stays put, moves gradually (a sweep), or jumps (for example, hops)."""
    power = spectrogram.power
    steps, bins = power.shape
    peaks = power.argmax(axis=1)
    peak_power = power[np.arange(steps), peaks]
    clear = (peak_power > 0) & (
        peak_power >= np.median(power, axis=1) * 10 ** (CLEAR_PEAK_DB / 10)
    )
    track = [int(peak) for peak in peaks[clear]]
    count = len(track)
    rule = f"a bin at least {CLEAR_PEAK_DB:g} dB above the step's median bin"
    if count == 0:
        return f" No time step has a clear strongest frequency ({rule})."
    if 2 * count < steps:
        return (
            f" Most time steps have no clear strongest frequency: only {count:,} of "
            f"{steps:,} {'has' if count == 1 else 'have'} {rule}."
        )
    if count == steps:
        lead = " The strongest frequency of each time step"
    else:
        lead = (
            f" In the {count:,} of {steps:,} time steps with a clear peak ({rule}), "
            "the strongest frequency"
        )

    def where(position: int) -> str:
        return _frequency_phrase(float(spectrogram.frequencies[position]), sample_rate, None)

    move = max(1, int(bins * PEAK_MOVE_SHARE))
    ordered = sorted(track)
    low, high = ordered[count // 10], ordered[(9 * count) // 10]
    if high - low <= move:
        verb = "stays at" if ordered[0] == ordered[-1] else "stays near"
        return f"{lead} {verb} {where(_lower_median(track))}."

    third = -(-count // 3)  # At least one time step.
    start = _lower_median(track[:third])
    middle = _lower_median(track[count // 3 : count - count // 3])
    end = _lower_median(track[-third:])
    jumps = sum(abs(after - before) > move for before, after in zip(track, track[1:]))
    if jumps == 0:
        return (
            f"{lead} moves gradually: it is {where(start)} near the start, "
            f"{where(middle)} in the middle, and {where(end)} near the end."
        )
    groups = [[ordered[0]]]
    for position in ordered[1:]:
        if position - groups[-1][-1] > move:
            groups.append([])
        groups[-1].append(position)
    if len(groups) == 2:
        if track[0] in groups[1]:
            groups.reverse()
        first, second = (_lower_median(group) for group in groups)
        if jumps == 1:
            return f"{lead} changes abruptly once, from {where(first)} to {where(second)}."
        return (
            f"{lead} switches abruptly {jumps:,} times between {where(first)} "
            f"and {where(second)}."
        )
    return (
        f"{lead} changes abruptly {jumps:,} times: it is {where(start)} near the start, "
        f"{where(middle)} in the middle, and {where(end)} near the end."
    )


def spectrogram_summary(
    spectrogram: Spectrogram,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
) -> str:
    """Describe the spectrogram: its layout, the strongest frequency of the
    time-averaged spectrum, how the strongest frequency of each time step
    changes, and how much the overall level changes between time steps. Every
    number comes from the plotted power.
    """
    power = spectrogram.power
    steps, bins = power.shape
    segments = spectrogram.segments
    length = f"{spectrogram.segment_length:,}-sample"
    if segments.max() == 1:
        layout = f"each is one {length} segment, overlapping the next by half"
    else:
        layout = f"each averages {segments[0]:,} overlapping {length} segments"
        if segments[-1] != segments[0]:
            layout += f" (the last averages {segments[-1]:,})"
    text = (
        f"The spectrogram shows {bins:,} frequency bins across and "
        f"{steps:,} time step{'' if steps == 1 else 's'} down; {layout}."
    )
    if spectrogram.ignored_samples:
        text += (
            f" The last {spectrogram.ignored_samples:,} samples, which do not fill a "
            "segment, are left out."
        )
    if not power.any():
        return (
            f"{text} Every cell is at the {FLOOR_DB:g} dB floor because no segment holds "
            "any power, for example because every sample is zero."
        )

    # Each row is the mean of its segments, so weighting the rows by their
    # segment counts gives the mean over all segments.
    spectrum = segments @ power / segments.sum()
    peak = int(np.argmax(spectrum))
    above = _decibels_below(float(np.median(spectrum) / spectrum[peak]))
    if above < 0.05:
        text += (
            " Averaged over time, the spectrum is flat; no bin is more than 0.1 dB above "
            "the median bin."
        )
    else:
        where = _frequency_phrase(
            float(spectrogram.frequencies[peak]), sample_rate, center_frequency
        )
        text += (
            f" Averaged over time, the strongest frequency is {where}, "
            f"{_amount(above)} above the median bin."
        )
    if steps == 1:
        return text

    text += _peak_track(spectrogram, sample_rate)
    totals = power.sum(axis=1)
    loudest, quietest = int(np.argmax(totals)), int(np.argmin(totals))
    change = round(_decibels_below(float(totals[quietest] / totals[loudest])), 1)
    if change < STEADY_LEVEL_DB:
        return (
            f"{text} The overall level is steady over time: every time step is within "
            f"{change:.1f} dB of the loudest."
        )
    return (
        f"{text} The overall level changes over time: the quietest time step, around "
        f"{_time_step_position(spectrogram, quietest, sample_rate)}, is {_amount(change)} "
        f"below the loudest, around {_time_step_position(spectrogram, loudest, sample_rate)}."
    )


def format_against_limit(value: float, limit: float, decimals: int = 2) -> tuple[str, str]:
    """A value and the limit it is compared with, each with the given number of
    decimal places, or with more when the two differ but would print the same.

    Rounding never reverses their order, so a value past a limit never appears
    equal to it: 0.597 and 0.6 print as "0.597" and "0.600", not "0.60" twice.
    """
    for places in range(decimals, MAX_DECIMALS + 1):
        shown, bound = f"{value:.{places}f}", f"{limit:.{places}f}"
        if shown != bound or value == limit:
            break
    return shown, bound


def decision_summary(decision, distance_threshold: float, min_top_score: float) -> str:
    """Explain a spectrum_sense.Decision: the answer, the reason for an
    uncertain one, and the leading class with its score.

    Limits have two decimal places, as in the About tab, and each value has
    enough places to differ from the limit it is compared with.
    """
    distance, distance_limit = format_against_limit(decision.distance, distance_threshold)
    score, score_minimum = format_against_limit(decision.top_score, min_top_score)
    if decision.label != UNCERTAIN:
        return (
            f"The model names {decision.label}: the window is {distance} from the nearest "
            f"trained class (limit {distance_limit}) and the top score is {score} "
            f"(minimum {score_minimum})."
        )
    if decision.reason == REASON_UNFAMILIAR:
        score = f"{decision.top_score:.2f}"  # Not compared with the minimum here.
        why = (
            f"the window is {decision.reason}: it is {distance} from the nearest trained "
            f"class, beyond the limit of {distance_limit}"
        )
    else:
        why = (
            f"the {decision.reason}: the top score is {score}, below the minimum of "
            f"{score_minimum}"
        )
    return (
        f"Uncertain because {why}. The leading candidate is {decision.leading} "
        f"with a score of {score}."
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
