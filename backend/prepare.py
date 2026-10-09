"""Prepare a recording so that it better matches what the demo model expects:
the signal near 0 Hz, noise that fills the band, and 8 samples per symbol.

Every step is optional. prepare_recording returns the prepared recording and a
plain description of what was done, which the app shows and adds to the
per-window results. loudest_windows ranks windows by power, so the app can
jump to the ones most likely to hold a signal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.signal import fftconvolve, firwin, resample_poly

from iq_csv import MAX_SAMPLES
from iq_formats import Recording
from signal_config import SAMPLES_PER_SYMBOL, SAMPLES_PER_WINDOW
from spectrogram import FLOOR_DB
from summaries import format_frequency, format_sample_rate


# Resampling changes the sample rate by up/down, with both terms at most this.
MAX_RATIO_TERM = 100
# The resampling factor (new rate / old rate) must lie in this range.
MIN_FACTOR = 1 / MAX_RATIO_TERM
MAX_FACTOR = 8.0
# The low-pass filter keeps a band at least this wide, in cycles per sample.
MIN_BANDWIDTH = 0.01
# The filter's edge is about this share of the band wide. A Hamming-windowed
# filter needs about 3.3 / (edge width in cycles per sample) taps.
EDGE_SHARE = 0.1
MAX_FILTER_TAPS = 1001
# Below this share of the prepared band holding recorded content, the app
# warns that the model may answer "uncertain" more often.
FULL_BAND_SHARE = 0.95
# The app lists this many of the loudest windows.
LOUD_WINDOWS = 5
NO_PREPARATION = "none"


@dataclass(frozen=True)
class PrepareSettings:
    """What to do to a recording; the defaults change nothing.

    Frequencies are in hertz when the recording states its sample rate and in
    cycles per sample otherwise.
    """

    offset: float = 0.0  # the signal's frequency, which is moved to 0
    bandwidth: float = 0.0  # the band kept around 0; 0 keeps every frequency
    symbol_rate: float = 0.0  # symbols per second; 0 keeps the timing (needs a sample rate)
    factor: float = 1.0  # otherwise, resample by this factor (new rate / old rate)


@dataclass(frozen=True, eq=False)
class Prepared:
    """A prepared recording and what was done to it."""

    recording: Recording  # the new samples, sample rate, and center frequency
    steps: tuple[str, ...]  # one plain sentence per step, in order; empty if nothing was done
    ratio: tuple[int, int]  # resampling (up, down); (1, 1) without resampling
    samples_per_symbol: float | None  # after preparation, when a symbol rate was given
    # The share of the prepared band that the recording can fill. A filter, or
    # resampling to a higher rate, leaves the rest of the band empty.
    filled_share: float
    band_note: str  # why part of the band is empty and what that means; "" if it is full

    @property
    def description(self) -> str:
        """Every step in one line, or NO_PREPARATION."""
        return " ".join(self.steps) or NO_PREPARATION


def resample_ratio(factor: float) -> tuple[int, int]:
    """The fraction up/down closest to factor with both terms at most
    MAX_RATIO_TERM, in lowest terms; of equally close fractions, the one with
    the smallest terms."""
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError("The resampling factor must be a positive number.")
    best = (math.inf, 1, 1)
    for down in range(1, MAX_RATIO_TERM + 1):
        up = min(max(round(factor * down), 1), MAX_RATIO_TERM)
        error = abs(up / down - factor)
        if error < best[0]:
            best = (error, up, down)
    _, up, down = best
    common = math.gcd(up, down)
    return up // common, down // common


def lowpass_taps(bandwidth: float) -> np.ndarray:
    """A linear-phase low-pass filter with unit gain at 0 that keeps a band
    `bandwidth` cycles per sample wide, centered on 0. It has an odd number
    of taps, so it delays the signal by a whole number of samples."""
    if not MIN_BANDWIDTH <= bandwidth < 1:
        raise ValueError(
            f"The bandwidth must be at least {MIN_BANDWIDTH:g} and less than 1 cycle per sample."
        )
    count = min(math.ceil(3.3 / (EDGE_SHARE * bandwidth)), MAX_FILTER_TAPS)
    count += 1 - count % 2
    return firwin(count, bandwidth / 2, fs=1.0)


def _frequency(value: float, sample_rate: float | None) -> str:
    if sample_rate is None:
        return f"{value:.4g} cycles per sample"
    return format_frequency(value)


def _signed(value: float, sample_rate: float | None) -> str:
    text = _frequency(value, sample_rate)
    return text if text.startswith("-") else f"+{text}"


def _number(value: float) -> str:
    """Up to six significant digits with thousands separators: 2,375 or 0.4."""
    return f"{value:,.6g}"


def _check(settings: PrepareSettings, sample_rate: float | None) -> float:
    """Check settings for a recording with this sample rate and return the
    resampling factor they ask for."""
    values = (settings.offset, settings.bandwidth, settings.symbol_rate, settings.factor)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("Every preparation setting must be a finite number.")
    rate = 1.0 if sample_rate is None else sample_rate
    if abs(settings.offset) > rate / 2:
        raise ValueError(
            f"The carrier offset must be between {_frequency(-rate / 2, sample_rate)} and "
            f"{_frequency(rate / 2, sample_rate)}, the edges of the recorded band."
        )
    if settings.bandwidth and not MIN_BANDWIDTH * rate <= settings.bandwidth < rate:
        raise ValueError(
            f"The bandwidth must be at least {_frequency(MIN_BANDWIDTH * rate, sample_rate)} "
            f"and less than {_frequency(rate, sample_rate)}; 0 keeps every frequency."
        )
    limits = f"between {MIN_FACTOR:g} and {MAX_FACTOR:g}"
    if not settings.symbol_rate:
        if not MIN_FACTOR <= settings.factor <= MAX_FACTOR:
            raise ValueError(f"The resampling factor must be {limits}; 1 keeps the timing.")
        return float(settings.factor)
    if settings.symbol_rate < 0:
        raise ValueError("The symbol rate must be positive, or 0 to keep the timing.")
    if settings.factor != 1:
        raise ValueError("Give a symbol rate or a resampling factor, not both.")
    if sample_rate is None:
        raise ValueError(
            "A symbol rate needs the recording's sample rate, which this file does not "
            "state. Use a resampling factor instead."
        )
    factor = SAMPLES_PER_SYMBOL * settings.symbol_rate / sample_rate
    if not MIN_FACTOR <= factor <= MAX_FACTOR:
        raise ValueError(
            f"At {format_sample_rate(sample_rate)}, {_number(settings.symbol_rate)} symbols "
            f"per second means {_number(sample_rate / settings.symbol_rate)} samples per "
            f"symbol. Reaching {SAMPLES_PER_SYMBOL} would take resampling by {factor:.3g}, "
            f"but the factor must be {limits}."
        )
    return factor


def prepare_recording(
    recording: Recording, settings: PrepareSettings, *, limit_samples: bool = True
) -> Prepared:
    """Shift the carrier offset to 0, keep the band around it, and resample,
    in that order; each step is skipped when its setting is at its default.

    The recording itself is never changed. At most MAX_SAMPLES samples are
    kept, the app's limit, unless limit_samples is False (for evaluating long
    recordings offline), and at least one SAMPLES_PER_WINDOW-sample window
    must remain. Invalid settings raise a plain-language ValueError.
    """
    rate = recording.sample_rate
    factor = _check(settings, rate)
    samples = np.asarray(recording.samples, dtype=np.complex128)
    if samples.ndim != 1 or len(samples) < SAMPLES_PER_WINDOW:
        raise ValueError(f"The recording needs at least {SAMPLES_PER_WINDOW:,} samples.")
    if not np.isfinite(samples).all():
        raise ValueError("The I/Q samples must contain only finite values.")
    up, down = resample_ratio(factor)
    resampled = (up, down) != (1, 1)
    count = -(-len(samples) * up // down)  # resample_poly's output length
    capped = limit_samples and count > MAX_SAMPLES
    if count < SAMPLES_PER_WINDOW:
        raise ValueError(
            f"Resampling by {up}/{down} would leave {count:,} samples, fewer than one "
            f"{SAMPLES_PER_WINDOW:,}-sample window. Resample by a larger factor."
        )

    cycles = 1.0 if rate is None else rate  # settings / cycles is in cycles per sample
    zero = "0" if rate is None else "0 Hz"
    steps: list[str] = []
    center = recording.center_frequency
    width = 1.0  # the band that the recording fills, in cycles per sample
    with np.errstate(over="ignore", invalid="ignore"):
        if settings.offset:
            shift = settings.offset / cycles
            samples = samples * np.exp(-2j * np.pi * shift * np.arange(len(samples)))
            center = None if center is None or rate is None else center + settings.offset
            steps.append(
                f"Moved {_frequency(settings.offset, rate)} to {zero} "
                f"(a shift of {_signed(-settings.offset, rate)})."
            )
        if settings.bandwidth:
            width = settings.bandwidth / cycles
            taps = lowpass_taps(width)
            # Mirrored samples at each end keep the level steady up to the edges.
            padded = np.pad(samples, len(taps) // 2, mode="reflect")
            samples = fftconvolve(padded, taps, mode="valid")
            steps.append(
                f"Kept a band {_frequency(settings.bandwidth, rate)} wide around {zero} "
                f"with a low-pass filter of {len(taps):,} taps."
            )
        if resampled:
            if capped:
                # Resample only what is kept, plus the reach of resample_poly's
                # filter (10 * max(up, down) taps on each side at the higher rate).
                reach = -(-10 * max(up, down) // up) + 1
                samples = samples[: -(-MAX_SAMPLES * down // up) + reach]
            samples = resample_poly(samples, up, down, padtype="reflect")
        if capped:
            samples = samples[:MAX_SAMPLES]
        if not np.isfinite(samples).all():
            raise ValueError(
                "The recording's values are too large to prepare: shifting or filtering them "
                "would exceed the largest floating-point number. Scale the recording down."
            )

    new_rate = None if rate is None else rate * up / down
    samples_per_symbol = None
    if settings.symbol_rate:
        samples_per_symbol = new_rate / settings.symbol_rate
        timing = (
            f"{_number(samples_per_symbol)} samples per symbol at "
            f"{_number(settings.symbol_rate)} symbols per second"
        )
        if resampled:
            steps.append(
                f"Resampled by {up}/{down} from {format_sample_rate(rate)} to "
                f"{format_sample_rate(new_rate)}, which gives {timing}."
            )
        else:
            steps.append(f"Not resampled: the recording already has {timing}.")
    elif resampled:
        closest = "" if up / down == factor else f", the closest to {_number(factor)}"
        rates = (
            "" if rate is None
            else f" from {format_sample_rate(rate)} to {format_sample_rate(new_rate)}"
        )
        steps.append(f"Resampled by {up}/{down} (a factor of {up / down:.4g}{closest}){rates}.")
    if capped:
        steps.append(f"Kept the first {MAX_SAMPLES:,} of {count:,} samples, the app's limit.")
    if not steps:
        return Prepared(recording, (), (1, 1), None, 1.0, "")
    prepared = replace(recording, samples=samples, sample_rate=new_rate, center_frequency=center)
    filled = min(width * down / up, 1.0)
    return Prepared(
        prepared, tuple(steps), (up, down), samples_per_symbol, filled,
        _band_note(filled, filtered=width < min(up / down, 1.0), upsampled=up > down),
    )


def _band_note(filled: float, *, filtered: bool, upsampled: bool) -> str:
    if filled >= FULL_BAND_SHARE:
        return ""
    causes = [
        cause
        for cause, applies in (("the filter", filtered), ("the higher sample rate", upsampled))
        if applies
    ]
    return (
        f"Only about {filled:.0%} of the prepared band holds the recording's signal and "
        f"noise; {' and '.join(causes)} left the rest empty. The model learned noise that "
        'fills the whole band, so expect more "uncertain" answers.'
    )


def result_summary(prepared: Prepared) -> str:
    """The prepared recording's length, sample rate, and samples per symbol."""
    recording = prepared.recording
    text = f"{len(recording.samples):,} samples"
    if recording.sample_rate is not None:
        text += f" at {format_sample_rate(recording.sample_rate)}"
    if prepared.samples_per_symbol is not None:
        text += (
            f", {_number(prepared.samples_per_symbol)} samples per symbol "
            f"(the model expects {SAMPLES_PER_SYMBOL})"
        )
    elif recording.sample_rate is not None:
        text += "; enter the symbol rate to see the samples per symbol"
    return text + "."


def window_levels(samples: np.ndarray, window_size: int = SAMPLES_PER_WINDOW) -> np.ndarray:
    """The mean power of each complete window in dB relative to the median
    window. Power is computed on a copy scaled to at most 1, so extreme
    amplitudes cannot overflow, and no window counts as more than -FLOOR_DB
    below the loudest."""
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Provide a one-dimensional array of finite I/Q samples.")
    count = len(samples) // window_size
    if count == 0:
        raise ValueError(f"The recording needs at least {window_size:,} samples.")
    windows = samples[: count * window_size].reshape(count, window_size)
    scale = max(np.max(np.abs(windows.real)), np.max(np.abs(windows.imag)))
    if scale == 0:
        return np.zeros(count)
    power = np.mean((windows.real / scale) ** 2 + (windows.imag / scale) ** 2, axis=1)
    power = np.maximum(power, power.max() * 10 ** (FLOOR_DB / 10))
    return 10 * np.log10(power / np.median(power))


def loudest_windows(
    samples: np.ndarray, count: int = LOUD_WINDOWS, window_size: int = SAMPLES_PER_WINDOW
) -> list[tuple[int, float]]:
    """Up to count (window index from 0, level in dB relative to the median
    window) pairs, loudest first. Windows of equal power keep their order, and
    windows without any power are left out."""
    samples = np.asarray(samples, dtype=np.complex128)
    levels = window_levels(samples, window_size)
    silent = ~samples[: len(levels) * window_size].reshape(len(levels), window_size).any(axis=1)
    order = [int(index) for index in np.argsort(-levels, kind="stable") if not silent[index]]
    return [(index, float(levels[index])) for index in order[:count]]


def level_phrase(level: float) -> str:
    """A window's level relative to the median window in words, to 0.1 dB."""
    shown = round(level, 1)
    if shown == 0:
        return "the same power as the median window"
    side = "above" if shown > 0 else "below"
    return f"{abs(shown):.1f} dB {side} the median window"
