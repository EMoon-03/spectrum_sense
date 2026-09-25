"""Short-time spectra (a spectrogram) of a whole I/Q recording.

Only NumPy is used, so the app, the chart summaries, and the tests share it
without loading plotting or machine-learning packages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Samples per FFT segment; neighboring segments overlap by half.
SEGMENT_LENGTH = 256
# At most this many time steps are shown; neighboring segments are averaged to fit.
MAX_TIME_STEPS = 400
# Levels are in dB relative to the strongest cell and never below this floor.
FLOOR_DB = -80.0


@dataclass(frozen=True, eq=False)
class Spectrogram:
    """Power by time step (rows) and frequency (columns)."""

    frequencies: np.ndarray  # cycles per sample, ascending, one per column
    starts: np.ndarray  # first sample of each time step
    stops: np.ndarray  # one past the last sample of each time step
    segments: np.ndarray  # segments averaged into each time step
    power: np.ndarray  # (time steps, frequencies); the strongest cell is 1, or all are 0
    segment_length: int
    ignored_samples: int  # samples after the last complete segment

    @property
    def levels(self) -> np.ndarray:
        """Power in dB relative to the strongest cell, never below FLOOR_DB."""
        return 10 * np.log10(np.maximum(self.power, 10 ** (FLOOR_DB / 10)))

    @property
    def centers(self) -> np.ndarray:
        """The middle sample of each time step (possibly between two samples)."""
        return (self.starts + self.stops - 1) / 2

    def frequency_axis(self, sample_rate: float | None = None) -> tuple[np.ndarray, str]:
        """Column frequencies and an axis title: hertz from the band center when
        the sample rate is known, otherwise cycles per sample."""
        if sample_rate is None:
            return self.frequencies, "Frequency (cycles per sample)"
        return self.frequencies * sample_rate, "Frequency from the band center (Hz)"

    def time_axis(self, sample_rate: float | None = None) -> tuple[np.ndarray, str]:
        """The middle of each time step and an axis title: seconds when the
        sample rate is known and every time is finite, otherwise sample numbers."""
        if sample_rate is not None:
            with np.errstate(over="ignore"):
                seconds = self.centers / sample_rate
            if np.isfinite(seconds).all():
                return seconds, "Time (s)"
        return self.centers, "Sample"


def hann(length: int) -> np.ndarray:
    """The periodic Hann window: copies shifted by half its length add to a constant."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)


def compute_spectrogram(
    samples: np.ndarray,
    *,
    segment_length: int = SEGMENT_LENGTH,
    max_time_steps: int = MAX_TIME_STEPS,
) -> Spectrogram:
    """Hann-windowed FFTs of segments that overlap by half, zero frequency in
    the middle. The power of each group of neighboring segments is averaged so
    that there are at most max_time_steps rows.

    A copy scaled to at most 1 is analyzed. Levels are relative to the
    strongest cell, so extreme amplitudes neither overflow nor change them.
    """
    if segment_length < 2 or segment_length % 2 or max_time_steps < 1:
        raise ValueError("Use an even segment length of at least 2 and at least one time step.")
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim != 1 or len(samples) < segment_length:
        raise ValueError(
            f"Provide a one-dimensional recording of at least {segment_length:,} samples."
        )
    if not np.isfinite(samples).all():
        raise ValueError("The I/Q samples must contain only finite values.")

    hop = segment_length // 2
    count = 1 + (len(samples) - segment_length) // hop
    group = -(-count // max_time_steps)
    first = np.arange(0, count, group)  # the first segment of each time step
    segments = np.diff(np.append(first, count))
    frequencies = np.fft.fftshift(np.fft.fftfreq(segment_length))

    # Scale each component separately, as extract_features does.
    scale = max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))
    if scale == 0:
        power = np.zeros((len(first), segment_length))
    else:
        scaled = np.empty_like(samples)
        scaled.real = samples.real / scale
        scaled.imag = samples.imag / scale
        frames = sliding_window_view(scaled, segment_length)[::hop]
        spectra = np.fft.fft(frames * hann(segment_length), axis=1)
        frame_power = spectra.real**2 + spectra.imag**2
        power = np.add.reduceat(frame_power, first, axis=0) / segments[:, None]
        power = np.fft.fftshift(power, axes=1)
        peak = power.max()
        if peak > 0:  # 0 if every nonzero sample is weighted zero or left out.
            power /= peak
    return Spectrogram(
        frequencies=frequencies,
        starts=first * hop,
        stops=(first + segments - 1) * hop + segment_length,
        segments=segments,
        power=power,
        segment_length=segment_length,
        ignored_samples=len(samples) - ((count - 1) * hop + segment_length),
    )
