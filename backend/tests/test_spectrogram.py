"""Tests for the whole-recording spectrogram."""

from pathlib import Path
import subprocess
import sys
import unittest
import warnings

import numpy as np

from spectrogram import (
    FLOOR_DB,
    MAX_TIME_STEPS,
    SEGMENT_LENGTH,
    compute_spectrogram,
    hann,
)
from spectrum_sense import generate_capture


LARGEST = float(np.finfo(np.float64).max)
SMALLEST = float.fromhex("0x0.0000000000001p-1022")
HOP = SEGMENT_LENGTH // 2


def tone(cycles: float, n: int) -> np.ndarray:
    return np.exp(2j * np.pi * cycles * np.arange(n))


def segment_count(n: int) -> int:
    return 1 + (n - SEGMENT_LENGTH) // HOP


class SpectrogramTestCase(unittest.TestCase):
    def compute_checked(self, samples: np.ndarray, **options):
        """compute_spectrogram with warnings and numeric problems turned into errors.

        Tiny values may underflow harmlessly, as in the feature tests.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                spectrogram = compute_spectrogram(samples, **options)
                levels = spectrogram.levels
        self.assertTrue(np.isfinite(spectrogram.power).all())
        self.assertTrue(np.isfinite(levels).all())
        self.assertGreaterEqual(levels.min(), FLOOR_DB)
        self.assertLessEqual(levels.max(), 0)
        return spectrogram


class ComputeTests(SpectrogramTestCase):
    def test_tone_appears_at_its_frequency_bin_in_every_time_step(self):
        n = 16 * SEGMENT_LENGTH
        for bin_offset in (0, 32, -50, 127, -128):
            with self.subTest(bin=bin_offset):
                spectrogram = self.compute_checked(tone(bin_offset / SEGMENT_LENGTH, n))
                column = bin_offset + SEGMENT_LENGTH // 2
                self.assertEqual(spectrogram.frequencies[column], bin_offset / SEGMENT_LENGTH)
                np.testing.assert_array_equal(
                    np.argmax(spectrogram.power, axis=1), np.full(len(spectrogram.power), column)
                )
                np.testing.assert_allclose(spectrogram.levels[:, column], 0, atol=1e-9)

    def test_tone_between_bins_peaks_at_the_nearest_bin(self):
        spectrogram = self.compute_checked(tone(0.1, 4096))
        peak = int(np.argmax(spectrogram.power.mean(axis=0)))
        nearest = round(0.1 * SEGMENT_LENGTH) / SEGMENT_LENGTH
        self.assertEqual(spectrogram.frequencies[peak], nearest)

    def test_frequencies_run_from_minus_half_with_zero_in_the_middle(self):
        spectrogram = self.compute_checked(tone(0.2, 1024))
        frequencies = spectrogram.frequencies
        self.assertEqual(len(frequencies), SEGMENT_LENGTH)
        self.assertEqual((frequencies[0], frequencies[SEGMENT_LENGTH // 2]), (-0.5, 0.0))
        self.assertTrue((np.diff(frequencies) > 0).all())

    def test_zero_input_gives_finite_levels_at_the_floor(self):
        spectrogram = self.compute_checked(np.zeros(1024, dtype=complex))
        self.assertFalse(spectrogram.power.any())
        np.testing.assert_array_equal(spectrogram.levels, FLOOR_DB)

    def test_silent_stretches_sit_at_the_floor(self):
        samples = tone(0.25, 4096)
        samples[1024:3072] = 0
        spectrogram = self.compute_checked(samples)
        silent = (spectrogram.starts >= 1024) & (spectrogram.stops <= 3072)
        self.assertTrue(silent.any())
        np.testing.assert_array_equal(spectrogram.levels[silent], FLOOR_DB)

    def test_nonzero_samples_that_the_window_ignores_leave_the_floor(self):
        # The periodic Hann window is zero at a segment's first sample.
        samples = np.zeros(1024, dtype=complex)
        samples[0] = 1
        spectrogram = self.compute_checked(samples)
        np.testing.assert_array_equal(spectrogram.levels, FLOOR_DB)

    def test_amplitude_does_not_change_the_levels(self):
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(4), windows=2)
        samples = capture.ravel()
        peak = max(np.abs(samples.real).max(), np.abs(samples.imag).max())
        unit = samples / peak
        reference = self.compute_checked(unit).levels
        for factor in (LARGEST, 1e-300):
            with self.subTest(factor=factor):
                np.testing.assert_allclose(
                    self.compute_checked(unit * factor).levels, reference, atol=1e-6
                )

    def test_extreme_and_subnormal_values_stay_finite(self):
        rng = np.random.default_rng(29)
        mixed = rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
        mixed[::97] = complex(LARGEST, -LARGEST)
        mixed[1::89] = complex(SMALLEST, 0)
        for samples in (
            np.full(1024, complex(LARGEST, LARGEST)),
            np.full(1024, complex(-LARGEST, LARGEST)),
            np.full(1024, complex(SMALLEST, SMALLEST)),
            mixed,
        ):
            with self.subTest(first=samples[0]):
                self.compute_checked(samples)

    def test_time_steps_average_neighboring_segments(self):
        rng = np.random.default_rng(6)
        samples = rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
        spectrogram = self.compute_checked(samples, max_time_steps=10)
        # 38 segments in groups of 4: nine full groups and a last group of 2.
        np.testing.assert_array_equal(spectrogram.segments, [4] * 9 + [2])
        np.testing.assert_array_equal(spectrogram.starts, np.arange(10) * 4 * HOP)
        np.testing.assert_array_equal(
            spectrogram.stops[:-1], spectrogram.starts[:-1] + 3 * HOP + SEGMENT_LENGTH
        )
        self.assertEqual(spectrogram.stops[-1], 37 * HOP + SEGMENT_LENGTH)
        self.assertEqual(spectrogram.ignored_samples, 5000 - spectrogram.stops[-1])
        self.assertEqual(spectrogram.ignored_samples, 8)

        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SEGMENT_LENGTH) / SEGMENT_LENGTH)
        frames = [
            np.abs(np.fft.fftshift(np.fft.fft(samples[start : start + SEGMENT_LENGTH] * window)))
            ** 2
            for start in range(0, 37 * HOP + 1, HOP)
        ]
        rows = np.array([np.mean(frames[first : first + 4], axis=0) for first in range(0, 38, 4)])
        np.testing.assert_allclose(spectrogram.power, rows / rows.max(), rtol=1e-9, atol=1e-15)

    def test_time_steps_are_capped(self):
        for n, steps, per_step in (
            (1024, 7, 1),
            ((MAX_TIME_STEPS - 1) * HOP + SEGMENT_LENGTH, MAX_TIME_STEPS, 1),
            (MAX_TIME_STEPS * HOP + SEGMENT_LENGTH, MAX_TIME_STEPS // 2 + 1, 2),
            (100_000, 390, 2),
        ):
            with self.subTest(samples=n):
                spectrogram = self.compute_checked(tone(0.05, n))
                self.assertEqual(spectrogram.power.shape, (steps, SEGMENT_LENGTH))
                self.assertLessEqual(steps, MAX_TIME_STEPS)
                self.assertEqual(spectrogram.segments[0], per_step)
                self.assertEqual(spectrogram.segments.sum(), segment_count(n))
                self.assertEqual(spectrogram.starts[0], 0)
                self.assertLessEqual(spectrogram.stops[-1], n)
                self.assertLess(spectrogram.ignored_samples, HOP)

    def test_time_step_centers(self):
        spectrogram = self.compute_checked(tone(0.05, 1024))
        np.testing.assert_array_equal(spectrogram.centers, np.arange(7) * HOP + 127.5)

    def test_hann_window_copies_half_a_window_apart_add_to_one(self):
        window = hann(SEGMENT_LENGTH)
        self.assertEqual(window[0], 0)
        self.assertEqual(window[SEGMENT_LENGTH // 2], 1)
        np.testing.assert_allclose(window[:HOP] + window[HOP:], 1, atol=1e-15)

    def test_invalid_input_is_rejected(self):
        for samples, options in (
            (np.ones(SEGMENT_LENGTH - 1, dtype=complex), {}),
            (np.ones((2, 1024), dtype=complex), {}),
            (np.array([1, np.nan] * 512, dtype=complex), {}),
            (np.array([1, complex(0, np.inf)] * 512), {}),
            (np.ones(1024, dtype=complex), {"segment_length": 255}),
            (np.ones(1024, dtype=complex), {"max_time_steps": 0}),
        ):
            with self.subTest(shape=samples.shape, options=options):
                with self.assertRaises(ValueError):
                    compute_spectrogram(samples, **options)


class AxisTests(SpectrogramTestCase):
    def test_without_a_sample_rate_axes_use_cycles_and_sample_numbers(self):
        spectrogram = self.compute_checked(tone(0.05, 1024))
        frequencies, frequency_title = spectrogram.frequency_axis()
        times, time_title = spectrogram.time_axis()
        np.testing.assert_array_equal(frequencies, spectrogram.frequencies)
        np.testing.assert_array_equal(times, spectrogram.centers)
        self.assertEqual((frequency_title, time_title), ("Frequency (cycles per sample)", "Sample"))

    def test_a_sample_rate_gives_hertz_and_seconds(self):
        spectrogram = self.compute_checked(tone(0.05, 1024))
        frequencies, frequency_title = spectrogram.frequency_axis(2.4e6)
        times, time_title = spectrogram.time_axis(2.4e6)
        self.assertEqual((frequencies[0], frequencies[-1]), (-1.2e6, 1.2e6 * 127 / 128))
        np.testing.assert_allclose(times, spectrogram.centers / 2.4e6)
        self.assertEqual(frequency_title, "Frequency from the band center (Hz)")
        self.assertEqual(time_title, "Time (s)")

    def test_unrepresentable_times_fall_back_to_sample_numbers(self):
        spectrogram = self.compute_checked(tone(0.05, 1024))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            times, title = spectrogram.time_axis(SMALLEST)
        self.assertEqual(title, "Sample")
        np.testing.assert_array_equal(times, spectrogram.centers)


class ImportTests(unittest.TestCase):
    def test_spectrogram_and_summaries_do_not_load_app_or_training_modules(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import spectrogram, summaries; "
                "heavy = {'streamlit', 'sklearn', 'matplotlib', 'plotly', 'spectrum_sense'}; "
                "loaded = heavy & sys.modules.keys(); "
                "assert not loaded, loaded",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
