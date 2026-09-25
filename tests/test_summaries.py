"""Tests for the plain-language chart summaries shown under each app chart."""

import re
import unittest
import warnings

import numpy as np

from signal_config import CLASSES, SAMPLES_PER_WINDOW
from spectrum_sense import generate_capture
from summaries import (
    constellation_summary,
    format_frequency,
    format_sample_rate,
    scores_summary,
    spectrum_db,
    spectrum_summary,
    time_summary,
)


LARGEST = float(np.finfo(np.float64).max)
SMALLEST = float.fromhex("0x0.0000000000001p-1022")


def tone(bin_index: int, n: int = SAMPLES_PER_WINDOW) -> np.ndarray:
    return np.exp(2j * np.pi * bin_index * np.arange(n) / n)


def tone_over_flat_floor(bin_index: int, n: int = SAMPLES_PER_WINDOW) -> np.ndarray:
    """An impulse of height n/9 has a flat spectrum of n/9; adding a unit tone
    makes one bin n + n/9, so the peak is exactly 20 dB above the median bin."""
    samples = tone(bin_index, n)
    samples[0] += n / 9
    return samples


class SummaryTestCase(unittest.TestCase):
    def all_summaries(self, samples: np.ndarray) -> list[str]:
        """Every sample-based summary, with numeric problems turned into errors.

        Tiny values may underflow harmlessly, as in the feature tests.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                texts = [
                    time_summary(samples, 128),
                    constellation_summary(samples, 2),
                    spectrum_summary(samples),
                    spectrum_summary(samples, 2.4e6, 433.92e6),
                ]
                _, levels = spectrum_db(samples)
        self.assertTrue(np.isfinite(levels).all())
        for text in texts:
            self.assertNotRegex(text.lower(), r"\b(nan|inf)\b")
        return texts


class ScoresSummaryTests(unittest.TestCase):
    def test_scores_are_listed_from_highest_to_lowest(self):
        self.assertEqual(
            scores_summary(["BPSK", "QPSK", "noise"], np.array([0.07, 0.912, 0.018])),
            "Scores: QPSK 0.91, BPSK 0.07, noise 0.02.",
        )

    def test_ties_keep_the_model_class_order(self):
        self.assertEqual(
            scores_summary(["BPSK", "QPSK", "noise"], [0.25, 0.25, 0.5]),
            "Scores: noise 0.50, BPSK 0.25, QPSK 0.25.",
        )

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            scores_summary(CLASSES, [0.5, 0.5])


class TimeSummaryTests(SummaryTestCase):
    def test_reports_the_span_and_rms_of_each_component(self):
        samples = np.full(SAMPLES_PER_WINDOW, 3 - 4j)
        self.assertEqual(
            time_summary(samples, 128),
            "The time plot shows the first 128 of 1,024 samples. Over that span, "
            "the RMS amplitude is 3 for I and 4 for Q.",
        )

    def test_rms_covers_only_the_plotted_span(self):
        samples = np.concatenate([np.full(128, 0.5 + 0j), np.full(896, 100 + 100j)])
        self.assertIn("is 0.5 for I and 0 for Q", time_summary(samples, 128))

    def test_short_recording_is_shown_in_full(self):
        text = time_summary(np.ones(100, dtype=complex), 128)
        self.assertTrue(text.startswith("The time plot shows all 100 samples."), text)

    def test_invalid_input_is_rejected(self):
        for samples, shown in (
            (np.array([], dtype=complex), 128),
            (np.ones((2, 64), dtype=complex), 128),
            (np.array([1, np.nan], dtype=complex), 128),
            (np.ones(64, dtype=complex), 0),
        ):
            with self.subTest(shape=samples.shape, shown=shown):
                with self.assertRaises(ValueError):
                    time_summary(samples, shown)


class ConstellationSummaryTests(SummaryTestCase):
    def test_points_on_one_circle_have_no_variation(self):
        text = constellation_summary(2 * tone(5), 2)
        self.assertIn("shows 512 points (one of every 2 samples)", text)
        self.assertIn("averages 2, with a coefficient of variation of 0.00", text)

    def test_known_spread_of_distances(self):
        # Distances 1 and 3: mean 2, standard deviation 1.
        samples = np.tile([1 + 0j, 0 - 3j], 32)
        text = constellation_summary(samples)
        self.assertIn("shows 64 points.", text)
        self.assertIn("averages 2, with a coefficient of variation of 0.50", text)

    def test_summary_does_not_name_a_modulation(self):
        rng = np.random.default_rng(3)
        for label in CLASSES:
            windows, _, _ = generate_capture(label, rng, windows=1)
            with self.subTest(label=label):
                text = constellation_summary(windows[0], 2)
                for name in CLASSES:
                    self.assertNotIn(name.lower(), text.lower())


class SpectrumSummaryTests(SummaryTestCase):
    def test_strongest_bin_and_level_above_median(self):
        self.assertEqual(
            spectrum_summary(tone_over_flat_floor(100)),
            "Spectrum of 1,024 frequency bins: the strongest bin is at 0.098 cycles "
            "per sample. It is 20.0 dB above the median bin.",
        )

    def test_sample_rate_adds_hertz_and_center_frequency_adds_radio_frequency(self):
        samples = tone_over_flat_floor(-100)
        self.assertIn(
            "-0.098 cycles per sample (100 kHz below the center of the recorded band)",
            spectrum_summary(samples, sample_rate=1.024e6),
        )
        self.assertIn(
            "(100 kHz below the center of the recorded band, about 433.82 MHz)",
            spectrum_summary(samples, sample_rate=1.024e6, center_frequency=433.92e6),
        )
        self.assertIn(
            "0.098 cycles per sample (100 kHz above the center",
            spectrum_summary(tone_over_flat_floor(100), sample_rate=1.024e6),
        )

    def test_zero_frequency_peak(self):
        samples = tone_over_flat_floor(0)
        self.assertIn("at 0.000 cycles per sample.", spectrum_summary(samples))
        self.assertIn(
            "(the center of the recorded band, about 100 MHz)",
            spectrum_summary(samples, sample_rate=1e6, center_frequency=100e6),
        )

    def test_flat_spectrum_names_no_peak(self):
        impulse = np.zeros(SAMPLES_PER_WINDOW, dtype=complex)
        impulse[0] = 1
        self.assertEqual(
            spectrum_summary(impulse),
            "Spectrum of 1,024 frequency bins: flat; no bin is more than 0.1 dB above "
            "the median bin.",
        )

    def test_summary_matches_the_plotted_levels(self):
        rng = np.random.default_rng(8)
        windows, _, _ = generate_capture("QPSK", rng, windows=1, freq_offset_override=0.02)
        frequencies, levels = spectrum_db(windows[0])
        peak = int(np.argmax(levels))
        text = spectrum_summary(windows[0])
        self.assertIn(f"at {round(frequencies[peak], 3) + 0.0:.3f} cycles per sample", text)
        self.assertIn(f"{levels[peak] - np.median(levels):.1f} dB above", text)

    def test_unrepresentable_radio_frequency_is_left_out(self):
        # Valid metadata can still overflow when the offset is added.
        text = spectrum_summary(
            tone_over_flat_floor(100), sample_rate=LARGEST, center_frequency=LARGEST
        )
        self.assertIn("above the center of the recorded band)", text)
        self.assertNotIn("about", text)
        self.assertNotRegex(text.lower(), r"\b(nan|inf)\b")


class EdgeCaseTests(SummaryTestCase):
    def test_all_zero_window(self):
        time_text, constellation_text, spectrum_text, _ = self.all_summaries(
            np.zeros(SAMPLES_PER_WINDOW, dtype=complex)
        )
        self.assertIn("RMS amplitude is 0 for I and 0 for Q", time_text)
        self.assertIn("all at the origin because every sample is zero", constellation_text)
        self.assertIn("flat, because every sample is zero", spectrum_text)

    def test_largest_finite_components(self):
        for value, rms_i, rms_q, distance in (
            (complex(LARGEST, 0), "1.80e+308", "0", "1.80e+308"),
            (complex(LARGEST, LARGEST), "1.80e+308", "1.80e+308", "2.54e+308"),
            (complex(-LARGEST, LARGEST), "1.80e+308", "1.80e+308", "2.54e+308"),
        ):
            with self.subTest(value=value):
                time_text, constellation_text, _, _ = self.all_summaries(
                    np.full(SAMPLES_PER_WINDOW, value)
                )
                self.assertIn(f"is {rms_i} for I and {rms_q} for Q", time_text)
                self.assertIn(f"averages {distance}, with a coefficient of variation of 0.00", constellation_text)

    def test_subnormal_components(self):
        time_text, constellation_text, _, _ = self.all_summaries(
            np.full(SAMPLES_PER_WINDOW, complex(SMALLEST, SMALLEST))
        )
        self.assertIn("is 4.94e-324 for I and 4.94e-324 for Q", time_text)
        self.assertIn("averages 6.99e-324", constellation_text)

    def test_mixed_extreme_and_typical_amplitudes(self):
        rng = np.random.default_rng(29)
        samples = rng.standard_normal(SAMPLES_PER_WINDOW) + 1j * rng.standard_normal(
            SAMPLES_PER_WINDOW
        )
        samples[::97] = complex(LARGEST, -LARGEST)
        samples[1::89] = complex(SMALLEST, 0)
        samples[2::83] = 0
        for text in self.all_summaries(samples):
            self.assertRegex(text, r"\d")

    def test_generated_captures_summarize_cleanly(self):
        rng = np.random.default_rng(42)
        for label in CLASSES:
            windows, _, _ = generate_capture(label, rng, windows=1)
            with self.subTest(label=label):
                texts = self.all_summaries(windows[0])
                self.assertTrue(all(re.search(r"\d", text) for text in texts))


class FormattingTests(unittest.TestCase):
    def test_frequencies(self):
        for hertz, digits, expected in (
            (433.92e6, 6, "433.92 MHz"),
            (2.4e9, 6, "2.4 GHz"),
            (-30_468.75, 3, "-30.5 kHz"),
            (999_999.9, 3, "1 MHz"),
            (12.5, 6, "12.5 Hz"),
        ):
            with self.subTest(hertz=hertz):
                self.assertEqual(format_frequency(hertz, digits), expected)

    def test_sample_rates(self):
        for rate, expected in (
            (2.4e6, "2.4 million samples per second"),
            (48_000, "48 thousand samples per second"),
            (999_999.9999, "1 million samples per second"),
            (250, "250 samples per second"),
        ):
            with self.subTest(rate=rate):
                self.assertEqual(format_sample_rate(rate), expected)


if __name__ == "__main__":
    unittest.main()
