"""Tests for the plain-language chart summaries shown under each app chart."""

import re
import unittest
import warnings

import numpy as np

from signal_config import (
    CLASSES,
    NOT_CLASSIFIABLE,
    REASON_CLOSE_SCORES,
    REASON_UNFAMILIAR,
    SAMPLES_PER_WINDOW,
    UNCERTAIN,
)
from spectrogram import FLOOR_DB, SEGMENT_LENGTH, Spectrogram, compute_spectrogram
from spectrum_sense import Decision, decision_order, generate_capture
from summaries import (
    MAX_LISTED_RUNS,
    STEADY_LEVEL_DB,
    constellation_summary,
    decision_summary,
    format_duration,
    format_frequency,
    format_sample_rate,
    scores_summary,
    spectrogram_summary,
    spectrum_db,
    spectrum_summary,
    time_summary,
    timeline_summary,
)


LARGEST = float(np.finfo(np.float64).max)
SMALLEST = float.fromhex("0x0.0000000000001p-1022")


def tone(bin_index: int, n: int = SAMPLES_PER_WINDOW) -> np.ndarray:
    return np.exp(2j * np.pi * bin_index * np.arange(n) / n)


def tone_at(cycles: float, n: int) -> np.ndarray:
    """A unit tone at a frequency in cycles per sample."""
    return np.exp(2j * np.pi * cycles * np.arange(n))


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
                spectrogram = compute_spectrogram(samples)
                texts = [
                    time_summary(samples, 128),
                    constellation_summary(samples, 2),
                    spectrum_summary(samples),
                    spectrum_summary(samples, 2.4e6, 433.92e6),
                    spectrogram_summary(spectrogram),
                    spectrogram_summary(spectrogram, 2.4e6, 433.92e6),
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


class DecisionSummaryTests(unittest.TestCase):
    def test_named_class(self):
        self.assertEqual(
            decision_summary(Decision("BPSK", None, "BPSK", 0.987, 2.34), 4.43, 0.6),
            "The model names BPSK: the window is 2.3 from the nearest trained class "
            "(limit 4.4) and the top score is 0.99 (minimum 0.60).",
        )

    def test_unlike_the_training_data(self):
        decision = Decision(UNCERTAIN, REASON_UNFAMILIAR, "QPSK", 0.996, 12.06)
        self.assertEqual(
            decision_summary(decision, 4.43, 0.6),
            "Uncertain because the window is unlike the training data: it is 12.1 from the "
            "nearest trained class, beyond the limit of 4.4. The leading candidate is QPSK "
            "with a score of 1.00.",
        )

    def test_scores_are_close(self):
        decision = Decision(UNCERTAIN, REASON_CLOSE_SCORES, "noise", 0.52, 1.2)
        self.assertEqual(
            decision_summary(decision, 4.43, 0.6),
            "Uncertain because the scores are close: the top score is 0.52, below the "
            "minimum of 0.60. The leading candidate is noise with a score of 0.52.",
        )


class TimelineSummaryTests(unittest.TestCase):
    ORDER = decision_order(CLASSES)

    def test_single_window(self):
        self.assertEqual(
            timeline_summary(["QPSK"], self.ORDER), "The timeline shows 1 window: 1 QPSK."
        )

    def test_one_decision_throughout(self):
        self.assertEqual(
            timeline_summary(["noise"] * 4, self.ORDER),
            "The timeline shows 4 windows: 4 noise. All windows have the same decision.",
        )

    def test_counts_follow_the_given_order_and_runs_are_listed(self):
        labels = [UNCERTAIN, "BPSK", "BPSK", NOT_CLASSIFIABLE, "noise", "noise"]
        self.assertEqual(
            timeline_summary(labels, self.ORDER),
            "The timeline shows 6 windows: 2 noise, 2 BPSK, 1 uncertain, 1 not classifiable. "
            "In order, window 1: uncertain; windows 2 to 3: BPSK; window 4: not classifiable; "
            "windows 5 to 6: noise.",
        )

    def test_many_changes_are_summarized_by_the_longest_run(self):
        labels = ["BPSK", "QPSK"] * MAX_LISTED_RUNS + ["noise"] * 1500 + ["BPSK"] * 1500
        self.assertEqual(
            timeline_summary(labels, self.ORDER),
            "The timeline shows 3,012 windows: 1,500 noise, 1,506 BPSK, 6 QPSK. The decision "
            "changes 13 times; the longest run is windows 13 to 1,512 (noise).",
        )

    def test_labels_outside_the_order_are_listed_last(self):
        text = timeline_summary(["mystery", "BPSK"], self.ORDER)
        self.assertTrue(text.startswith("The timeline shows 2 windows: 1 BPSK, 1 mystery."), text)

    def test_empty_timeline_is_rejected(self):
        with self.assertRaises(ValueError):
            timeline_summary([], self.ORDER)


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


def hand_made_spectrogram(power, segments, segment_length: int = 8) -> Spectrogram:
    """A spectrogram with chosen power, laid out as compute_spectrogram lays it out."""
    segments = np.asarray(segments)
    hop = segment_length // 2
    first = np.concatenate([[0], np.cumsum(segments)[:-1]])
    return Spectrogram(
        frequencies=np.fft.fftshift(np.fft.fftfreq(segment_length)),
        starts=first * hop,
        stops=(first + segments - 1) * hop + segment_length,
        segments=segments,
        power=np.asarray(power, dtype=float),
        segment_length=segment_length,
        ignored_samples=0,
    )


def weighted_example() -> Spectrogram:
    """Eight bins (index 5 is 0.125 and index 1 is -0.375 cycles per sample) in
    three time steps of 3, 3, and 1 segments. Weighted by segments, bin 5
    averages 0.2 and the median bin 0.002, exactly 20 dB apart; an unweighted
    mean would pick bin 1 instead. The first step is the quietest (total 0.247)
    and the last the loudest (total 1.014): 6.1 dB apart."""
    common = np.full(8, 0.002)
    first = common.copy()
    first[5] = (1.4 - 0.002) / 6
    last = common.copy()
    last[1] = 1.0
    return hand_made_spectrogram([first, first, last], [3, 3, 1])


class SpectrogramSummaryTests(SummaryTestCase):
    def summarize(self, spectrogram: Spectrogram, *metadata) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                text = spectrogram_summary(spectrogram, *metadata)
        self.assertNotRegex(text.lower(), r"\b(nan|inf)\b")
        return text

    def test_numbers_come_from_the_segment_weighted_average(self):
        self.assertEqual(
            self.summarize(weighted_example()),
            "The spectrogram shows 8 frequency bins across and 3 time steps down; each "
            "averages 3 overlapping 8-sample segments (the last averages 1). Averaged over "
            "time, the strongest frequency is 0.125 cycles per sample, 20.0 dB above the "
            "median bin. The overall level changes over time: the quietest time step, around "
            "sample 8, is 6.1 dB below the loudest, around sample 28.",
        )

    def test_sample_rate_gives_hertz_and_times_and_center_gives_radio_frequency(self):
        text = self.summarize(weighted_example(), 1000.0, 433.92e6)
        self.assertIn(
            "0.125 cycles per sample (125 Hz above the center of the recorded band, about "
            "433.92 MHz), 20.0 dB above the median bin.",
            text,
        )
        self.assertIn(
            "the quietest time step, around 7.5 milliseconds, is 6.1 dB below the loudest, "
            "around 27.5 milliseconds.",
            text,
        )

    def test_unrepresentable_times_fall_back_to_sample_numbers(self):
        text = self.summarize(weighted_example(), SMALLEST)
        self.assertIn("around sample 8, is 6.1 dB below the loudest, around sample 28.", text)

    def test_steady_tone(self):
        self.assertEqual(
            self.summarize(compute_spectrogram(tone_at(0.125, 4096))),
            "The spectrogram shows 256 frequency bins across and 31 time steps down; each is "
            "one 256-sample segment, overlapping the next by half. Averaged over time, the "
            "strongest frequency is 0.125 cycles per sample, at least 80 dB above the median "
            "bin. The overall level is steady over time: every time step is within 0.0 dB of "
            "the loudest.",
        )

    def test_known_level_changes_are_located(self):
        samples = tone_at(0.125, 4096)
        samples[768:1024] *= 2  # One whole segment 6.0 dB louder...
        samples[2048:2304] *= 0.1  # ...and one 20 dB quieter: 26.0 dB apart.
        spectrogram = compute_spectrogram(samples)
        self.assertIn(
            "The overall level changes over time: the quietest time step, around sample "
            "2,176, is 26.0 dB below the loudest, around sample 896.",
            self.summarize(spectrogram),
        )
        text = self.summarize(spectrogram, 2e6)
        self.assertIn("0.125 cycles per sample (250 kHz above the center", text)
        self.assertIn(
            "around 1.09 milliseconds, is 26.0 dB below the loudest, around 448 microseconds.",
            text,
        )

    def test_steadiness_threshold_matches_the_printed_value(self):
        for change, expected in (
            (STEADY_LEVEL_DB - 0.06, "is steady over time: every time step is within 2.9 dB"),
            (STEADY_LEVEL_DB - 0.04, "changes over time: the quietest time step, around "
             "sample 8, is 3.0 dB below"),
        ):
            with self.subTest(change=change):
                power = np.zeros((2, 8))
                power[:, 0] = 1.0, 10 ** (-change / 10)
                self.assertIn(expected, self.summarize(hand_made_spectrogram(power, [1, 1])))

    def test_flat_spectrum_and_silence(self):
        samples = np.zeros(SAMPLES_PER_WINDOW, dtype=complex)
        samples[128] = 1  # Seen only by the first segment, in the middle of its window.
        self.assertEqual(
            self.summarize(compute_spectrogram(samples)),
            "The spectrogram shows 256 frequency bins across and 7 time steps down; each is "
            "one 256-sample segment, overlapping the next by half. Averaged over time, the "
            "spectrum is flat; no bin is more than 0.1 dB above the median bin. The overall "
            "level changes over time: the quietest time step, around sample 256, is at least "
            "80 dB below the loudest, around sample 128.",
        )

    def test_all_zero_recording(self):
        text = self.summarize(compute_spectrogram(np.zeros(2048, dtype=complex)))
        self.assertTrue(text.startswith("The spectrogram shows 256 frequency bins across and 15"))
        self.assertTrue(
            text.endswith(
                f"Every cell is at the {FLOOR_DB:g} dB floor because no segment holds any "
                "power, for example because every sample is zero."
            ),
            text,
        )

    def test_samples_after_the_last_segment_are_mentioned(self):
        text = self.summarize(compute_spectrogram(tone_at(0.125, 1100)))
        self.assertIn("The last 76 samples, which do not fill a segment, are left out.", text)
        self.assertNotIn("left out", self.summarize(compute_spectrogram(tone_at(0.125, 1024))))

    def test_one_time_step_has_no_level_change(self):
        text = self.summarize(compute_spectrogram(tone_at(0.125, SEGMENT_LENGTH)))
        self.assertIn("1 time step down; each is one", text)
        self.assertNotIn("overall level", text)

    def test_summary_matches_the_plotted_levels(self):
        rng = np.random.default_rng(8)
        capture, _, _ = generate_capture("QPSK", rng, windows=4, freq_offset_override=0.02)
        spectrogram = compute_spectrogram(capture.ravel())
        # Recover power from the plotted dB levels; none of them is at the floor.
        levels = spectrogram.levels
        self.assertGreater(levels.min(), FLOOR_DB)
        power = 10 ** (levels / 10)
        spectrum = spectrogram.segments @ power / spectrogram.segments.sum()
        peak = int(np.argmax(spectrum))
        above = 10 * np.log10(spectrum[peak] / np.median(spectrum))
        totals = power.sum(axis=1)
        change = 10 * np.log10(totals.max() / totals.min())
        text = self.summarize(spectrogram)
        where = round(float(spectrogram.frequencies[peak]), 3) + 0.0
        self.assertIn(f"the strongest frequency is {where:.3f} cycles per sample", text)
        self.assertIn(f", {above:.1f} dB above the median bin.", text)
        self.assertIn(f"within {change:.1f} dB of the loudest", text)


class EdgeCaseTests(SummaryTestCase):
    def test_all_zero_window(self):
        time_text, constellation_text, spectrum_text, _, spectrogram_text, _ = (
            self.all_summaries(np.zeros(SAMPLES_PER_WINDOW, dtype=complex))
        )
        self.assertIn("RMS amplitude is 0 for I and 0 for Q", time_text)
        self.assertIn("all at the origin because every sample is zero", constellation_text)
        self.assertIn("flat, because every sample is zero", spectrum_text)
        self.assertIn(f"Every cell is at the {FLOOR_DB:g} dB floor", spectrogram_text)

    def test_largest_finite_components(self):
        for value, rms_i, rms_q, distance in (
            (complex(LARGEST, 0), "1.80e+308", "0", "1.80e+308"),
            (complex(LARGEST, LARGEST), "1.80e+308", "1.80e+308", "2.54e+308"),
            (complex(-LARGEST, LARGEST), "1.80e+308", "1.80e+308", "2.54e+308"),
        ):
            with self.subTest(value=value):
                time_text, constellation_text, *_ = self.all_summaries(
                    np.full(SAMPLES_PER_WINDOW, value)
                )
                self.assertIn(f"is {rms_i} for I and {rms_q} for Q", time_text)
                self.assertIn(f"averages {distance}, with a coefficient of variation of 0.00", constellation_text)

    def test_subnormal_components(self):
        time_text, constellation_text, *_ = self.all_summaries(
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
