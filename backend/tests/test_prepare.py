"""Tests for preparing a recording: frequency shift, low-pass filter,
resampling, the loudest windows, and what the demo model makes of the result."""

from collections import Counter
from itertools import product
import math
import unittest
import warnings

import numpy as np
from scipy.signal import freqz, resample_poly

from iq_csv import MAX_SAMPLES
from iq_formats import Recording
from prepare import (
    FULL_BAND_SHARE,
    LOUD_WINDOWS,
    MAX_FACTOR,
    MAX_FILTER_TAPS,
    MAX_RATIO_TERM,
    MIN_BANDWIDTH,
    MIN_FACTOR,
    NO_PREPARATION,
    PrepareSettings,
    level_phrase,
    loudest_windows,
    lowpass_taps,
    prepare_recording,
    resample_ratio,
    result_summary,
    window_levels,
)
from signal_config import (
    FREQ_OFFSET_RANGE,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
    UNCERTAIN,
)
from spectrum_sense import (
    build_demo_model,
    classify_recording,
    complex_gaussian,
    extract_features,
    generate_capture,
)


LARGEST = float(np.finfo(np.float64).max)


def tone(cycles: float, n: int) -> np.ndarray:
    return np.exp(2j * np.pi * cycles * np.arange(n))


def recording(samples: np.ndarray, sample_rate: float | None = None, center: float | None = None):
    return Recording(np.asarray(samples, dtype=np.complex128), sample_rate, center, "test")


def peak(samples: np.ndarray) -> float:
    """The strongest frequency in cycles per sample, found with a fine FFT."""
    size = 1 << 18
    spectrum = np.abs(np.fft.fft(samples * np.hanning(len(samples)), size))
    return float(np.fft.fftfreq(size)[np.argmax(spectrum)])


def rectangular_capture(
    label: str, samples_per_symbol: int, rng: np.random.Generator, windows: int
) -> np.ndarray:
    """BPSK or QPSK with rectangular pulses at any timing, like the training
    generator: random timing and phase, SNR from the training range, and a
    carrier offset whose change per symbol is within the training range. The
    capture holds `windows` windows at SAMPLES_PER_SYMBOL samples per symbol,
    plus one window's worth at each end."""
    count = (windows + 2) * SAMPLES_PER_WINDOW * samples_per_symbol // SAMPLES_PER_SYMBOL
    symbols = rng.choice((-1.0, 1.0), size=count // samples_per_symbol + 2).astype(complex)
    if label == "QPSK":
        symbols = (symbols + 1j * rng.choice((-1.0, 1.0), size=len(symbols))) / np.sqrt(2)
    start = int(rng.integers(0, samples_per_symbol))
    signal = np.repeat(symbols, samples_per_symbol)[start : start + count]
    offset = rng.uniform(*FREQ_OFFSET_RANGE) * SAMPLES_PER_SYMBOL / samples_per_symbol
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * offset * np.arange(count)
    snr_db = rng.uniform(*SNR_DB_RANGE)
    return signal * np.exp(1j * phase) + complex_gaussian(rng, count) * 10 ** (-snr_db / 20)


class ResampleRatioTests(unittest.TestCase):
    def test_exact_ratios_in_lowest_terms(self):
        for factor, expected in (
            (2.0, (2, 1)), (0.4, (2, 5)), (1.0, (1, 1)), (0.5, (1, 2)), (8.0, (8, 1)),
            (1 / 3, (1, 3)), (0.76, (19, 25)), (MIN_FACTOR, (1, MAX_RATIO_TERM)),
        ):
            with self.subTest(factor=factor):
                self.assertEqual(resample_ratio(factor), expected)

    def test_other_ratios_are_the_closest_with_small_terms(self):
        pairs = list(product(range(1, MAX_RATIO_TERM + 1), repeat=2))
        for factor in (0.076, 0.0137, 2.718281828, 7.99, 1 / 7.3, 0.333):
            with self.subTest(factor=factor):
                up, down = resample_ratio(factor)
                self.assertLessEqual(max(up, down), MAX_RATIO_TERM)
                self.assertEqual(math.gcd(up, down), 1)
                best = min(abs(a / b - factor) for a, b in pairs)
                self.assertAlmostEqual(abs(up / down - factor), best, places=15)
        # RDS at 2,375 symbols per second from 250,000 samples per second, as the README says.
        up, down = resample_ratio(8 * 2_375 / 250_000)
        self.assertEqual((up, down), (6, 79))
        self.assertLess(abs(up / down / 0.076 - 1), 0.0007)

    def test_invalid_factors_are_rejected(self):
        for factor in (0.0, -2.0, math.nan, math.inf):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                resample_ratio(factor)


class LowpassTests(unittest.TestCase):
    def test_keeps_the_band_and_removes_the_rest(self):
        for bandwidth in (MIN_BANDWIDTH * 5, 0.1, 0.34, 0.9):
            with self.subTest(bandwidth=bandwidth):
                taps = lowpass_taps(bandwidth)
                self.assertEqual(len(taps) % 2, 1)
                np.testing.assert_allclose(taps, taps[::-1])  # linear phase
                self.assertAlmostEqual(float(taps.sum()), 1.0, places=12)
                inside, outside = 0.4 * bandwidth, min(0.6 * bandwidth, 0.5)
                _, response = freqz(taps, worN=[inside, outside], fs=1.0)
                self.assertGreater(20 * np.log10(abs(response[0])), -0.1)
                if outside < 0.5:
                    self.assertLess(20 * np.log10(abs(response[1])), -40)

    def test_narrow_bands_use_at_most_the_largest_filter(self):
        self.assertEqual(len(lowpass_taps(MIN_BANDWIDTH)), MAX_FILTER_TAPS)
        self.assertLess(len(lowpass_taps(0.5)), MAX_FILTER_TAPS)

    def test_invalid_bandwidths_are_rejected(self):
        for bandwidth in (0.0, MIN_BANDWIDTH / 2, 1.0, 1.5, math.nan):
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                lowpass_taps(bandwidth)


class PrepareRecordingTests(unittest.TestCase):
    def assert_no_numeric_warnings(self, function, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="raise"):
                return function(*args)

    def test_default_settings_change_nothing(self):
        original = recording(tone(0.1, 3 * SAMPLES_PER_WINDOW), 500.0, 1e9)
        prepared = prepare_recording(original, PrepareSettings())
        self.assertIs(prepared.recording, original)
        self.assertEqual(prepared.steps, ())
        self.assertEqual(prepared.description, NO_PREPARATION)
        self.assertEqual(prepared.ratio, (1, 1))
        self.assertIsNone(prepared.samples_per_symbol)
        self.assertEqual((prepared.filled_share, prepared.band_note), (1.0, ""))

    def test_a_tone_moves_to_zero_hertz(self):
        samples = tone(37 / 500, 8 * SAMPLES_PER_WINDOW)
        original = recording(samples.copy(), 500.0, 3_671_055_830.0)
        prepared = prepare_recording(original, PrepareSettings(offset=37.0))
        self.assertAlmostEqual(peak(prepared.recording.samples), 0.0, places=6)
        np.testing.assert_allclose(prepared.recording.samples, 1.0, atol=1e-9)
        np.testing.assert_array_equal(original.samples, samples)  # the original is unchanged
        self.assertEqual(prepared.recording.sample_rate, 500.0)
        # The band's new center is where the tone was.
        self.assertEqual(prepared.recording.center_frequency, 3_671_055_867.0)
        self.assertEqual(prepared.steps, ("Moved 37 Hz to 0 Hz (a shift of -37 Hz).",))
        self.assertEqual(prepared.ratio, (1, 1))
        self.assertEqual(prepared.band_note, "")

    def test_a_shift_alone_leaves_the_features_unchanged(self):
        # The app says so: the features remove a steady frequency offset themselves.
        for label in ("BPSK", "QPSK", "noise"):
            with self.subTest(label=label):
                capture, _, _ = generate_capture(label, np.random.default_rng(8), windows=2)
                samples = capture.ravel()
                prepared = prepare_recording(recording(samples), PrepareSettings(offset=0.2))
                for start in (0, SAMPLES_PER_WINDOW):
                    window = slice(start, start + SAMPLES_PER_WINDOW)
                    np.testing.assert_allclose(
                        extract_features(prepared.recording.samples[window]),
                        extract_features(samples[window]), rtol=1e-6, atol=1e-9,
                    )

    def test_without_a_sample_rate_the_offset_is_in_cycles_per_sample(self):
        original = recording(tone(-0.123, 4 * SAMPLES_PER_WINDOW), None, 1e9)
        prepared = prepare_recording(original, PrepareSettings(offset=-0.123))
        self.assertAlmostEqual(peak(prepared.recording.samples), 0.0, places=6)
        self.assertIsNone(prepared.recording.sample_rate)
        self.assertIsNone(prepared.recording.center_frequency)  # unknown without a rate
        self.assertEqual(
            prepared.description,
            "Moved -0.123 cycles per sample to 0 (a shift of +0.123 cycles per sample).",
        )

    def test_the_filter_keeps_the_band_to_its_edges_and_removes_the_rest(self):
        n = 4 * SAMPLES_PER_WINDOW
        inside, outside = tone(0.05, n), tone(0.3, n)
        prepared = prepare_recording(
            recording(inside + outside, 1_000.0), PrepareSettings(bandwidth=200.0)
        )
        kept = prepared.recording.samples
        self.assertEqual(len(kept), n)
        taps = len(lowpass_taps(0.2))
        half = taps // 2
        middle = slice(half, n - half)
        self.assertLess(np.mean(np.abs(kept[middle] - inside[middle]) ** 2), 1e-5)
        # Mirrored ends keep the level steady up to the first and last sample.
        for end in (kept[:half], kept[-half:]):
            self.assertAlmostEqual(float(np.mean(np.abs(end) ** 2)), 1.0, delta=0.05)
        self.assertEqual(
            prepared.steps,
            (f"Kept a band 200 Hz wide around 0 Hz with a low-pass filter of {taps} taps.",),
        )
        self.assertAlmostEqual(prepared.filled_share, 0.2)
        self.assertEqual(
            prepared.band_note,
            "Only about 20% of the prepared band holds the recording's signal and noise; "
            "the filter left the rest empty. The model learned noise that fills the whole "
            'band, so expect more "uncertain" answers.',
        )

    def test_resampling_to_a_symbol_rate_keeps_a_tone_at_its_frequency(self):
        samples = tone(37 / 500, 6 * SAMPLES_PER_WINDOW)
        for symbol_rate, ratio, new_rate in ((125.0, (2, 1), 1_000.0), (25.0, (2, 5), 200.0)):
            with self.subTest(symbol_rate=symbol_rate):
                prepared = prepare_recording(
                    recording(samples, 500.0), PrepareSettings(symbol_rate=symbol_rate)
                )
                self.assertEqual(prepared.ratio, ratio)
                result = prepared.recording
                self.assertEqual(result.sample_rate, new_rate)
                self.assertEqual(len(result.samples), -(-len(samples) * ratio[0] // ratio[1]))
                self.assertAlmostEqual(peak(result.samples) * new_rate, 37.0, delta=0.01)
                self.assertEqual(prepared.samples_per_symbol, SAMPLES_PER_SYMBOL)
        self.assertEqual(
            prepare_recording(recording(samples, 500.0), PrepareSettings(symbol_rate=125.0)).steps,
            (
                "Resampled by 2/1 from 500 samples per second to 1 thousand samples per "
                "second, which gives 8 samples per symbol at 125 symbols per second.",
            ),
        )

    def test_resampling_by_a_factor_without_a_sample_rate(self):
        samples = tone(0.05, 6 * SAMPLES_PER_WINDOW)
        for factor, ratio, text in (
            (0.4, (2, 5), "Resampled by 2/5 (a factor of 0.4)."),
            (0.333, (1, 3), "Resampled by 1/3 (a factor of 0.3333, the closest to 0.333)."),
            (2.0, (2, 1), "Resampled by 2/1 (a factor of 2)."),
        ):
            with self.subTest(factor=factor):
                prepared = prepare_recording(recording(samples), PrepareSettings(factor=factor))
                self.assertEqual(prepared.ratio, ratio)
                self.assertEqual(prepared.steps, (text,))
                self.assertIsNone(prepared.recording.sample_rate)
                self.assertIsNone(prepared.samples_per_symbol)
                self.assertAlmostEqual(
                    peak(prepared.recording.samples), 0.05 * ratio[1] / ratio[0], places=5
                )

    def test_a_recording_already_at_eight_samples_per_symbol_is_not_resampled(self):
        original = recording(tone(0.01, 2 * SAMPLES_PER_WINDOW), 500.0)
        prepared = prepare_recording(original, PrepareSettings(symbol_rate=62.5))
        self.assertEqual(prepared.ratio, (1, 1))
        np.testing.assert_array_equal(prepared.recording.samples, original.samples)
        self.assertEqual(
            prepared.steps,
            ("Not resampled: the recording already has 8 samples per symbol at 62.5 symbols "
             "per second.",),
        )

    def test_the_steps_run_in_order_and_are_all_described(self):
        rng = np.random.default_rng(3)
        samples = complex_gaussian(rng, 10 * SAMPLES_PER_WINDOW) + 3 * tone(-80 / 500, 10_240)
        prepared = prepare_recording(
            recording(samples, 500.0, 1e9),
            PrepareSettings(offset=-80.0, bandwidth=170.0, symbol_rate=125.0),
        )
        self.assertEqual(len(prepared.steps), 3)
        self.assertTrue(prepared.steps[0].startswith("Moved -80 Hz to 0 Hz (a shift of +80 Hz)."))
        self.assertTrue(prepared.steps[1].startswith("Kept a band 170 Hz wide around 0 Hz"))
        self.assertTrue(prepared.steps[2].startswith("Resampled by 2/1"))
        self.assertEqual(prepared.description, " ".join(prepared.steps))
        self.assertAlmostEqual(prepared.filled_share, 0.17)
        self.assertIn(
            "the filter and the higher sample rate left the rest empty", prepared.band_note
        )
        self.assertAlmostEqual(peak(prepared.recording.samples), 0.0, places=4)

    def test_output_is_capped_at_the_sample_limit(self):
        rng = np.random.default_rng(4)
        samples = complex_gaussian(rng, 60_000)
        prepared = prepare_recording(
            recording(samples, 1_000.0), PrepareSettings(symbol_rate=250.0)
        )
        self.assertEqual(prepared.ratio, (2, 1))
        self.assertEqual(len(prepared.recording.samples), MAX_SAMPLES)
        self.assertEqual(
            prepared.steps[-1],
            f"Kept the first {MAX_SAMPLES:,} of 120,000 samples, the app's limit.",
        )
        # Only what is kept is resampled, but the kept samples match a full resampling.
        full = resample_poly(samples, 2, 1, padtype="reflect")
        np.testing.assert_allclose(prepared.recording.samples, full[:MAX_SAMPLES], atol=1e-12)

    def test_too_few_samples_after_resampling_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fewer than one 1,024-sample window"):
            prepare_recording(
                recording(np.ones(2 * SAMPLES_PER_WINDOW)), PrepareSettings(factor=0.4)
            )

    def test_invalid_settings_are_rejected_with_a_reason(self):
        rated = recording(np.ones(4 * SAMPLES_PER_WINDOW), 500.0)
        plain = recording(np.ones(4 * SAMPLES_PER_WINDOW))
        for original, settings, message in (
            (rated, PrepareSettings(offset=math.nan), "finite number"),
            (rated, PrepareSettings(factor=math.inf), "finite number"),
            (rated, PrepareSettings(offset=251.0), "between -250 Hz and 250 Hz"),
            (plain, PrepareSettings(offset=-0.6), "between -0.5 cycles per sample"),
            (rated, PrepareSettings(bandwidth=4.0), "at least 5 Hz and less than 500 Hz"),
            (rated, PrepareSettings(bandwidth=500.0), "less than 500 Hz"),
            (plain, PrepareSettings(bandwidth=1.0), "less than 1 cycles per sample"),
            (plain, PrepareSettings(symbol_rate=125.0), "needs the recording's sample rate"),
            (rated, PrepareSettings(symbol_rate=125.0, factor=2.0), "not both"),
            (rated, PrepareSettings(symbol_rate=-1.0), "must be positive"),
            (rated, PrepareSettings(symbol_rate=600.0), "between 0.01 and 8"),
            (rated, PrepareSettings(symbol_rate=0.5), "500 samples per second, 0.5 symbols"),
            (plain, PrepareSettings(factor=MAX_FACTOR * 1.01), "between 0.01 and 8"),
            (plain, PrepareSettings(factor=0.0), "between 0.01 and 8"),
        ):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, message):
                prepare_recording(original, settings)

    def test_invalid_recordings_are_rejected(self):
        for samples in (np.ones(SAMPLES_PER_WINDOW - 1), np.full(SAMPLES_PER_WINDOW, np.nan)):
            with self.subTest(length=len(samples)), self.assertRaises(ValueError):
                prepare_recording(recording(samples), PrepareSettings(offset=0.1))

    def test_extreme_amplitudes_give_a_plain_error_without_warnings(self):
        samples = np.full(2 * SAMPLES_PER_WINDOW, complex(LARGEST, LARGEST))
        with self.assertRaisesRegex(ValueError, "too large to prepare"):
            self.assert_no_numeric_warnings(
                prepare_recording, recording(samples), PrepareSettings(offset=0.1)
            )
        # Resampling to a lower rate keeps them finite.
        prepared = self.assert_no_numeric_warnings(
            prepare_recording, recording(samples * 0.5), PrepareSettings(factor=0.5)
        )
        self.assertTrue(np.isfinite(prepared.recording.samples).all())

    def test_the_band_note_names_each_cause(self):
        samples = recording(np.ones(4 * SAMPLES_PER_WINDOW), 1_000.0)
        for settings, share, cause in (
            (PrepareSettings(symbol_rate=250.0), 0.5, "; the higher sample rate left"),
            (PrepareSettings(bandwidth=300.0, symbol_rate=62.5), 0.6, "; the filter left"),
            (PrepareSettings(bandwidth=900.0, symbol_rate=250.0), 0.45, "; the filter and the"),
            (PrepareSettings(bandwidth=600.0, symbol_rate=62.5), 1.0, None),
            (PrepareSettings(symbol_rate=62.5), 1.0, None),
        ):
            with self.subTest(settings=settings):
                prepared = prepare_recording(samples, settings)
                self.assertAlmostEqual(prepared.filled_share, share)
                if cause is None:
                    self.assertEqual(prepared.band_note, "")
                else:
                    self.assertLess(share, FULL_BAND_SHARE)
                    self.assertIn(
                        f"Only about {share:.0%} of the prepared band", prepared.band_note
                    )
                    self.assertIn(cause, prepared.band_note)

    def test_result_summary(self):
        samples = tone(0.01, 3 * SAMPLES_PER_WINDOW)
        for original, settings, expected in (
            (recording(samples, 500.0), PrepareSettings(symbol_rate=125.0),
             "6,144 samples at 1 thousand samples per second, 8 samples per symbol "
             "(the model expects 8)."),
            (recording(samples, 500.0), PrepareSettings(offset=10.0),
             "3,072 samples at 500 samples per second; enter the symbol rate to see the "
             "samples per symbol."),
            (recording(samples), PrepareSettings(factor=0.5), "1,536 samples."),
        ):
            with self.subTest(settings=settings):
                self.assertEqual(result_summary(prepare_recording(original, settings)), expected)


class LoudestWindowTests(unittest.TestCase):
    def windows(self, amplitudes) -> np.ndarray:
        rng = np.random.default_rng(9)
        return np.concatenate(
            [amplitude * complex_gaussian(rng, SAMPLES_PER_WINDOW) for amplitude in amplitudes]
        )

    def test_levels_are_relative_to_the_median_window(self):
        samples = np.ones(3 * SAMPLES_PER_WINDOW + 5, dtype=complex)
        samples[SAMPLES_PER_WINDOW : 2 * SAMPLES_PER_WINDOW] = 2
        expected = [0.0, 10 * np.log10(4), 0.0]
        np.testing.assert_allclose(window_levels(samples), expected, atol=1e-12)

    def test_the_loudest_come_first_and_silent_windows_are_left_out(self):
        amplitudes = [1, 4, 0, 2, 4, 1, 3, 1]  # the median power is (1 + 4) / 2
        loud = loudest_windows(self.windows(amplitudes))
        self.assertEqual(len(loud), LOUD_WINDOWS)
        self.assertEqual(sorted(index for index, _ in loud[:2]), [1, 4])
        self.assertEqual([index for index, _ in loud[2:4]], [6, 3])
        levels = [level for _, level in loud]
        self.assertEqual(levels, sorted(levels, reverse=True))
        self.assertAlmostEqual(levels[0], 10 * np.log10(16 / 2.5), delta=0.5)
        self.assertNotIn(2, [index for index, _ in loudest_windows(self.windows(amplitudes), 8)])
        # Equal windows keep their order.
        same = np.tile(np.exp(0.3j * np.arange(SAMPLES_PER_WINDOW)), 3)
        self.assertEqual([index for index, _ in loudest_windows(same, 2)], [0, 1])
        self.assertEqual(loudest_windows(np.zeros(2 * SAMPLES_PER_WINDOW)), [])

    def test_extreme_amplitudes_do_not_overflow(self):
        samples = np.full(3 * SAMPLES_PER_WINDOW, complex(LARGEST, -LARGEST))
        samples[:SAMPLES_PER_WINDOW] *= 1e-300
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            levels = window_levels(samples)
            loud = loudest_windows(samples)
        self.assertTrue(np.isfinite(levels).all())
        self.assertEqual([index for index, _ in loud], [1, 2, 0])
        # The quiet window is floored at 80 dB below the loudest.
        self.assertAlmostEqual(levels[0], -80.0)

    def test_invalid_input_is_rejected(self):
        for samples in (np.ones(SAMPLES_PER_WINDOW - 1), np.full(SAMPLES_PER_WINDOW, np.inf)):
            with self.subTest(length=len(samples)), self.assertRaises(ValueError):
                window_levels(samples)

    def test_level_phrase(self):
        self.assertEqual(level_phrase(6.02), "6.0 dB above the median window")
        self.assertEqual(level_phrase(-3.0), "3.0 dB below the median window")
        self.assertEqual(level_phrase(0.04), "the same power as the median window")
        self.assertEqual(level_phrase(-0.04), "the same power as the median window")


class ModelOnPreparedSignalsTests(unittest.TestCase):
    """What the demo model makes of rectangular BPSK and QPSK recorded at other
    timings, as recorded and resampled to 8 samples per symbol."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_demo_model().model

    def shares(self, samples_per_symbol: int, captures: int = 8) -> tuple[dict, dict]:
        """The shares of BPSK and QPSK windows answered correctly ("correct"),
        uncertain, or with another class, as recorded and after resampling to
        SAMPLES_PER_SYMBOL. The resampled windows at each end are left out."""
        rng = np.random.default_rng(samples_per_symbol)
        recorded, resampled = Counter(), Counter()
        factor = SAMPLES_PER_SYMBOL / samples_per_symbol
        for label, _ in product(("BPSK", "QPSK"), range(captures)):
            samples = rectangular_capture(label, samples_per_symbol, rng, 2)
            prepared = prepare_recording(recording(samples), PrepareSettings(factor=factor))
            for counts, values in (
                (recorded, samples),
                (resampled, prepared.recording.samples[SAMPLES_PER_WINDOW:-SAMPLES_PER_WINDOW]),
            ):
                labels = classify_recording(self.model, values).labels
                counts.update("correct" if item == label else item for item in labels)
        return tuple(
            {name: count / sum(counts.values()) for name, count in counts.items()}
            for counts in (recorded, resampled)
        )

    def test_resampling_twenty_samples_per_symbol_to_eight_helps(self):
        recorded, resampled = self.shares(20)
        self.assertLessEqual(recorded.get("correct", 0), 0.1)
        self.assertGreaterEqual(resampled["correct"], 0.5)
        # The other windows are uncertain, never the wrong class.
        self.assertLessEqual(set(resampled), {"correct", UNCERTAIN})

    def test_resampling_four_samples_per_symbol_to_eight_leaves_the_model_uncertain(self):
        recorded, resampled = self.shares(4)
        self.assertGreaterEqual(recorded["correct"], 0.3)
        self.assertGreaterEqual(resampled.get(UNCERTAIN, 0), 0.9)

if __name__ == "__main__":
    unittest.main()
