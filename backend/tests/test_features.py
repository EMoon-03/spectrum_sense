"""Regression tests for reliable I/Q feature extraction."""

import unittest

import numpy as np

from spectrum_sense import (
    CLASSES,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    extract_features,
    generate_capture,
    line_strength,
)


class FeatureExtractionTests(unittest.TestCase):
    def extract_checked(self, window):
        # Tiny contributions may underflow harmlessly; invalid operations,
        # overflow, and division by zero must never be hidden by the tests.
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            features = extract_features(window)
        self.assertEqual(features.shape, (8,))
        self.assertTrue(np.isfinite(features).all())
        return features

    def test_constant_nonzero_window_has_known_features(self):
        expected = [1, 1, 0, 1, 1, 1, 1, 1]
        for value in (1 + 0j, 3 + 4j, -3 - 4j):
            with self.subTest(value=value):
                features = self.extract_checked(
                    np.full(SAMPLES_PER_WINDOW, value, dtype=np.complex128)
                )
                np.testing.assert_allclose(features, expected, rtol=1e-11, atol=1e-12)

    def test_features_are_invariant_to_extreme_amplitude_scaling(self):
        rng = np.random.default_rng(17)
        window = rng.standard_normal(SAMPLES_PER_WINDOW) + 1j * rng.standard_normal(
            SAMPLES_PER_WINDOW
        )
        expected = self.extract_checked(window)
        for scale in (1e50, 1e300, 1e-200, 1e-300):
            with self.subTest(scale=scale):
                actual = self.extract_checked(window * scale)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_extreme_finite_components_have_known_features(self):
        largest = np.finfo(np.float64).max
        smallest = float.fromhex("0x0.0000000000001p-1022")
        for value in (
            complex(largest, 0),
            complex(largest, largest),
            complex(-largest, largest),
            complex(smallest, 0),
            complex(smallest, smallest),
        ):
            with self.subTest(value=value):
                features = self.extract_checked(
                    np.full(SAMPLES_PER_WINDOW, value, dtype=np.complex128)
                )
                np.testing.assert_allclose(
                    features, [1, 1, 0, 1, 1, 1, 1, 1], rtol=1e-11, atol=1e-12
                )

    def test_all_zero_window_is_rejected(self):
        with self.assertRaises(ValueError):
            self.extract_checked(np.zeros(SAMPLES_PER_WINDOW, dtype=np.complex128))

    def test_nonfinite_components_are_rejected(self):
        for value in (
            complex(np.nan, 0),
            complex(0, np.nan),
            complex(np.inf, 0),
            complex(0, -np.inf),
        ):
            with self.subTest(value=value):
                window = np.ones(SAMPLES_PER_WINDOW, dtype=np.complex128)
                window[SAMPLES_PER_WINDOW // 2] = value
                with self.assertRaises(ValueError):
                    self.extract_checked(window)

    def test_windows_too_short_for_symbol_correlation_are_rejected(self):
        for length in (0, 1, SAMPLES_PER_SYMBOL):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    self.extract_checked(np.ones(length, dtype=np.complex128))

    def test_minimum_length_window_is_accepted(self):
        features = self.extract_checked(
            np.ones(SAMPLES_PER_SYMBOL + 1, dtype=np.complex128)
        )
        np.testing.assert_allclose(
            features, [1, 1, 0, 1, 1, 1, 1, 1], rtol=1e-11, atol=1e-12
        )

    def test_scalar_and_multidimensional_inputs_are_rejected(self):
        for shape in ((), (1, SAMPLES_PER_WINDOW), (SAMPLES_PER_WINDOW, 1)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.extract_checked(np.ones(shape, dtype=np.complex128))

    def test_caller_samples_are_preserved_including_read_only_arrays(self):
        rng = np.random.default_rng(23)
        original = rng.standard_normal(SAMPLES_PER_WINDOW) + 1j * rng.standard_normal(
            SAMPLES_PER_WINDOW
        )
        for writable in (True, False):
            with self.subTest(writable=writable):
                window = original.copy()
                window.flags.writeable = writable
                self.extract_checked(window)
                np.testing.assert_array_equal(window, original)
                self.assertEqual(window.flags.writeable, writable)

    def test_each_line_feature_sees_the_signal_it_is_for(self):
        # Squaring BPSK, or raising QPSK to the fourth power, leaves a spectral
        # line wherever the carrier offset puts it. An earlier version removed
        # an estimated carrier and averaged, which washed the line out: the
        # median clean QPSK window scored 0.04 on the fourth-power line.
        def median(label):
            rng = np.random.default_rng(14)
            rows = [
                extract_features(generate_capture(
                    label, rng, 1, snr_db_override=14, freq_offset_override=0.006
                )[0][0])
                for _ in range(30)
            ]
            return np.median(rows, axis=0)

        bpsk, qpsk, noise = median("BPSK"), median("QPSK"), median("noise")
        self.assertGreater(bpsk[0], 0.9)  # squared-signal line
        self.assertGreater(bpsk[1], 0.85)  # fourth-power line
        self.assertGreater(qpsk[1], 0.85)
        self.assertLess(qpsk[0], 0.3)
        self.assertLess(noise[0], 0.3)
        self.assertLess(noise[1], 0.6)

    def test_line_strength_is_exact_wherever_the_line_lies(self):
        k = np.arange(SAMPLES_PER_WINDOW)
        for frequency in (0.0, 0.1234567, -0.3999, 0.49, 0.5, 1 / 3):
            with self.subTest(frequency=frequency):
                self.assertAlmostEqual(line_strength(np.exp(2j * np.pi * frequency * k)), 1.0, places=10)
        # Half the power in a line: the strength is the line's share of amplitude.
        rng = np.random.default_rng(3)
        noise = (rng.standard_normal(SAMPLES_PER_WINDOW) + 1j * rng.standard_normal(SAMPLES_PER_WINDOW)) / 2
        tone = 0.5 * np.exp(2j * np.pi * 0.2071 * k)
        self.assertAlmostEqual(line_strength(tone + noise * 0.01), 0.5, delta=0.01)

    def test_a_frequency_shift_changes_no_feature(self):
        k = np.arange(SAMPLES_PER_WINDOW)
        for label in CLASSES:
            window = generate_capture(label, np.random.default_rng(21), 1)[0][0]
            for shift in (0.2, -0.137, 0.0003):
                with self.subTest(label=label, shift=shift):
                    np.testing.assert_allclose(
                        extract_features(window * np.exp(2j * np.pi * shift * k)),
                        extract_features(window), rtol=1e-9, atol=1e-12,
                    )

    def test_generated_captures_for_each_class_have_finite_features(self):
        rng = np.random.default_rng(42)
        for label in CLASSES:
            windows, _, _ = generate_capture(label, rng, windows=3)
            for index, window in enumerate(windows):
                with self.subTest(label=label, window=index):
                    self.extract_checked(window)


if __name__ == "__main__":
    unittest.main()
