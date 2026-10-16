"""Possible causes: each check's margin, as causes.py states it."""

from __future__ import annotations

import unittest

import numpy as np

from causes import (
    CARRIER_CORRELATION,
    CENTER_COMPONENT,
    POWER_STEP_DB,
    center_component,
    causes_summary,
    possible_causes,
    power_step_db,
)
from real_example import EXAMPLE_PATH
from iq_formats import load_recording
from signal_config import CLASSES, SAMPLES_PER_WINDOW, UNCERTAIN
from spectrum_sense import extract_features, generate_capture, generate_probe_capture


def keys(window) -> list[str]:
    try:
        features = extract_features(window)
    except ValueError:
        features = None
    return [cause.key for cause in possible_causes(window, features)]


class MarginTests(unittest.TestCase):
    """The 1,500 training-like windows the module docstring describes."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.windows = [
            generate_capture(label, rng, 1, snr_db_override=snr)[0][0]
            for label in CLASSES
            for snr in (-2, 2, 6, 10, 14)
            for _ in range(100)
        ]

    def test_training_windows_stay_well_inside_every_limit(self):
        steps = [power_step_db(w) for w in self.windows]
        carriers = [extract_features(w)[5] for w in self.windows]
        centers = [center_component(w) for w in self.windows]
        self.assertLess(max(steps), 2.5)
        self.assertLess(max(steps), POWER_STEP_DB)
        self.assertLess(max(carriers), 0.25)
        self.assertLess(max(carriers), CARRIER_CORRELATION)
        self.assertLess(max(centers), 0.25)
        self.assertLess(max(centers), CENTER_COMPONENT)
        self.assertFalse(any(keys(w) for w in self.windows))

    def test_half_burst_half_gap_windows_change_power(self):
        rng = np.random.default_rng(1)
        hits = 0
        for trial in range(100):
            burst = generate_capture(("BPSK", "QPSK")[trial % 2], rng, 1, snr_db_override=6)[0][0]
            gap = generate_capture("noise", rng, 1)[0][0] * 10 ** (-6 / 20)
            cut = int(rng.integers(256, 768))
            window = np.concatenate([burst[:cut], gap[cut:]])
            hits += "power-step" in keys(window)
        self.assertGreaterEqual(hits, 95)

    def test_a_carrier_is_named_and_a_centered_one_twice(self):
        rng = np.random.default_rng(2)
        for window in generate_probe_capture("tone", rng, 20, snr_db=6):
            self.assertIn("bare-carrier", keys(window))
        for window in generate_probe_capture("tone", rng, 20, snr_db=6, freq_offset_override=0.0):
            self.assertEqual(keys(window), ["bare-carrier", "center-component"])

    def test_a_dc_offset_is_a_center_component(self):
        window = generate_capture("QPSK", np.random.default_rng(3), 1, snr_db_override=8)[0][0]
        self.assertEqual(keys(window), [])
        self.assertEqual(keys(window + 0.2), [])
        # An offset as strong as the signal is a carrier at 0 Hz in all but name,
        # so it is named both ways; the center-component text says so.
        self.assertEqual(keys(window + 1.0), ["bare-carrier", "center-component"])
        self.assertEqual(keys(window + 3.0), ["bare-carrier", "center-component"])

    def test_a_silent_part_and_an_all_zero_window(self):
        window = generate_capture("QPSK", np.random.default_rng(4), 1)[0][0].copy()
        window[:128] = 0
        self.assertEqual(power_step_db(window), float("inf"))
        (cause,) = [c for c in possible_causes(window, None) if c.key == "power-step"]
        self.assertIn("a silent part above the quietest", cause.detail)
        zeros = np.zeros(SAMPLES_PER_WINDOW, dtype=complex)
        self.assertEqual(possible_causes(zeros, None), ())

    def test_the_bundled_recording(self):
        samples = load_recording(EXAMPLE_PATH.name, EXAMPLE_PATH.read_bytes()).samples
        windows = samples[: len(samples) // SAMPLES_PER_WINDOW * SAMPLES_PER_WINDOW]
        found = [keys(w) for w in windows.reshape(-1, SAMPLES_PER_WINDOW)]
        self.assertEqual(sum(k == ["power-step"] for k in found), 20)
        self.assertEqual(sum(k == [] for k in found), 10)
        steps = sorted(power_step_db(w) for w in windows.reshape(-1, SAMPLES_PER_WINDOW))
        self.assertLess(steps[9], 1.5)  # inside a packet
        self.assertGreater(steps[10], 6)  # across a packet edge

    def test_extreme_and_nonfinite_samples(self):
        largest = np.finfo(np.float64).max
        window = generate_capture("QPSK", np.random.default_rng(5), 1)[0][0]
        extreme = window / np.max(np.abs(window)) * largest / 2
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            self.assertAlmostEqual(power_step_db(extreme), power_step_db(window), places=9)
            self.assertAlmostEqual(center_component(extreme), center_component(window), places=9)
            self.assertEqual(keys(extreme), keys(window))
        broken = window.copy()
        broken[3] = complex(np.nan, 0)
        self.assertEqual(possible_causes(broken, None), ())
        with self.assertRaises(ValueError):
            power_step_db(broken)

    def test_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            power_step_db(np.ones(7, dtype=complex))


class SummaryTests(unittest.TestCase):
    def test_counts_only_uncertain_windows(self):
        labels = [UNCERTAIN, UNCERTAIN, "QPSK", UNCERTAIN]
        causes = [["power-step"], [], ["power-step"], ["power-step", "bare-carrier"]]
        self.assertEqual(
            causes_summary(labels, causes),
            "Of the 3 uncertain windows, 2 change power within the window and 1 looks like "
            "an unmodulated carrier (a window can show more than one); 1 shows no measured cause.",
        )

    def test_none_and_all_explained(self):
        self.assertEqual(
            causes_summary([UNCERTAIN], [[]]),
            "None of the 1 uncertain window shows a measured cause: their features lie "
            "outside what the model learned in ways these checks do not name.",
        )
        self.assertEqual(
            causes_summary([UNCERTAIN, UNCERTAIN], [["center-component"], ["center-component"]]),
            "Of the 2 uncertain windows, 2 have a steady component at 0 Hz; every one shows "
            "at least one cause.",
        )
        self.assertEqual(causes_summary(["BPSK"], [[]]), "No window is uncertain.")

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            causes_summary([UNCERTAIN], [])


if __name__ == "__main__":
    unittest.main()
