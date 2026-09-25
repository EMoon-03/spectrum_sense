"""Tests for the "uncertain" answer, the probe signals, and whole-file classification."""

from collections import Counter
import csv
from io import StringIO
import math
import unittest
from unittest.mock import patch

import numpy as np

from signal_config import (
    CLASSES,
    DISTANCE_PERCENTILE,
    FREQ_OFFSET_RANGE,
    MIN_TOP_SCORE,
    NOT_CLASSIFIABLE,
    PROBE_KINDS,
    REASON_CLOSE_SCORES,
    REASON_UNFAMILIAR,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    SNR_DB_RANGE,
    UNCERTAIN,
)
from spectrum_sense import (
    Decision,
    FileClassification,
    GroupRates,
    build_demo_model,
    classify_features,
    classify_recording,
    complex_gaussian,
    decide,
    decision_counts,
    decision_order,
    extract_features,
    fit_novelty,
    generate_capture,
    generate_probe_capture,
    make_dataset,
    novelty_distance,
    rates_by_snr,
    results_csv,
    train_classifier,
    uncertain_rates,
)


def original_generate_capture(label: str, rng: np.random.Generator, windows: int) -> np.ndarray:
    """generate_capture as it was before probes shared its transmitter code."""
    n = windows * SAMPLES_PER_WINDOW
    offset = float(rng.uniform(*FREQ_OFFSET_RANGE))
    snr_db = None if label == "noise" else float(rng.uniform(*SNR_DB_RANGE))
    if label == "noise":
        samples = complex_gaussian(rng, n)
    else:
        n_symbols = n // SAMPLES_PER_SYMBOL + 2
        i = rng.choice((-1.0, 1.0), size=n_symbols)
        if label == "BPSK":
            symbols = i.astype(complex)
        else:
            q = rng.choice((-1.0, 1.0), size=n_symbols)
            symbols = (i + 1j * q) / np.sqrt(2)
        start = int(rng.integers(0, SAMPLES_PER_SYMBOL))
        signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)[start : start + n]
        phase = float(rng.uniform(0, 2 * np.pi))
        signal = signal * np.exp(1j * (phase + 2 * np.pi * offset * np.arange(n)))
        samples = signal + complex_gaussian(rng, n) * 10 ** (-snr_db / 20)
    samples = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
    return samples.reshape(windows, SAMPLES_PER_WINDOW)


class ProbeGeneratorTests(unittest.TestCase):
    def test_shape_power_and_reproducibility(self):
        for kind in PROBE_KINDS:
            with self.subTest(kind=kind):
                first = generate_probe_capture(kind, np.random.default_rng(4), 3, snr_db=5)
                again = generate_probe_capture(kind, np.random.default_rng(4), 3, snr_db=5)
                other = generate_probe_capture(kind, np.random.default_rng(5), 3, snr_db=5)
                self.assertEqual(first.shape, (3, SAMPLES_PER_WINDOW))
                self.assertAlmostEqual(float(np.mean(np.abs(first) ** 2)), 1.0, places=12)
                np.testing.assert_array_equal(first, again)
                self.assertFalse(np.array_equal(first, other))

    def test_unknown_probe_type_is_rejected(self):
        for kind in ("QPSK", "8psk", ""):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    generate_probe_capture(kind, np.random.default_rng(0), 1, snr_db=10)

    def test_symbol_alphabets_without_noise_or_offset(self):
        def probe(kind):
            rng = np.random.default_rng(6)
            return generate_probe_capture(kind, rng, 2, snr_db=120, freq_offset_override=0).ravel()

        # 8PSK: eight phases, 45 degrees apart, relative to the random carrier phase.
        samples = probe("8PSK")
        steps = np.angle(samples * np.conj(samples[0])) / (2 * np.pi / 8)
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-4)
        self.assertEqual(len(np.unique(np.round(steps) % 8)), 8)
        np.testing.assert_allclose(np.abs(samples), 1, atol=1e-4)

        # 16-QAM: three amplitude rings in the ratio sqrt(2) : sqrt(10) : sqrt(18).
        amplitude = np.abs(probe("16-QAM"))
        rings = np.unique(np.round(amplitude / amplitude.min(), 3))
        np.testing.assert_allclose(rings, [1, np.sqrt(5), 3], atol=1e-3)

        # Tone: one constant sample value.
        samples = probe("tone")
        np.testing.assert_allclose(samples, samples[0], atol=1e-5)

    def test_carrier_offset_and_snr_match_generate_capture(self):
        # With a fixed offset, the tone advances by that many cycles per sample.
        tone = generate_probe_capture(
            "tone", np.random.default_rng(7), 1, snr_db=60, freq_offset_override=0.005
        )[0]
        advance = np.angle(np.mean(tone[1:] * np.conj(tone[:-1]))) / (2 * np.pi)
        self.assertAlmostEqual(advance, 0.005, places=6)
        # Random offsets come from the training range.
        for seed in range(20):
            tone = generate_probe_capture("tone", np.random.default_rng(seed), 1, snr_db=60)[0]
            advance = np.angle(np.mean(tone[1:] * np.conj(tone[:-1]))) / (2 * np.pi)
            self.assertTrue(FREQ_OFFSET_RANGE[0] <= advance <= FREQ_OFFSET_RANGE[1], advance)
        # At 0 dB the unit-power signal carries half of the normalized power.
        tone = generate_probe_capture(
            "tone", np.random.default_rng(8), 4, snr_db=0, freq_offset_override=0
        ).ravel()
        self.assertAlmostEqual(float(np.abs(np.mean(tone)) ** 2), 0.5, delta=0.02)

    def test_generate_capture_keeps_its_random_sequence(self):
        # The recorded CLI results depend on generate_capture's exact draws.
        for label in CLASSES:
            for seed in (0, 42):
                with self.subTest(label=label, seed=seed):
                    actual, _, _ = generate_capture(label, np.random.default_rng(seed), 3)
                    expected = original_generate_capture(label, np.random.default_rng(seed), 3)
                    np.testing.assert_array_equal(actual, expected)


class NoveltyTests(unittest.TestCase):
    def test_distance_to_the_nearest_class_matches_hand_computation(self):
        # Class a: mean (0, 0) and covariance diag(0.5, 2); class b is class a moved by (10, 0).
        base = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
        features = np.vstack([base, base + [10.0, 0.0]])
        novelty = fit_novelty(features, np.array(["a"] * 4 + ["b"] * 4), shrinkage=0.0)
        self.assertEqual(novelty.classes, ("a", "b"))
        np.testing.assert_allclose(novelty.means, [[0, 0], [10, 0]], atol=1e-12)
        points = np.array([[1.0, 2.0], [11.0, 0.0], [5.0, 0.0]])
        # sqrt(2 * 1 + 0.5 * 4), sqrt(2 * 1), and sqrt(2 * 25) from either class.
        np.testing.assert_allclose(
            novelty_distance(novelty, points), [2.0, np.sqrt(2), np.sqrt(50)], rtol=1e-12
        )
        np.testing.assert_allclose(novelty_distance(novelty, points[0]), [2.0], rtol=1e-12)

    def test_shrinkage_keeps_a_flat_class_invertible(self):
        # Points on a line have a singular covariance; shrinkage makes it invertible
        # while stepping off the line still costs far more than moving along it.
        t = np.linspace(-1, 1, 50)
        features = np.column_stack([t, 2 * t])
        novelty = fit_novelty(features, np.array(["line"] * 50))
        self.assertTrue(np.isfinite(novelty.precisions).all())
        along, across = novelty_distance(novelty, np.array([[0.5, 1.0], [0.5, -1.0]]))
        self.assertLess(along, 2)
        self.assertGreater(across, 10 * along)

    def test_each_class_needs_two_windows(self):
        with self.assertRaises(ValueError):
            fit_novelty(np.zeros((3, 2)), np.array(["a", "a", "b"]))


class DecideTests(unittest.TestCase):
    CLASSES = ("BPSK", "QPSK", "noise")

    def run_decide(self, scores, distance, threshold=4.0, min_top_score=0.6):
        return decide(
            self.CLASSES, scores, distance,
            distance_threshold=threshold, min_top_score=min_top_score,
        )

    def test_familiar_window_with_a_clear_winner_gets_that_class(self):
        self.assertEqual(
            self.run_decide([0.1, 0.85, 0.05], 2.0), Decision("QPSK", None, "QPSK", 0.85, 2.0)
        )

    def test_distant_window_is_uncertain_even_with_a_confident_score(self):
        self.assertEqual(
            self.run_decide([0.999, 0.0005, 0.0005], 9.5),
            Decision(UNCERTAIN, REASON_UNFAMILIAR, "BPSK", 0.999, 9.5),
        )

    def test_low_top_score_is_uncertain(self):
        self.assertEqual(
            self.run_decide([0.45, 0.2, 0.35], 1.0),
            Decision(UNCERTAIN, REASON_CLOSE_SCORES, "BPSK", 0.45, 1.0),
        )

    def test_distance_is_reported_before_close_scores(self):
        self.assertEqual(self.run_decide([0.4, 0.3, 0.3], 5.0).reason, REASON_UNFAMILIAR)

    def test_limits_themselves_are_accepted(self):
        self.assertEqual(self.run_decide([0.6, 0.3, 0.1], 4.0).label, "BPSK")

    def test_nan_distance_counts_as_unfamiliar(self):
        self.assertEqual(self.run_decide([0.9, 0.05, 0.05], math.nan).reason, REASON_UNFAMILIAR)

    def test_default_minimum_score_comes_from_the_settings(self):
        for top, expected in ((MIN_TOP_SCORE - 0.01, UNCERTAIN), (MIN_TOP_SCORE, "BPSK")):
            with self.subTest(top=top):
                decision = decide(self.CLASSES, [top, 0.0, 1 - top], 0.0, distance_threshold=1.0)
                self.assertEqual(decision.label, expected)

    def test_scores_must_match_the_classes(self):
        for scores in ([0.5, 0.5], [], [[0.2, 0.3, 0.5]]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    self.run_decide(scores, 1.0)


class RatesTests(unittest.TestCase):
    def test_false_rejects_and_accuracy_of_the_rest(self):
        decisions = [
            Decision("BPSK", None, "BPSK", 0.9, 1.0),
            Decision("QPSK", None, "QPSK", 0.9, 1.0),
            Decision(UNCERTAIN, REASON_UNFAMILIAR, "BPSK", 0.9, 9.0),
            Decision("noise", None, "noise", 0.9, 1.0),
        ]
        rejected, accuracy = uncertain_rates(decisions, ["BPSK", "BPSK", "QPSK", "noise"])
        self.assertEqual(rejected, 0.25)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_all_uncertain_has_no_accuracy(self):
        decisions = [Decision(UNCERTAIN, REASON_CLOSE_SCORES, "BPSK", 0.4, 1.0)]
        rejected, accuracy = uncertain_rates(decisions, ["BPSK"])
        self.assertEqual(rejected, 1.0)
        self.assertTrue(math.isnan(accuracy))

    def test_lengths_must_match(self):
        for truth in ([], ["BPSK", "QPSK"]):
            with self.subTest(truth=truth):
                with self.assertRaises(ValueError):
                    uncertain_rates([Decision("BPSK", None, "BPSK", 0.9, 1.0)], truth)


class RatesBySnrTests(unittest.TestCase):
    def test_signal_windows_by_snr_band_then_noise(self):
        rows = [
            (-1.0, "BPSK", Decision(UNCERTAIN, REASON_CLOSE_SCORES, "BPSK", 0.5, 1.0)),
            (0.5, "QPSK", Decision("QPSK", None, "QPSK", 0.9, 1.0)),
            (1.9, "BPSK", Decision("QPSK", None, "QPSK", 0.9, 1.0)),
            (-2.0, "QPSK", Decision("QPSK", None, "QPSK", 0.9, 1.0)),
            (14.0, "BPSK", Decision("BPSK", None, "BPSK", 0.9, 1.0)),  # The top edge counts.
            (math.nan, "noise", Decision("noise", None, "noise", 0.9, 1.0)),
            (math.nan, "noise", Decision(UNCERTAIN, REASON_UNFAMILIAR, "noise", 0.9, 9.0)),
        ]
        snrs, truth, decisions = zip(*rows)
        self.assertEqual(
            rates_by_snr(decisions, truth, snrs),
            (
                GroupRates("BPSK and QPSK at -2 to 2 dB SNR", 4, 0.25, 2 / 3),
                GroupRates("BPSK and QPSK at 10 to 14 dB SNR", 1, 0.0, 1.0),
                GroupRates("Noise (no SNR)", 2, 0.5, 1.0),
            ),
        )

    def test_a_group_answered_only_uncertain_has_no_accuracy(self):
        (rates,) = rates_by_snr(
            [Decision(UNCERTAIN, REASON_UNFAMILIAR, "BPSK", 0.9, 9.0)], ["BPSK"], [7.0]
        )
        self.assertEqual((rates.group, rates.windows, rates.false_reject_rate), (
            "BPSK and QPSK at 6 to 10 dB SNR", 1, 1.0
        ))
        self.assertTrue(math.isnan(rates.accepted_accuracy))

    def test_lengths_must_match(self):
        decision = Decision("BPSK", None, "BPSK", 0.9, 1.0)
        for truth, snrs in ((["BPSK"], []), ([], [1.0]), (["BPSK", "QPSK"], [1.0, 2.0])):
            with self.subTest(truth=truth, snrs=snrs):
                with self.assertRaises(ValueError):
                    rates_by_snr([decision], truth, snrs)


class TrainClassifierTests(unittest.TestCase):
    def test_calibration_captures_are_separate_and_set_the_threshold(self):
        windows, labels, groups, _, _ = make_dataset(np.random.default_rng(3), 20, 4)
        features = np.vstack([extract_features(window) for window in windows])
        model, fit_rows, calibration_rows = train_classifier(features, labels, groups, seed=3)
        self.assertTrue(set(groups[fit_rows]).isdisjoint(groups[calibration_rows]))
        self.assertEqual(len(fit_rows) + len(calibration_rows), len(labels))
        self.assertEqual(set(model.classes), set(CLASSES))
        scaled = model.pipeline[0].transform(features[calibration_rows])
        distances = novelty_distance(model.novelty, scaled)
        self.assertEqual(model.distance_threshold, np.percentile(distances, DISTANCE_PERCENTILE))
        # np.percentile interpolates, so allow one window beyond the exact share.
        self.assertLessEqual(
            np.mean(distances > model.distance_threshold),
            1 - DISTANCE_PERCENTILE / 100 + 1 / len(distances),
        )
        self.assertEqual(model.min_top_score, MIN_TOP_SCORE)


class DemoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = build_demo_model()

    def test_captures_and_measurements_are_consistent(self):
        demo = self.demo
        self.assertEqual(demo.fit_captures + demo.calibration_captures + demo.test_captures, 300)
        self.assertEqual(demo.test_windows, 4 * demo.test_captures)
        self.assertLess(demo.false_reject_rate, 0.1)
        self.assertGreater(demo.accepted_accuracy, 0.95)
        self.assertGreater(demo.test_accuracy, 0.95)
        self.assertEqual(set(demo.probe_flag_rates), {5.0, 10.0})
        for rates in demo.probe_flag_rates.values():
            self.assertEqual(tuple(rates), PROBE_KINDS)
            self.assertTrue(all(0 <= rate <= 1 for rate in rates.values()))
        self.assertEqual(demo.probe_windows, 100)

    def test_rates_by_snr_cover_every_held_out_window(self):
        demo = self.demo
        groups = demo.rates_by_snr
        self.assertEqual(sum(rates.windows for rates in groups), demo.test_windows)
        self.assertAlmostEqual(
            sum(rates.windows * rates.false_reject_rate for rates in groups) / demo.test_windows,
            demo.false_reject_rate,
        )
        self.assertEqual(groups[-1].group, "Noise (no SNR)")
        # The About tab says uncertain answers and errors are most common near
        # the bottom of the training range.
        signal = groups[:-1]
        self.assertTrue(signal[0].group.startswith(f"BPSK and QPSK at {SNR_DB_RANGE[0]:g} to"))
        self.assertEqual(signal[0].false_reject_rate, max(r.false_reject_rate for r in signal))
        self.assertEqual(signal[0].accepted_accuracy, min(r.accepted_accuracy for r in signal))

    def test_rebuilding_gives_the_same_model(self):
        again = build_demo_model()
        self.assertEqual(again.model.distance_threshold, self.demo.model.distance_threshold)
        self.assertEqual(again.false_reject_rate, self.demo.false_reject_rate)
        self.assertEqual(again.probe_flag_rates, self.demo.probe_flag_rates)

    def test_distance_check_catches_a_tone_that_the_scores_call_bpsk(self):
        # The scores alone confidently call an unmodulated tone BPSK; the
        # distance check flags it. The probe seed differs from the demo's.
        model = self.demo.model
        tone = generate_probe_capture("tone", np.random.default_rng(11), 25, snr_db=10)
        scores, decisions = classify_features(model, np.vstack([extract_features(w) for w in tone]))
        self.assertGreater(np.mean(scores.max(axis=1) >= MIN_TOP_SCORE), 0.9)
        self.assertGreater(np.mean([item.leading == "BPSK" for item in decisions]), 0.9)
        self.assertGreater(np.mean([item.reason == REASON_UNFAMILIAR for item in decisions]), 0.9)


class SnrBehaviourTests(unittest.TestCase):
    """How "uncertain" behaves with SNR, as the README and the About tab
    describe it. Each point uses 25 captures of 4 windows from a fixed seed."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_demo_model().model

    def outcomes(self, kind: str, snr_db: float) -> Counter:
        """How the demo model answers 100 windows of a trained class or probe."""
        rng = np.random.default_rng(1234)
        if kind in PROBE_KINDS:
            captures = [generate_probe_capture(kind, rng, 4, snr_db=snr_db) for _ in range(25)]
        else:
            captures = [generate_capture(kind, rng, 4, snr_db_override=snr_db)[0] for _ in range(25)]
        features = np.vstack([extract_features(window) for window in np.vstack(captures)])
        _, decisions = classify_features(self.model, features)
        return Counter(item.label for item in decisions)

    def share(self, kind: str, snr_db: float, label: str = UNCERTAIN) -> float:
        return self.outcomes(kind, snr_db)[label] / 100

    def test_probes_inside_the_training_range(self):
        for snr in range(-2, 15, 2):
            with self.subTest(snr=snr):
                self.assertGreaterEqual(self.share("tone", snr), 0.95)
                if snr >= 4:
                    self.assertGreater(self.share("16-QAM", snr), 0.5)
        for snr in (0, 2, 4, 6):
            with self.subTest(kind="8PSK", snr=snr):
                self.assertLessEqual(self.share("8PSK", snr), 0.1)
        self.assertTrue(0.4 <= self.share("8PSK", 10) <= 0.8)  # "Roughly 60%".
        for snr in (12, 14):
            with self.subTest(kind="8PSK", snr=snr):
                self.assertGreaterEqual(self.share("8PSK", snr), 0.95)

    def test_below_the_training_range_flags_say_nothing_about_signal_type(self):
        for kind in ("BPSK", "QPSK", *PROBE_KINDS):
            for snr in (-8, -6, -4):
                with self.subTest(kind=kind, snr=snr):
                    self.assertGreaterEqual(self.share(kind, snr), 0.8)
            if kind != "tone":
                with self.subTest(kind=kind, snr=-12):
                    self.assertGreater(self.share(kind, -12, "noise"), 0.7)

    def test_known_signals_above_the_training_range_are_often_flagged(self):
        for kind in ("BPSK", "QPSK"):
            with self.subTest(kind=kind):
                self.assertGreater(self.share(kind, 20), 0.5)

    def test_bpsk_at_the_bottom_of_the_range_in_the_explore_tab(self):
        # The Explore tab makes one window from the seed, with no carrier offset
        # unless the user sets one.
        answers = []
        for seed in range(100):
            capture, _, _ = generate_capture(
                "BPSK", np.random.default_rng(seed), windows=1,
                snr_db_override=SNR_DB_RANGE[0], freq_offset_override=0.0,
            )
            _, (decision,) = classify_features(self.model, extract_features(capture[0]))
            answers.append(decision.label)
        uncertain = answers.count(UNCERTAIN) / len(answers)
        self.assertTrue(0.6 <= uncertain <= 0.9, uncertain)  # "About three in four".
        self.assertGreater(answers.count("QPSK"), 0)  # "Some of the rest are called QPSK".


class ClassifyRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_demo_model(captures_per_class=20, probe_snrs_db=()).model

    def known_samples(self, windows: int) -> np.ndarray:
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(9), windows)
        return capture.ravel()

    def test_every_complete_window_is_decided_once(self):
        samples = np.concatenate([self.known_samples(3), np.ones(100)])
        with patch.object(
            self.model.pipeline, "predict_proba", wraps=self.model.pipeline.predict_proba
        ) as predict:
            result = classify_recording(self.model, samples)
        self.assertEqual(predict.call_count, 1)
        np.testing.assert_array_equal(result.starts, np.arange(3) * SAMPLES_PER_WINDOW)
        self.assertEqual(result.ignored_samples, 100)
        self.assertEqual(result.classes, self.model.classes)
        np.testing.assert_allclose(result.scores.sum(axis=1), 1)
        features = np.vstack([extract_features(w) for w in samples[:-100].reshape(3, -1)])
        scores, decisions = classify_features(self.model, features)
        np.testing.assert_allclose(result.scores, scores)
        self.assertEqual(result.labels, tuple(item.label for item in decisions))
        np.testing.assert_allclose(result.distances, [item.distance for item in decisions])

    def test_zero_window_is_not_classifiable(self):
        samples = self.known_samples(3)
        samples[SAMPLES_PER_WINDOW : 2 * SAMPLES_PER_WINDOW] = 0
        result = classify_recording(self.model, samples)
        self.assertEqual(result.labels[1], NOT_CLASSIFIABLE)
        self.assertIn("only zeros", result.reasons[1])
        self.assertTrue(np.isnan(result.scores[1]).all())
        self.assertTrue(np.isnan(result.distances[1]))
        for index in (0, 2):
            self.assertNotEqual(result.labels[index], NOT_CLASSIFIABLE)
            self.assertTrue(np.isfinite(result.scores[index]).all())
        self.assertEqual(result.ignored_samples, 0)

    def test_all_zero_recording_needs_no_model_call(self):
        with patch.object(self.model.pipeline, "predict_proba") as predict:
            result = classify_recording(self.model, np.zeros(2 * SAMPLES_PER_WINDOW + 5))
        predict.assert_not_called()
        self.assertEqual(result.labels, (NOT_CLASSIFIABLE, NOT_CLASSIFIABLE))
        self.assertEqual(result.ignored_samples, 5)

    def test_short_or_multidimensional_input_is_rejected(self):
        for samples in (np.ones(SAMPLES_PER_WINDOW - 1), np.ones((2, SAMPLES_PER_WINDOW))):
            with self.subTest(shape=samples.shape):
                with self.assertRaises(ValueError):
                    classify_recording(self.model, samples)


def file_result(labels, scores, classes=("BPSK", "QPSK", "noise")) -> FileClassification:
    count = len(labels)
    reasons = tuple(REASON_UNFAMILIAR if label == UNCERTAIN else "" for label in labels)
    distances = np.array([np.nan if label == NOT_CLASSIFIABLE else 1.5 for label in labels])
    return FileClassification(
        tuple(classes), np.arange(count) * SAMPLES_PER_WINDOW, tuple(labels), reasons,
        np.asarray(scores, dtype=float), distances, 0,
    )


class CountsAndCsvTests(unittest.TestCase):
    def test_display_order(self):
        self.assertEqual(
            decision_order(("QPSK", "noise", "BPSK")),
            ("noise", "BPSK", "QPSK", UNCERTAIN, NOT_CLASSIFIABLE),
        )

    def test_counts_list_every_class_and_uncertain(self):
        scores = [[0.9, 0.05, 0.05]] * 3
        result = file_result(["BPSK", UNCERTAIN, "BPSK"], scores)
        self.assertEqual(
            decision_counts(result),
            [("noise", 0), ("BPSK", 2), ("QPSK", 0), (UNCERTAIN, 1)],
        )
        nan_row = [np.nan] * 3
        result = file_result(["QPSK", NOT_CLASSIFIABLE], [[0.1, 0.8, 0.1], nan_row])
        self.assertEqual(decision_counts(result)[-1], (NOT_CLASSIFIABLE, 1))

    def test_results_csv_has_one_row_per_window_with_scores_by_class(self):
        result = file_result(
            ["BPSK", UNCERTAIN, NOT_CLASSIFIABLE],
            [[0.7, 0.2, 0.1], [0.25, 0.25, 0.5], [np.nan] * 3],
        )
        rows = list(csv.reader(StringIO(results_csv(result))))
        self.assertEqual(
            rows[0],
            ["window", "start_sample", "decision", "reason", "distance",
             "score_noise", "score_BPSK", "score_QPSK"],
        )
        self.assertEqual(rows[1], ["1", "0", "BPSK", "", "1.5", "0.1", "0.7", "0.2"])
        self.assertEqual(
            rows[2], ["2", "1024", UNCERTAIN, REASON_UNFAMILIAR, "1.5", "0.5", "0.25", "0.25"]
        )
        self.assertEqual(rows[3], ["3", "2048", NOT_CLASSIFIABLE, "", "", "", "", ""])
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
