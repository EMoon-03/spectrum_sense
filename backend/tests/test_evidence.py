"""Feature evidence: exact, read-only, and true to the window it explains."""

from __future__ import annotations

import unittest

import numpy as np

from signal_config import FEATURE_NAMES, FEATURES, REASON_UNFAMILIAR, UNCERTAIN
from spectrum_sense import (
    build_demo_model,
    classify_features,
    explain_features,
    extract_features,
    generate_capture,
    generate_probe_capture,
    novelty_distance,
)
from summaries import evidence_summary


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_demo_model(probe_snrs_db=()).model

    def windows(self):
        rng = np.random.default_rng(31)
        for label in ("noise", "BPSK", "QPSK"):
            for snr in (-2.0, 6.0, 14.0):
                yield label, generate_capture(label, rng, 1, snr_db_override=snr)[0][0]
        for kind in ("8PSK", "16-QAM", "tone"):
            yield kind, generate_probe_capture(kind, rng, 1, snr_db=8)[0]

    def test_features_are_named_in_column_order(self):
        self.assertEqual(len(FEATURES), len(extract_features(np.ones(64, dtype=complex))))
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))

    def test_shares_split_the_squared_distance_exactly(self):
        for label, window in self.windows():
            with self.subTest(label=label):
                features = extract_features(window)
                evidence = explain_features(self.model, features)
                distance = novelty_distance(
                    self.model.novelty, self.model.pipeline[0].transform(features[None, :])
                )[0]
                self.assertAlmostEqual(evidence.distance, distance, places=9)
                self.assertTrue((evidence.distance_shares >= 0).all())
                self.assertAlmostEqual(float(evidence.distance_shares.sum()), 1.0, places=9)

    def test_the_expected_value_gives_the_distance_it_promises(self):
        scaler, novelty = self.model.pipeline[0], self.model.novelty
        for label, window in self.windows():
            features = extract_features(window)
            evidence = explain_features(self.model, features)
            k = novelty.classes.index(evidence.nearest)
            for index in range(len(features)):
                with self.subTest(label=label, feature=FEATURE_NAMES[index]):
                    changed = features.copy()
                    changed[index] = evidence.expected[index]
                    offset = scaler.transform(changed[None, :])[0] - novelty.means[k]
                    distance = np.sqrt(offset @ novelty.precisions[k] @ offset)
                    self.assertAlmostEqual(distance, evidence.distance_if_expected[index], places=9)
                    self.assertLessEqual(evidence.distance_if_expected[index], evidence.distance + 1e-12)
                    # No other value of this feature brings the window closer.
                    for nudge in (-0.01, 0.01):
                        moved = changed.copy()
                        moved[index] += nudge
                        offset = scaler.transform(moved[None, :])[0] - novelty.means[k]
                        self.assertGreaterEqual(offset @ novelty.precisions[k] @ offset, distance**2)

    def test_score_push_matches_the_logistic_regression(self):
        logistic = self.model.pipeline[-1]
        for label, window in self.windows():
            with self.subTest(label=label):
                features = extract_features(window)
                evidence = explain_features(self.model, features)
                scores, (decision,) = classify_features(self.model, features)
                classes = list(self.model.classes)
                self.assertEqual(evidence.leading, decision.leading)
                lead, runner = classes.index(evidence.leading), classes.index(evidence.runner_up)
                log_odds = np.log(scores[0][lead] / scores[0][runner])
                intercepts = logistic.intercept_[lead] - logistic.intercept_[runner]
                self.assertAlmostEqual(evidence.score_push.sum() + intercepts, log_odds, places=6)

    def test_explaining_changes_no_decision(self):
        rng = np.random.default_rng(5)
        windows = np.vstack([generate_capture("QPSK", rng, 4)[0] for _ in range(5)])
        features = np.vstack([extract_features(w) for w in windows])
        before = classify_features(self.model, features)[1]
        for row in features:
            explain_features(self.model, row)
        self.assertEqual(classify_features(self.model, features)[1], before)

    def test_a_tone_is_far_mostly_in_symbol_spaced_correlation(self):
        index = FEATURE_NAMES.index("symbol-spaced correlation")
        tones = generate_probe_capture("tone", np.random.default_rng(8), 25, snr_db=10)
        for window in tones:
            evidence = explain_features(self.model, extract_features(window))
            self.assertGreater(evidence.distance_shares[index], 0.8)

    def test_8psk_is_far_mostly_in_the_fourth_power_line(self):
        index = FEATURE_NAMES.index("fourth-power line")
        windows = generate_probe_capture("8PSK", np.random.default_rng(9), 25, snr_db=10)
        shares = [
            explain_features(self.model, extract_features(w)).distance_shares[index] for w in windows
        ]
        self.assertGreater(np.median(shares), 0.4)

    def test_one_feature_must_not_be_a_single_share_story_by_default(self):
        # A summary names features until they hold half the distance, at most three.
        evidence = explain_features(
            self.model, extract_features(generate_capture("BPSK", np.random.default_rng(1), 1)[0][0])
        )
        text = evidence_summary(evidence, self.model.distance_threshold)
        named = [name for name in FEATURE_NAMES if f"the {name} (" in text]
        self.assertTrue(1 <= len(named) <= 3)
        shares = sorted(evidence.distance_shares, reverse=True)
        self.assertTrue(sum(shares[: len(named)]) >= 0.5 or len(named) == 3)


class EvidenceSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_demo_model(probe_snrs_db=()).model

    def test_an_unfamiliar_window_gets_the_change_that_helps_most(self):
        tone = generate_probe_capture("tone", np.random.default_rng(3), 1, snr_db=8)[0]
        features = extract_features(tone)
        (decision,) = classify_features(self.model, features)[1]
        self.assertEqual((decision.label, decision.reason), (UNCERTAIN, REASON_UNFAMILIAR))
        evidence = explain_features(self.model, features)
        text = evidence_summary(evidence, self.model.distance_threshold)
        self.assertIn(f"Most of the distance to {evidence.nearest}, the nearest trained class", text)
        self.assertIn("symbol-spaced correlation", text)
        self.assertIn("changing that alone would take the distance from", text)
        self.assertTrue(text.endswith(f"the limit of {self.model.distance_threshold:.2f}."), text)
        self.assertNotIn("caused", text)

    def test_a_named_window_gets_the_feature_behind_its_lead(self):
        window = generate_capture("BPSK", np.random.default_rng(4), 1, snr_db_override=12)[0][0]
        evidence = explain_features(self.model, extract_features(window))
        text = evidence_summary(evidence, self.model.distance_threshold)
        self.assertIn("does most to put BPSK ahead of", text)
        self.assertNotIn("changing that alone", text)

    def test_wording_with_hand_made_evidence(self):
        class Fake:
            nearest, leading, runner_up = "QPSK", "QPSK", "BPSK"
            distance = 7.41
            values = np.array([0.1, 0.25, 0, 0, 0.69, 0, 0, 0])
            expected = np.array([0.1, 0.53, 0, 0, 0.55, 0, 0, 0])
            distance_shares = np.array([0, 0.44, 0.04, 0.03, 0.41, 0.06, 0.01, 0.01])
            distance_if_expected = np.array([7.41, 4.23, 7.4, 7.4, 5.41, 7.09, 7.4, 7.4])
            score_push = np.zeros(8)

        self.assertEqual(
            evidence_summary(Fake, 4.67),
            "Most of the distance to QPSK, the nearest trained class, lies in the "
            "fourth-power line (44%) and the lag-1 correlation (41%). Given the window's "
            "other features, the QPSK fit expects about 0.53 for the fourth-power line, "
            "not 0.25; changing that alone would take the distance from 7.41 to 4.23, "
            "inside the limit of 4.67.",
        )
        Fake.distance_if_expected = np.full(8, 5.0)
        Fake.distance_if_expected[1] = 4.9
        self.assertIn("to 4.90, still beyond the limit of 4.67.", evidence_summary(Fake, 4.67))


if __name__ == "__main__":
    unittest.main()
