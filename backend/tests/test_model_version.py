"""The model fingerprint covers everything that can change an answer."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spectrum_sense  # noqa: E402
from app import model  # noqa: E402


def _defined_in_spectrum_sense() -> set[str]:
    return {
        name
        for name, item in vars(spectrum_sense).items()
        if (inspect.isfunction(item) or inspect.isclass(item))
        and item.__module__ == spectrum_sense.__name__
    }


class FingerprintCoverageTests(unittest.TestCase):
    def test_every_function_and_class_is_fingerprinted_or_excused(self):
        fingerprinted = {item.__name__ for item in model.FINGERPRINTED}
        excused = set(model.NOT_FINGERPRINTED)
        self.assertFalse(fingerprinted & excused, "listed in both")
        missing = _defined_in_spectrum_sense() - fingerprinted - excused
        self.assertFalse(
            missing,
            f"Add {sorted(missing)} to FINGERPRINTED in app/model.py, or to "
            "NOT_FINGERPRINTED with the reason it cannot change an answer.",
        )

    def test_lists_name_only_things_that_exist(self):
        defined = _defined_in_spectrum_sense()
        self.assertLessEqual(set(model.NOT_FINGERPRINTED), defined)
        for item in model.FINGERPRINTED:
            self.assertIs(getattr(spectrum_sense, item.__name__), item)

    def test_every_excuse_gives_a_reason(self):
        for name, reason in model.NOT_FINGERPRINTED.items():
            with self.subTest(name=name):
                self.assertTrue(reason.strip())


class FingerprintChangeTests(unittest.TestCase):
    def test_fingerprint_is_stable_within_a_process(self):
        self.assertEqual(model._fingerprint(), model.MODEL_VERSION)
        self.assertRegex(model.MODEL_VERSION, r"^[0-9a-f]{16}$")

    def test_a_library_upgrade_changes_the_fingerprint(self):
        for library in (model.numpy, model.scipy, model.sklearn):
            with self.subTest(library=library.__name__):
                with patch.object(library, "__version__", "0.0.0-test"):
                    self.assertNotEqual(model._fingerprint(), model.MODEL_VERSION)

    def test_an_edit_to_decision_code_changes_the_fingerprint(self):
        real = inspect.getsource

        def edited(item):
            source = real(item)
            return source + "# edited\n" if item is spectrum_sense.novelty_distance else source

        with patch.object(model.inspect, "getsource", edited):
            self.assertNotEqual(model._fingerprint(), model.MODEL_VERSION)

    def test_a_training_setting_changes_the_fingerprint(self):
        with patch.object(model.config, "MODEL_SEED", model.config.MODEL_SEED + 1):
            self.assertNotEqual(model._fingerprint(), model.MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
