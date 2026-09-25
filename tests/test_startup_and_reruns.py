"""Regression tests for import cost, the plotting CLI, and upload caching."""

from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

import iq_csv
import iq_formats
import spectrum_sense
from spectrum_sense import SAMPLES_PER_WINDOW, generate_capture


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class LazyPlottingTests(unittest.TestCase):
    def test_importing_spectrum_sense_does_not_load_matplotlib(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import spectrum_sense; "
                "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_line_still_writes_figures(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "spectrum_sense.py",
                    "--captures-per-class",
                    "10",
                    "--windows-per-capture",
                    "2",
                    "--output-dir",
                    output_dir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("example_signals.png", "confusion_matrix.png"):
                with self.subTest(figure=name):
                    image = Path(output_dir, name).read_bytes()
                    self.assertTrue(image.startswith(b"\x89PNG"))


class UploadCacheTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def start_app(self, payload: bytes):
        """Start the app with a CSV upload and count calls to the file loader.

        The whole-file classifier's call counter is kept in self.classifier, and
        self.uploads is the list that the patched uploader returns.
        """
        upload = BytesIO(payload)
        upload.name = "recording.csv"
        self.uploads = [upload]
        uploader = patch("streamlit.file_uploader", return_value=self.uploads)
        parser = patch("iq_formats.load_uploads", wraps=iq_formats.load_uploads)
        classifier = patch(
            "spectrum_sense.classify_recording", wraps=spectrum_sense.classify_recording
        )
        for patcher in (uploader, parser, classifier):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.classifier = spectrum_sense.classify_recording
        return AppTest.from_file(str(APP_PATH)).run(timeout=20), iq_formats.load_uploads

    def assert_no_exceptions(self, app):
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])

    def test_moving_the_window_slider_does_not_parse_or_classify_again(self):
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(5), windows=3)
        app, parser = self.start_app(iq_csv.to_csv(capture.ravel()).encode())
        self.assert_no_exceptions(app)
        for window_number in (2, 3):
            with self.subTest(window=window_number):
                slider = next(item for item in app.slider if item.label == "Window to inspect")
                app = slider.set_value(window_number).run(timeout=20)
                self.assert_no_exceptions(app)
                self.assertIn(
                    f"Showing window {window_number} of 3",
                    " ".join(caption.value for caption in app.caption),
                )
        self.assertEqual(parser.call_count, 1)
        self.assertEqual(self.classifier.call_count, 1)

    def test_a_different_file_is_classified_again(self):
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(5), windows=3)
        app, _ = self.start_app(iq_csv.to_csv(capture.ravel()).encode())
        self.assert_no_exceptions(app)
        self.assertIn("The timeline shows 3 windows", " ".join(c.value for c in app.caption))
        shorter = BytesIO(iq_csv.to_csv(capture.ravel()[: 2 * SAMPLES_PER_WINDOW]).encode())
        shorter.name = "recording.csv"
        self.uploads[:] = [shorter]
        app = app.run(timeout=20)
        self.assert_no_exceptions(app)
        self.assertIn("The timeline shows 2 windows", " ".join(c.value for c in app.caption))
        self.assertEqual(self.classifier.call_count, 2)

    def test_invalid_upload_error_is_not_cached(self):
        payload = b"I,Q\nwrong,2\n" + b"1,2\n" * SAMPLES_PER_WINDOW
        app, parser = self.start_app(payload)
        for run in (1, 2):
            with self.subTest(run=run):
                if run > 1:
                    app = app.run(timeout=20)
                self.assert_no_exceptions(app)
                self.assertEqual(len(app.error), 1)
                self.assertIn("Line 2", app.error[0].value)
                self.assertEqual(parser.call_count, run)


if __name__ == "__main__":
    unittest.main()
