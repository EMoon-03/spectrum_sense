"""Integration regressions for CSV uploads in the Streamlit app."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from streamlit.testing.v1 import AppTest

from signal_config import SAMPLES_PER_WINDOW


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def run_app(self, payload=None):
        upload = None if payload is None else BytesIO(payload)
        with patch("streamlit.file_uploader", return_value=upload):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    @staticmethod
    def uploaded_charts(app):
        return {
            chart.proto.id.split("-uploaded-", 1)[1]
            for chart in app.get("plotly_chart")
            if "-uploaded-" in chart.proto.id
        }

    def test_normal_startup_shows_signal_and_prediction(self):
        app = self.run_app()
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertTrue(any(metric.label == "Demo model prediction" for metric in app.metric))

    def test_malformed_csv_shows_error_without_crashing(self):
        app = self.run_app(b'I,Q\n"1"0,2\n')
        self.assertEqual(len(app.error), 1)
        self.assertIn("invalid CSV format", app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), set())

    def test_zero_window_shows_error_and_preserves_signal_plots(self):
        app = self.run_app(b"I,Q\n" + b"0,0\n" * SAMPLES_PER_WINDOW)
        self.assertEqual(len(app.error), 1)
        self.assertIn("only zeros", app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), {"time", "constellation", "fft"})

    def test_maximum_finite_iq_upload_has_no_numeric_warnings(self):
        maximum = repr(float(np.finfo(np.float64).max))
        payload = ("I,Q\n" + f"{maximum},{maximum}\n" * SAMPLES_PER_WINDOW).encode()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            app = self.run_app(payload)
        numeric_warnings = [str(item.message) for item in recorded if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(numeric_warnings, [])
        self.assertEqual(len(app.error), 0)
        self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})


if __name__ == "__main__":
    unittest.main()
