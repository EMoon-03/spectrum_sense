"""Integration regressions for recording uploads in the Streamlit app."""

from io import BytesIO
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from streamlit import config
from streamlit.testing.v1 import AppTest, element_tree

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import MAX_FILE_BYTES
from signal_config import SAMPLES_PER_SYMBOL, SAMPLES_PER_WINDOW
from spectrum_sense import generate_capture


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
# Older Streamlit releases cannot upload files through AppTest.
CAN_UPLOAD = hasattr(getattr(element_tree, "FileUploader", None), "set_value")


def named_upload(name: str, payload: bytes) -> BytesIO:
    """A stand-in for Streamlit's UploadedFile: a name and getvalue()."""
    upload = BytesIO(payload)
    upload.name = name
    return upload


def sigmf_files(samples: np.ndarray) -> tuple[bytes, bytes]:
    """SigMF metadata and cf32_le data for a 2.4 MS/s capture at 433.92 MHz."""
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 2.4e6, "core:version": "1.2.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": 433.92e6}],
        "annotations": [],
    }
    return json.dumps(meta).encode(), samples.astype(np.complex64).tobytes()


def sigmf_archive(meta: bytes, data: bytes) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in (("rec/rec.sigmf-meta", meta), ("rec/rec.sigmf-data", data)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return buffer.getvalue()


class AppTests(unittest.TestCase):
    def run_app(self, *uploads):
        """Run the app with uploads given as (file name, contents) pairs."""
        files = [named_upload(name, payload) for name, payload in uploads]
        with patch("streamlit.file_uploader", return_value=files):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def run_csv(self, payload: bytes):
        return self.run_app(("recording.csv", payload))

    @staticmethod
    def markdown_text(app) -> str:
        return " ".join(item.value for item in app.markdown)

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

    def test_upload_widget_limit_admits_every_file_the_parser_accepts(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        (uploader,) = app.get("file_uploader")
        limit_mb = uploader.proto.max_upload_size_mb
        # Browsers check 1,000,000 bytes per MB (1,048,576 on Windows); the
        # server checks its own option in 1,048,576-byte units.
        self.assertGreaterEqual(limit_mb * 1_000_000, MAX_FILE_BYTES)
        self.assertLess((limit_mb - 1) * 1_000_000, MAX_FILE_BYTES)
        self.assertLessEqual(limit_mb, config.get_option("server.maxUploadSize"))
        self.assertTrue(uploader.proto.multiple_files)
        for extension in (".csv", ".cu8", ".npy", ".sigmf", ".sigmf-meta", ".sigmf-data"):
            self.assertIn(extension, uploader.proto.type)
        instructions = self.markdown_text(app)
        self.assertIn(format_size(MAX_UPLOAD_BYTES), instructions)
        self.assertIn(f"{MAX_SAMPLES:,} I/Q samples", instructions)

    def test_full_length_lossless_export_is_accepted(self):
        windows = -(-MAX_SAMPLES // SAMPLES_PER_WINDOW)
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(7), windows=windows)
        app = self.run_csv(to_csv(capture.ravel()[:MAX_SAMPLES]).encode())
        self.assertEqual(len(app.error), 0)
        self.assertIn(f"**Samples:** {MAX_SAMPLES:,}", self.markdown_text(app))
        self.assertIn(
            f"Showing window 1 of {MAX_SAMPLES // SAMPLES_PER_WINDOW}",
            " ".join(caption.value for caption in app.caption),
        )
        self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})

    def test_malformed_csv_shows_error_without_crashing(self):
        app = self.run_csv(b'I,Q\n"1"0,2\n')
        self.assertEqual(len(app.error), 1)
        self.assertIn("invalid CSV format", app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), set())

    def test_zero_window_shows_error_and_preserves_signal_plots(self):
        app = self.run_csv(b"I,Q\n" + b"0,0\n" * SAMPLES_PER_WINDOW)
        self.assertEqual(len(app.error), 1)
        self.assertIn("only zeros", app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), {"time", "constellation", "fft"})

    def test_maximum_finite_iq_upload_has_no_numeric_warnings(self):
        maximum = repr(float(np.finfo(np.float64).max))
        payload = ("I,Q\n" + f"{maximum},{maximum}\n" * SAMPLES_PER_WINDOW).encode()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            app = self.run_csv(payload)
        numeric_warnings = [str(item.message) for item in recorded if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(numeric_warnings, [])
        self.assertEqual(len(app.error), 0)
        self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})

    def test_raw_sdr_upload_shows_format_and_windows(self):
        capture, _, _ = generate_capture("BPSK", np.random.default_rng(11), windows=2)
        components = np.stack([capture.ravel().real, capture.ravel().imag], axis=1)
        data = np.round(components / np.abs(components).max() * 32767).astype("<i2").tobytes()
        app = self.run_app(("capture.cs16", data))
        self.assertEqual(len(app.error), 0)
        details = self.markdown_text(app)
        self.assertIn("Raw complex int16, little-endian (.cs16)", details)
        self.assertIn(f"**Samples:** {2 * SAMPLES_PER_WINDOW:,}", details)
        self.assertNotIn("Sample rate:", details)
        self.assertIn("Showing window 1 of 2", " ".join(caption.value for caption in app.caption))
        self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})

    def test_sigmf_archive_and_pair_show_their_metadata(self):
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(12), windows=1)
        meta, data = sigmf_files(capture.ravel())
        for label, uploads, source in (
            ("archive", [("rec.sigmf", sigmf_archive(meta, data))], "SigMF archive (cf32_le)"),
            ("pair", [("rec.sigmf-data", data), ("rec.sigmf-meta", meta)], "SigMF recording (cf32_le)"),
        ):
            with self.subTest(upload=label):
                app = self.run_app(*uploads)
                self.assertEqual(len(app.error), 0)
                details = self.markdown_text(app)
                self.assertIn(source, details)
                self.assertIn("**Sample rate:** 2.4 million samples per second", details)
                self.assertIn("**Center frequency:** 433.92 MHz", details)
                timing_note = " ".join(item.value for item in app.info)
                self.assertIn(f"{SAMPLES_PER_SYMBOL} samples per symbol", timing_note)
                self.assertIn("sample rate does not change what the model expects", timing_note)
                self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})

    def test_invalid_uploads_show_one_error_without_charts(self):
        csv = to_csv(np.ones(SAMPLES_PER_WINDOW)).encode()
        for uploads, message in (
            ([("capture.cu8", bytes(2 * SAMPLES_PER_WINDOW + 1))], "whole number"),
            ([("rec.sigmf", b"not a tar archive" * 100)], "could not be read"),
            ([("a.csv", csv), ("b.csv", csv)], "2 files were uploaded"),
            ([("rec.sigmf-meta", b"{}")], "needs both files"),
        ):
            with self.subTest(names=[name for name, _ in uploads]):
                app = self.run_app(*uploads)
                self.assertEqual(len(app.error), 1)
                self.assertIn(message, app.error[0].value)
                self.assertEqual(self.uploaded_charts(app), set())

    @unittest.skipUnless(CAN_UPLOAD, "AppTest cannot upload files in this Streamlit version.")
    def test_real_uploader_accepts_a_sigmf_pair_and_keeps_it_across_reruns(self):
        capture, _, _ = generate_capture("BPSK", np.random.default_rng(13), windows=2)
        meta, data = sigmf_files(capture.ravel())
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        app.file_uploader[0].set_value(
            [
                ("Capture.sigmf-meta", meta, "application/octet-stream"),
                ("Capture.sigmf-data", data, "application/octet-stream"),
            ]
        )
        app = app.run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual(len(app.error), 0)
        self.assertIn("SigMF recording (cf32_le)", self.markdown_text(app))
        slider = next(item for item in app.slider if item.label == "Window to inspect")
        app = slider.set_value(2).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertIn("Showing window 2 of 2", " ".join(caption.value for caption in app.caption))
        self.assertEqual(self.uploaded_charts(app), {"scores", "time", "constellation", "fft"})


if __name__ == "__main__":
    unittest.main()
