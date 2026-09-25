"""Integration regressions for recording uploads and accessibility in the Streamlit app."""

import base64
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile
import unicodedata
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest, element_tree

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import MAX_FILE_BYTES, load_uploads
from signal_config import SAMPLES_PER_SYMBOL, SAMPLES_PER_WINDOW
from spectrum_sense import generate_capture


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
# Older Streamlit releases cannot upload files through AppTest.
CAN_UPLOAD = hasattr(getattr(element_tree, "FileUploader", None), "set_value")
EXAMPLE_LABEL = "Download an example CSV"
# The text summary that must sit directly under each chart, by chart key.
SUMMARY_STARTS = {
    "scores": "Scores: ",
    "time": f"The time plot shows the first 128 of {SAMPLES_PER_WINDOW:,} samples.",
    "constellation": f"The I/Q plane shows {SAMPLES_PER_WINDOW // 2} points",
    "fft": f"Spectrum of {SAMPLES_PER_WINDOW:,} frequency bins",
}


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


def plotly_values(values) -> np.ndarray:
    """Numbers from a Plotly JSON spec: a list, or base64 data in Plotly 6."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values, dtype=float)


def ordered_children(block) -> list:
    return [block.children[index] for index in sorted(block.children)]


def sigmf_archive(meta: bytes, data: bytes) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in (("rec/rec.sigmf-meta", meta), ("rec/rec.sigmf-data", data)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return buffer.getvalue()


class AppTestCase(unittest.TestCase):
    """Helpers for running the app; the tests are in the subclasses."""

    def run_app(self, *uploads):
        """Run the app with uploads given as (file name, contents) pairs."""
        files = [named_upload(name, payload) for name, payload in uploads]
        with patch("streamlit.file_uploader", return_value=files):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def run_csv(self, payload: bytes):
        return self.run_app(("recording.csv", payload))

    def run_and_record_downloads(self, *uploads):
        """Run the app and return it with the data of each download button, by label."""
        downloads = {}
        download_button = st.download_button

        def recording_download_button(label, data, *args, **kwargs):
            downloads[label] = data
            return download_button(label, data, *args, **kwargs)

        with patch("streamlit.download_button", new=recording_download_button):
            app = self.run_app(*uploads)
        return app, downloads

    def chart_summaries(self, block) -> dict[str, str]:
        """Map each chart's key to the caption directly below it."""
        found = {}
        children = ordered_children(block)
        for position, child in enumerate(children):
            if child.type == "plotly_chart":
                chart_key = child.proto.id.split("-", 2)[2]
                following = children[position + 1] if position + 1 < len(children) else None
                self.assertEqual(getattr(following, "type", None), "caption", chart_key)
                found[chart_key] = following.value
            elif hasattr(child, "children"):
                found.update(self.chart_summaries(child))
        return found

    def assert_charts_have_summaries(self, block, section: str) -> dict[str, str]:
        summaries = self.chart_summaries(block)
        self.assertEqual(set(summaries), {f"{section}-{kind}" for kind in SUMMARY_STARTS})
        for kind, start in SUMMARY_STARTS.items():
            with self.subTest(section=section, chart=kind):
                self.assertTrue(summaries[f"{section}-{kind}"].startswith(start), summaries)
        return summaries

    def example_csv(self) -> str:
        _, downloads = self.run_and_record_downloads()
        return downloads[EXAMPLE_LABEL]

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

    @staticmethod
    def chart_spec(app, chart_key: str) -> dict:
        (chart,) = [item for item in app.get("plotly_chart") if item.proto.id.endswith(f"-{chart_key}")]
        return json.loads(chart.proto.spec)


class AppTests(AppTestCase):
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
        self.assertIn(EXAMPLE_LABEL, app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), set())

    def test_zero_window_shows_error_and_preserves_signal_plots(self):
        app = self.run_csv(b"I,Q\n" + b"0,0\n" * SAMPLES_PER_WINDOW)
        self.assertEqual(len(app.error), 1)
        self.assertIn("only zeros", app.error[0].value)
        # The file loaded, so the hint about loading a file does not apply.
        self.assertNotIn(EXAMPLE_LABEL, app.error[0].value)
        self.assertEqual(self.uploaded_charts(app), {"time", "constellation", "fft"})
        summaries = self.chart_summaries(app.tabs[1])
        self.assertIn("every sample is zero", summaries["uploaded-constellation"])
        self.assertIn("every sample is zero", summaries["uploaded-fft"])

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
                spectrum = self.assert_charts_have_summaries(app.tabs[1], "uploaded")["uploaded-fft"]
                self.assertRegex(spectrum, r"the center of the recorded band, about [\d.]+ MHz\)")

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
                self.assertIn(EXAMPLE_LABEL, app.error[0].value)
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


class AccessibilityTests(AppTestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def test_title_is_plain_text_and_headings_are_in_order(self):
        app = self.run_app()
        (title,) = app.title
        self.assertEqual(title.value, "SpectrumSense Signal Lab")
        self.assertEqual(
            [char for char in title.value if unicodedata.category(char) in ("So", "Mn", "Cf")], []
        )
        self.assertEqual(title.tag, "h1")
        # Each tab opens with a level-2 heading, so no level is skipped.
        self.assertEqual(len(app.subheader), 0)
        for tab in app.tabs:
            with self.subTest(tab=tab.label):
                first = ordered_children(tab)[0]
                self.assertEqual((first.type, first.tag), ("header", "h2"))

    def test_lab_and_upload_charts_each_have_a_summary_directly_below(self):
        app = self.run_csv(self.example_csv().encode())
        self.assertEqual(len(app.error), 0)
        lab = self.assert_charts_have_summaries(app.tabs[0], "lab")
        uploaded = self.assert_charts_have_summaries(app.tabs[1], "uploaded")
        # Neither recording has a sample rate, so no frequency is given in hertz.
        self.assertNotIn("Hz", lab["lab-fft"])
        self.assertNotIn("Hz", uploaded["uploaded-fft"])

    def test_score_bars_show_their_values_and_match_the_summary(self):
        app = self.run_app()
        (bars,) = self.chart_spec(app, "lab-scores")["data"]
        values = plotly_values(bars["y"])
        self.assertEqual(bars["text"], [f"{value:.2f}" for value in values])
        self.assertIn(bars["textposition"], ("outside", "auto"))
        caption = self.chart_summaries(app.tabs[0])["lab-scores"]
        listed = dict(re.findall(r"(\w+) (\d\.\d\d)[,.]", caption))
        self.assertEqual(listed, dict(zip(bars["x"], bars["text"])))
        self.assertEqual(sorted(listed.values(), reverse=True), list(listed.values()))

    def test_i_and_q_differ_by_line_style_not_only_color(self):
        i_trace, q_trace = self.chart_spec(self.run_app(), "lab-time")["data"]
        self.assertEqual((i_trace["name"], q_trace["name"]), ("I", "Q"))
        self.assertIn(i_trace["line"].get("dash", "solid"), ("solid", None))
        self.assertEqual(q_trace["line"]["dash"], "dash")
        self.assertNotEqual(i_trace["line"]["color"], q_trace["line"]["color"])

    def test_every_input_widget_has_help_text(self):
        app = self.run_csv(self.example_csv().encode())
        widgets = [
            *app.selectbox, *app.slider, *app.number_input, *app.file_uploader,
            *app.get("download_button"),
        ]
        labels = [widget.proto.label for widget in widgets]
        for label in ("Signal type", "Example seed", "Window to inspect", EXAMPLE_LABEL):
            self.assertIn(label, labels)
        for widget in widgets:
            with self.subTest(widget=widget.proto.label):
                self.assertRegex(widget.proto.help, r"^[A-Z].*\.$")

    def test_upload_tab_example_is_fixed_and_loads(self):
        first = self.example_csv()
        st.cache_data.clear()
        self.assertEqual(self.example_csv(), first)
        recording = load_uploads([("example.csv", first.encode())])
        self.assertEqual(len(recording.samples), 2 * SAMPLES_PER_WINDOW)

        app = self.run_csv(first.encode())
        upload_tab = app.tabs[1]
        labels = [button.proto.label for button in upload_tab.get("download_button")]
        self.assertIn(EXAMPLE_LABEL, labels)
        self.assertEqual(len(app.error), 0)
        self.assertIn("Showing window 1 of 2", " ".join(caption.value for caption in app.caption))


if __name__ == "__main__":
    unittest.main()
