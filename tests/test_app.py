"""Integration regressions for recording uploads and accessibility in the Streamlit app."""

import base64
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import tarfile
import unicodedata
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import plotly.colors
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest, element_tree

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, to_csv
from iq_formats import MAX_FILE_BYTES, load_uploads
from signal_config import (
    NOT_CLASSIFIABLE,
    PROBE_KINDS,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    UNCERTAIN,
)
from spectrogram import FLOOR_DB, MAX_TIME_STEPS, SEGMENT_LENGTH, compute_spectrogram
from spectrum_sense import (
    build_demo_model,
    classify_recording,
    decision_counts,
    decision_order,
    generate_capture,
    generate_probe_capture,
)
from summaries import spectrogram_summary


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
# Older Streamlit releases cannot upload files through AppTest.
CAN_UPLOAD = hasattr(getattr(element_tree, "FileUploader", None), "set_value")
EXAMPLE_LABEL = "Download an example CSV"
RESULTS_LABEL = "Download per-window results as CSV"
LAB_DOWNLOAD_LABEL = "Download this I/Q example as CSV"
SHARE_HINT = "To share this exact example, copy the address from your browser's address bar."
LINK_NOTE_START = "Part of this link could not be used as given:"
# The text summary that must sit directly under each chart, by chart key.
SUMMARY_STARTS = {
    "scores": "Scores: ",
    "time": f"The time plot shows the first 128 of {SAMPLES_PER_WINDOW:,} samples.",
    "constellation": f"The I/Q plane shows {SAMPLES_PER_WINDOW // 2} points",
    "fft": f"Spectrum of {SAMPLES_PER_WINDOW:,} frequency bins",
}
# Only the upload tab has whole-recording charts: a timeline of every window's
# decision and a spectrogram.
UPLOAD_ONLY_STARTS = {
    "timeline": "The timeline shows ",
    "spectrogram": f"The spectrogram shows {SEGMENT_LENGTH} frequency bins across",
}
UPLOAD_CHARTS = {*UPLOAD_ONLY_STARTS, *SUMMARY_STARTS}


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
        array = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        if "shape" in values:
            array = array.reshape([int(size) for size in values["shape"].split(",")])
        return array
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

    def run_app(self, *uploads, query_params: dict | None = None):
        """Run the app with uploads given as (file name, contents) pairs, opened
        at a URL with the given query parameters."""
        files = [named_upload(name, payload) for name, payload in uploads]
        app = AppTest.from_file(str(APP_PATH))
        app.query_params.update(query_params or {})
        with patch("streamlit.file_uploader", return_value=files):
            app = app.run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        return app

    def run_csv(self, payload: bytes):
        return self.run_app(("recording.csv", payload))

    def run_and_record_downloads(self, *uploads, query_params: dict | None = None):
        """Run the app and return it with the data of each download button, by label."""
        downloads = {}
        download_button = st.download_button

        def recording_download_button(label, data, *args, **kwargs):
            downloads[label] = data
            return download_button(label, data, *args, **kwargs)

        with patch("streamlit.download_button", new=recording_download_button):
            app = self.run_app(*uploads, query_params=query_params)
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
        starts = dict(SUMMARY_STARTS)
        if section == "uploaded":
            starts.update(UPLOAD_ONLY_STARTS)
        self.assertEqual(set(summaries), {f"{section}-{kind}" for kind in starts})
        for kind, start in starts.items():
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
        self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)
        # The spectrogram averages neighboring segments to stay light.
        levels = plotly_values(self.chart_spec(app, "uploaded-spectrogram")["data"][0]["z"])
        self.assertEqual(levels.shape, (390, SEGMENT_LENGTH))
        self.assertLessEqual(len(levels), MAX_TIME_STEPS)

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
        # The whole-file timeline marks the window as not classifiable instead.
        self.assertEqual(
            self.uploaded_charts(app), {"timeline", "spectrogram", "time", "constellation", "fft"}
        )
        summaries = self.chart_summaries(app.tabs[1])
        self.assertIn("every sample is zero", summaries["uploaded-constellation"])
        self.assertIn("every sample is zero", summaries["uploaded-fft"])
        self.assertIn(
            f"Every cell is at the {FLOOR_DB:g} dB floor", summaries["uploaded-spectrogram"]
        )

    def test_maximum_finite_iq_upload_has_no_numeric_warnings(self):
        maximum = repr(float(np.finfo(np.float64).max))
        payload = ("I,Q\n" + f"{maximum},{maximum}\n" * SAMPLES_PER_WINDOW).encode()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            app = self.run_csv(payload)
        numeric_warnings = [str(item.message) for item in recorded if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(numeric_warnings, [])
        self.assertEqual(len(app.error), 0)
        self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)

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
        self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)

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
                self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)
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
        self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)


class AccessibilityTests(AppTestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def test_title_is_plain_text_and_headings_are_in_order(self):
        app = self.run_csv(self.example_csv().encode())
        (title,) = app.title
        self.assertEqual(title.value, "SpectrumSense Signal Lab")
        self.assertEqual(
            [char for char in title.value if unicodedata.category(char) in ("So", "Mn", "Cf")], []
        )
        self.assertEqual(title.tag, "h1")
        # Each tab opens with a level-2 heading and only level-3 headings follow
        # it, so no level is skipped.
        self.assertEqual(
            [item.value for item in app.subheader],
            [
                "Every window",
                "Frequency content over time",
                "One window in detail",
                'When the model answers "uncertain"',
            ],
        )
        for tab in app.tabs:
            with self.subTest(tab=tab.label):
                first = ordered_children(tab)[0]
                self.assertEqual((first.type, first.tag), ("header", "h2"))
                self.assertEqual({item.tag for item in tab.get("subheader")} - {"h3"}, set())

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
        for label in (
            "Signal type", "Example seed", "Window to inspect", EXAMPLE_LABEL, RESULTS_LABEL,
        ):
            self.assertIn(label, labels)
        for widget in widgets:
            with self.subTest(widget=widget.proto.label):
                self.assertRegex(widget.proto.help, r"^[A-Z].*\.$")

    def test_spectrogram_uses_an_ordered_color_scale_and_matches_its_summary(self):
        samples = load_uploads([("example.csv", self.example_csv().encode())]).samples
        app = self.run_csv(to_csv(samples).encode())
        spec = self.chart_spec(app, "uploaded-spectrogram")
        (heatmap,) = spec["data"]
        self.assertEqual(heatmap["type"], "heatmap")
        # Cividis grows lighter with level, so levels do not depend on hue.
        self.assertEqual(
            [color.lower() for _, color in heatmap["colorscale"]],
            [color.lower() for color in plotly.colors.sequential.Cividis],
        )
        self.assertEqual((heatmap["zmin"], heatmap["zmax"]), (FLOOR_DB, 0))
        self.assertEqual(heatmap["colorbar"]["title"]["text"], "Level (dB, 0 = strongest)")
        self.assertIn("Level: %{z:.1f} dB", heatmap["hovertemplate"])
        self.assertEqual(spec["layout"]["yaxis"]["autorange"], "reversed")
        self.assertEqual(spec["layout"]["xaxis"]["title"]["text"], "Frequency (cycles per sample)")
        self.assertEqual(spec["layout"]["yaxis"]["title"]["text"], "Sample")

        expected = compute_spectrogram(samples)
        np.testing.assert_array_equal(
            plotly_values(heatmap["z"]), expected.levels.astype(np.float32)
        )
        np.testing.assert_array_equal(plotly_values(heatmap["x"]), expected.frequencies)
        np.testing.assert_array_equal(plotly_values(heatmap["y"]), expected.centers)
        caption = self.chart_summaries(app.tabs[1])["uploaded-spectrogram"]
        self.assertEqual(caption, spectrogram_summary(expected))
        self.assertNotIn("Hz", caption)

    def test_spectrogram_uses_hertz_and_seconds_with_a_sample_rate(self):
        capture, _, _ = generate_capture("QPSK", np.random.default_rng(12), windows=2)
        meta, data = sigmf_files(capture.ravel())
        app = self.run_app(("rec.sigmf-meta", meta), ("rec.sigmf-data", data))
        spec = self.chart_spec(app, "uploaded-spectrogram")
        (heatmap,) = spec["data"]
        self.assertEqual(
            spec["layout"]["xaxis"]["title"]["text"], "Frequency from the band center (Hz)"
        )
        self.assertEqual(spec["layout"]["yaxis"]["title"]["text"], "Time (s)")
        expected = compute_spectrogram(capture.ravel().astype(np.complex64))
        np.testing.assert_allclose(plotly_values(heatmap["x"]), expected.frequencies * 2.4e6)
        np.testing.assert_allclose(plotly_values(heatmap["y"]), expected.centers / 2.4e6)
        caption = self.chart_summaries(app.tabs[1])["uploaded-spectrogram"]
        self.assertRegex(caption, r"the center of the recorded band, about [\d.]+ MHz\)")

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


class ShareableUrlTests(AppTestCase):
    @staticmethod
    def controls(app) -> tuple:
        """The Explore tab's signal type, SNR, carrier offset, and seed."""
        return (
            app.selectbox(key="explore_signal").value,
            app.slider(key="explore_snr").value,
            app.slider(key="explore_offset").value,
            app.number_input(key="explore_seed").value,
        )

    @staticmethod
    def link_notes(block) -> list[str]:
        return [item.value for item in block.info if item.value.startswith(LINK_NOTE_START)]

    @staticmethod
    def expected_csv(signal: str, snr: float, offset: float, seed: int) -> str:
        recording, _, _ = generate_capture(
            signal, np.random.default_rng(seed), windows=1,
            snr_db_override=snr, freq_offset_override=offset,
        )
        return to_csv(recording[0])

    def test_a_link_sets_the_controls_without_changing_other_tabs(self):
        link = {"signal": "QPSK", "snr": "3", "offset": "0.012", "seed": "7"}
        example = self.expected_csv("QPSK", 3.0, 0.012, 7)
        app, downloads = self.run_and_record_downloads(
            ("recording.csv", example.encode()), query_params=link
        )
        self.assertEqual(self.controls(app), ("QPSK", 3.0, 0.012, 7))
        self.assertEqual(downloads[LAB_DOWNLOAD_LABEL], example)
        lab = app.tabs[0]
        self.assertIn(SHARE_HINT, [caption.value for caption in lab.caption])
        self.assertIn("Actual simulated type: **QPSK**", " ".join(c.value for c in lab.caption))
        self.assertEqual(self.link_notes(app), [])
        # Opening a link never rewrites it.
        self.assertEqual(app.query_params, {key: [value] for key, value in link.items()})
        # The upload tab works as before and the link does not touch it.
        self.assertEqual(len(app.error), 0)
        self.assertEqual(self.uploaded_charts(app), UPLOAD_CHARTS)
        self.assertIn("Showing window 1 of 1", " ".join(c.value for c in app.caption))

    def test_unusable_link_values_fall_back_with_a_short_note(self):
        link = {
            "signal": "8PSK", "snr": "[here](https://example.com)", "offset": "0.5",
            "seed": ["1", "2"], "utm_source": "mail",
        }
        app = self.run_app(query_params=link)
        self.assertEqual(self.controls(app), ("BPSK", 6.0, 0.03, 42))
        (note,) = self.link_notes(app)
        self.assertEqual(self.link_notes(app.tabs[0]), [note])
        bullets = [line for line in note.splitlines() if line.startswith("- ")]
        self.assertEqual(len(bullets), 4, note)
        for words in (
            "signal type was ignored", "SNR was ignored", "offset of 0.5", "seed was ignored",
        ):
            self.assertIn(words, note)
        self.assertNotIn("example.com", note)
        # The link stays as opened; reopening it shows the same example and note.
        self.assertEqual(
            app.query_params,
            {
                "signal": ["8PSK"], "snr": [link["snr"]], "offset": ["0.5"], "seed": ["1", "2"],
                "utm_source": ["mail"],
            },
        )

        # The note describes the link, so it goes once a control changes.
        app = app.number_input(key="explore_seed").set_value(8).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual(self.link_notes(app), [])
        self.assertEqual(
            app.query_params,
            {
                "signal": ["BPSK"], "snr": ["6"], "offset": ["0.03"], "seed": ["8"],
                "utm_source": ["mail"],
            },
        )

    def test_changing_a_control_updates_the_url_that_reproduces_the_example(self):
        app = self.run_app(query_params={"utm_source": "mail"})
        self.assertEqual(self.controls(app), ("BPSK", 6.0, 0.0, 42))
        self.assertEqual(app.query_params, {"utm_source": ["mail"]})

        app = app.slider(key="explore_offset").set_value(0.005).run(timeout=20)
        app = app.selectbox(key="explore_signal").set_value("noise").run(timeout=20)
        app = app.number_input(key="explore_seed").set_value(7).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        written = {"signal": ["noise"], "snr": ["6"], "offset": ["0.005"], "seed": ["7"]}
        self.assertEqual(app.query_params, {**written, "utm_source": ["mail"]})
        self.assertTrue(app.slider(key="explore_snr").disabled)

        # A rerun without changes leaves the URL alone. Later runs trust the
        # controls, so a URL from before the last change cannot reset them.
        app = app.run(timeout=20)
        self.assertEqual(app.query_params, {**written, "utm_source": ["mail"]})
        app.query_params.clear()
        app.query_params.update({"signal": ["QPSK"], "seed": ["1"]})
        app = app.run(timeout=20)
        self.assertEqual(self.controls(app), ("noise", 6.0, 0.005, 7))

        # Opening the written URL in a new session shows exactly the same example.
        reopened, downloads = self.run_and_record_downloads(query_params=written)
        self.assertEqual(self.controls(reopened), ("noise", 6.0, 0.005, 7))
        self.assertEqual(self.link_notes(reopened), [])
        self.assertEqual(downloads[LAB_DOWNLOAD_LABEL], self.expected_csv("noise", 6.0, 0.005, 7))


class UncertainAndWholeFileTests(AppTestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = build_demo_model()

    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    @staticmethod
    def known_windows() -> np.ndarray:
        """The two BPSK windows of the upload tab's example CSV."""
        capture, _, _ = generate_capture("BPSK", np.random.default_rng(2024), windows=2)
        return capture.ravel()

    @staticmethod
    def tone(windows: int) -> np.ndarray:
        return generate_probe_capture("tone", np.random.default_rng(5), windows, snr_db=10).ravel()

    def test_default_lab_signal_gets_a_class_and_an_explanation(self):
        app = self.run_app()
        (metric,) = app.tabs[0].metric
        self.assertEqual((metric.label, metric.value), ("Demo model prediction", "BPSK"))
        text = self.markdown_text(app)
        self.assertRegex(
            text, r"The model names BPSK: the window is \d+\.\d from the nearest trained class "
            r"\(limit \d+\.\d\) and the top score is \d\.\d\d \(minimum 0\.60\)\."
        )

    def test_unfamiliar_upload_is_uncertain_and_names_the_leading_candidate(self):
        app = self.run_csv(to_csv(self.tone(1)).encode())
        self.assertEqual(len(app.error), 0)
        (metric,) = app.tabs[1].metric
        self.assertEqual((metric.label, metric.value), ("Demo model prediction", "Uncertain"))
        self.assertRegex(
            self.markdown_text(app),
            r"Uncertain because the window is unlike the training data: it is \d+\.\d from "
            r"the nearest trained class, beyond the limit of \d+\.\d\. The leading candidate "
            r"is (noise|BPSK|QPSK) with a score of \d\.\d\d\.",
        )
        captions = " ".join(caption.value for caption in app.tabs[1].caption)
        for claim in (
            "simple distance-and-score heuristic",
            "tested only on synthetic probe signals",
            "not validated unknown-signal detection",
            "very-low-SNR signals can still be called noise",
        ):
            self.assertIn(claim, captions)

    def test_every_window_is_classified_counted_and_downloadable(self):
        samples = np.concatenate(
            [self.known_windows(), np.zeros(SAMPLES_PER_WINDOW), np.ones(10)]
        )
        expected = classify_recording(self.demo.model, samples)
        self.assertEqual(expected.labels, ("BPSK", "BPSK", NOT_CLASSIFIABLE))
        app, downloads = self.run_and_record_downloads(("mixed.csv", to_csv(samples).encode()))
        # Window 1, which is inspected below the whole-file results, is classifiable.
        self.assertEqual(len(app.error), 0)
        upload_tab = app.tabs[1]
        summary = self.chart_summaries(upload_tab)["uploaded-timeline"]
        self.assertEqual(
            summary,
            f"The timeline shows 3 windows: 2 BPSK, 1 {NOT_CLASSIFIABLE}. In order, "
            f"windows 1 to 2: BPSK; window 3: {NOT_CLASSIFIABLE}.",
        )
        (table,) = upload_tab.dataframe
        self.assertEqual(
            list(zip(table.value["Decision"], table.value["Windows"])), decision_counts(expected)
        )
        self.assertEqual(list(table.value["Share of windows"])[1], "66.7%")
        captions = " ".join(caption.value for caption in upload_tab.caption)
        self.assertIn(
            "The last 10 samples, after window 3, were ignored because they do not fill a "
            f"complete {SAMPLES_PER_WINDOW:,}-sample window.",
            captions,
        )
        self.assertIn("Showing window 1 of 3", captions)

        rows = list(csv.DictReader(StringIO(downloads[RESULTS_LABEL])))
        self.assertEqual([row["window"] for row in rows], ["1", "2", "3"])
        starts = [str(index * SAMPLES_PER_WINDOW) for index in range(3)]
        self.assertEqual([row["start_sample"] for row in rows], starts)
        self.assertEqual([row["decision"] for row in rows], list(expected.labels))
        self.assertIn("only zeros", rows[2]["reason"])
        self.assertEqual(rows[2]["score_BPSK"], "")
        for row in rows[:2]:
            total = sum(float(row[f"score_{name}"]) for name in ("noise", "BPSK", "QPSK"))
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_whole_recording_leaves_no_ignored_samples_note(self):
        app = self.run_csv(to_csv(self.known_windows()).encode())
        captions = " ".join(caption.value for caption in app.tabs[1].caption)
        self.assertIn("The timeline shows 2 windows: 2 BPSK.", captions)
        self.assertNotIn("were ignored", captions)

    def test_timeline_shows_decisions_by_position_and_marker_shape(self):
        samples = np.concatenate(
            [self.known_windows(), self.tone(2), np.zeros(SAMPLES_PER_WINDOW)]
        )
        app = self.run_csv(to_csv(samples).encode())
        spec = self.chart_spec(app, "uploaded-timeline")
        self.assertEqual(list(spec["layout"]["yaxis"]["categoryarray"]), list(decision_order()))
        traces = spec["data"]
        self.assertEqual([trace["name"] for trace in traces], ["BPSK", UNCERTAIN, NOT_CLASSIFIABLE])
        symbols = [trace["marker"]["symbol"] for trace in traces]
        self.assertEqual(len(set(symbols)), len(symbols))
        windows = {}
        for trace in traces:
            self.assertEqual(trace["mode"], "markers")
            self.assertEqual(set(trace["y"]), {trace["name"]})
            windows[trace["name"]] = plotly_values(trace["x"]).tolist()
        self.assertEqual(windows, {"BPSK": [1, 2], UNCERTAIN: [3, 4], NOT_CLASSIFIABLE: [5]})
        self.assertIn(
            f"windows 3 to 4: {UNCERTAIN}", self.chart_summaries(app.tabs[1])["uploaded-timeline"]
        )

    def test_about_tab_reports_limits_and_measured_rates(self):
        app = self.run_app()
        about = app.tabs[2]
        demo, model = self.demo, self.demo.model
        metrics = {metric.label: metric.value for metric in about.metric}
        self.assertEqual(metrics["Calibration captures"], str(demo.calibration_captures))
        self.assertEqual(metrics["Synthetic test accuracy"], f"{demo.test_accuracy:.1%}")
        text = self.markdown_text(app)
        self.assertIn(f"more than {model.distance_threshold:.2f} from every trained class", text)
        self.assertIn(f"the top score is below {model.min_top_score:.2f}", text)
        self.assertIn(
            f"On {demo.test_windows:,} held-out windows of noise, BPSK, and QPSK, "
            f"{demo.false_reject_rate:.1%} were answered uncertain; the other windows were "
            f"{demo.accepted_accuracy:.1%} accurate.",
            text,
        )
        (table,) = about.dataframe
        self.assertEqual(list(table.value.iloc[:, 0]), list(PROBE_KINDS))
        for snr, rates in demo.probe_flag_rates.items():
            with self.subTest(snr=snr):
                column = table.value[f"Answered uncertain at {snr:g} dB SNR"]
                self.assertEqual(list(column), [f"{rates[kind]:.0%}" for kind in PROBE_KINDS])
        captions = " ".join(caption.value for caption in about.caption)
        self.assertIn("Neither the probes nor the held-out captures were used", captions)
        self.assertIn("not validated unknown-signal detection", captions)


if __name__ == "__main__":
    unittest.main()
