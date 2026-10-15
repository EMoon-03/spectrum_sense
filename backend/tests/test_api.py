"""The HTTP boundary: every endpoint, its error codes, and the guarantees the
schemas make (reasons and limits travel with decisions, NaN never crosses,
the model fingerprint is everywhere, file text is never echoed).

The model trains once for the module. Tests that need a run create one from
the bundled recording, so no network and no fixture files are needed.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import SpooledTemporaryFile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.formparsers import MultiPartParser  # noqa: E402

import iq_csv  # noqa: E402
import real_example  # noqa: E402
from app import config  # noqa: E402
from app.main import app  # noqa: E402
from app.model import MODEL_VERSION  # noqa: E402
from app.runs import store  # noqa: E402
from signal_config import (  # noqa: E402
    CLASSES,
    FEATURE_NAMES,
    NOT_CLASSIFIABLE,
    SAMPLES_PER_WINDOW,
    UNCERTAIN,
)
from spectrum_sense import generate_capture  # noqa: E402


def _finite_everywhere(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_everywhere(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_everywhere(item) for item in value)
    return True


class ApiTestCase(unittest.TestCase):
    client: TestClient

    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        store.clear()


class HealthTests(ApiTestCase):
    def test_health_reports_a_trained_model_and_its_fingerprint(self):
        for path in ("/", "/health"):
            body = self.client.get(path).json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["model"], {"trained": True, "version": MODEL_VERSION})
        self.assertRegex(MODEL_VERSION, r"^[0-9a-f]{16}$")

    def test_openapi_document_lists_every_route(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        for path in (
            "/health", "/model", "/explore", "/explore/csv", "/analyze", "/analyze/example",
            "/runs/{run_id}", "/runs/{run_id}/prepare", "/runs/{run_id}/windows/{number}",
            "/runs/{run_id}/results.csv", "/example", "/example/file",
        ):
            self.assertIn(path, paths)

    def test_cors_allows_only_the_configured_origins(self):
        allowed = config.ALLOWED_ORIGINS[0]
        yes = self.client.get("/health", headers={"Origin": allowed})
        self.assertEqual(yes.headers.get("access-control-allow-origin"), allowed)
        no = self.client.get("/health", headers={"Origin": "https://evil.example"})
        self.assertIsNone(no.headers.get("access-control-allow-origin"))


class ModelTests(ApiTestCase):
    def test_report_matches_the_lab_and_has_no_nan(self):
        body = self.client.get("/model").json()
        self.assertEqual(body["version"], MODEL_VERSION)
        self.assertEqual(body["training"]["classes"], list(CLASSES))
        self.assertGreater(body["test_accuracy"], 0.95)
        self.assertLess(body["false_reject_rate"], 0.1)
        self.assertEqual(body["controls"]["snr"]["minimum"], -12)
        self.assertEqual(body["controls"]["default_signal"], "BPSK")
        self.assertEqual({item["snr_db"] for item in body["probe_flag_rates"]}, {5.0, 10.0})
        self.assertTrue(_finite_everywhere(body))
        self.assertEqual(body["limits"]["min_top_score"], 0.6)


class ExploreTests(ApiTestCase):
    def test_a_link_reproduces_the_lab_example_exactly(self):
        body = self.client.get("/explore", params={"signal": "QPSK", "snr": 3, "offset": 0.012, "seed": 7}).json()
        self.assertEqual(body["settings"], {"signal": "QPSK", "snr_db": 3.0, "offset": 0.012, "seed": 7})
        self.assertEqual(body["notes"], [])
        self.assertEqual(body["in_training_range"], {"snr": True, "offset": False})
        self.assertEqual(body["model_version"], MODEL_VERSION)
        windows, _, _ = generate_capture(
            "QPSK", np.random.default_rng(7), windows=1, snr_db_override=3.0, freq_offset_override=0.012
        )
        np.testing.assert_allclose(body["view"]["samples"]["i"], windows[0].real)
        np.testing.assert_allclose(body["view"]["samples"]["q"], windows[0].imag)
        decision = body["decision"]
        self.assertEqual(decision["classes"], list(CLASSES))
        self.assertEqual(len(decision["scores"]), 3)
        self.assertAlmostEqual(sum(decision["scores"]), 1.0, places=6)
        self.assertIn(decision["label"], (*CLASSES, UNCERTAIN))
        self.assertIn("limit", decision["summary"])
        self.assertTrue(decision["summary"].startswith(("The model names", "Uncertain because")))
        for key in ("time", "constellation", "spectrum"):
            self.assertTrue(body["view"]["summaries"][key])

    def test_every_decision_carries_its_evidence_and_possible_causes(self):
        named = self.client.get("/explore", params={"signal": "BPSK", "snr": 12, "seed": 1}).json()["decision"]
        refused = self.client.get("/explore", params={"signal": "BPSK", "snr": -2, "seed": 11}).json()["decision"]
        for decision in (named, refused):
            evidence = decision["evidence"]
            self.assertEqual([item["name"] for item in evidence["features"]], list(FEATURE_NAMES))
            self.assertAlmostEqual(sum(item["distance_share"] for item in evidence["features"]), 1.0, places=6)
            self.assertAlmostEqual(evidence["distance"], decision["distance"], places=9)
            self.assertIn(evidence["nearest_class"], CLASSES)
            self.assertEqual(evidence["leading"], decision["leading"])
            self.assertEqual(decision["causes"], [])
        self.assertEqual(named["label"], "BPSK")
        self.assertIn("does most to put BPSK ahead of", named["evidence"]["summary"])
        self.assertEqual((refused["label"], refused["reason"]), (UNCERTAIN, "unlike the training data"))
        self.assertIn("changing that alone would take the distance", refused["evidence"]["summary"])

    def test_a_run_window_names_its_possible_causes(self):
        run = self.client.post("/analyze/example").json()
        number = next(w["number"] for w in run["windows"] if w["causes"] == ["power-step"])
        decision = self.client.get(f"/runs/{run['run_id']}/windows/{number}").json()["decision"]
        (cause,) = decision["causes"]
        self.assertEqual(cause["key"], "power-step")
        self.assertIn("dB above the quietest", cause["detail"])

    def test_unusable_values_fall_back_with_notes_and_never_echo_the_link(self):
        body = self.client.get(
            "/explore?signal=8PSK&snr=%5Bhere%5D(https://example.com)&offset=0.5&seed=1&seed=2&utm_source=mail"
        ).json()
        self.assertEqual(body["settings"], {"signal": "BPSK", "snr_db": 6.0, "offset": 0.03, "seed": 42})
        self.assertEqual(len(body["notes"]), 4)
        self.assertNotIn("example.com", json.dumps(body))

    def test_noise_has_no_snr_range(self):
        body = self.client.get("/explore", params={"signal": "noise"}).json()
        self.assertIsNone(body["in_training_range"]["snr"])

    def test_csv_download_is_the_same_window_and_round_trips(self):
        response = self.client.get("/explore/csv", params={"signal": "BPSK", "seed": 3})
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/csv")
        self.assertIn("seed-3", response.headers["content-disposition"])
        samples = iq_csv.from_csv(response.content)
        shown = self.client.get("/explore", params={"signal": "BPSK", "seed": 3}).json()["view"]["samples"]
        np.testing.assert_array_equal(samples.real, shown["i"])
        np.testing.assert_array_equal(samples.imag, shown["q"])


class AnalyzeTests(ApiTestCase):
    def analyze_example(self, **settings):
        response = self.client.post("/analyze/example", json=settings)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_the_bundled_recording_gives_the_documented_answers(self):
        body = self.analyze_example()
        self.assertEqual(body["source"], "example")
        self.assertEqual(body["example"]["license"], real_example.LICENSE)
        self.assertEqual(body["recording"]["sample_count"], real_example.SAMPLE_COUNT)
        self.assertEqual(body["recording"]["sample_rate"], real_example.SAMPLE_RATE)
        self.assertEqual(body["preparation"]["description"], "none")
        counts = {item["label"]: item["count"] for item in body["counts"]}
        self.assertEqual(counts, {"noise": 0, "BPSK": 0, "QPSK": 0, UNCERTAIN: 30})
        self.assertEqual(len(body["windows"]), 30)
        # The 20 windows that straddle a packet edge change power; the 10 inside
        # a packet show no measured cause, and their evidence says where they lie.
        self.assertEqual(
            {item["key"]: item["count"] for item in body["cause_counts"]},
            {"power-step": 20, "bare-carrier": 0, "center-component": 0},
        )
        self.assertEqual(
            body["causes_summary"],
            "Of the 30 uncertain windows, 20 change power within the window; "
            "10 show no measured cause.",
        )
        self.assertEqual(sum(window["causes"] == ["power-step"] for window in body["windows"]), 20)
        self.assertEqual(body["windows"][0], {**body["windows"][0], "number": 1, "start_sample": 0})
        self.assertEqual(body["model_version"], MODEL_VERSION)
        self.assertIn("The timeline shows 30 windows", body["timeline_summary"])
        self.assertEqual(body["spectrogram"]["time_axis_title"], "Time (s)")
        self.assertEqual(len(body["spectrogram"]["levels_db"]), len(body["spectrogram"]["times"]))
        self.assertEqual(len(body["spectrogram"]["levels_db"][0]), len(body["spectrogram"]["frequencies"]))
        self.assertEqual(len(body["loudest"]), 5)
        self.assertTrue(_finite_everywhere(body))
        self.assertGreater(body["expires_in_seconds"], 0)

    def test_resampling_through_the_run_keeps_the_id_and_matches_the_readme(self):
        first = self.analyze_example()
        response = self.client.post(f"/runs/{first['run_id']}/prepare", json={"symbol_rate": 125})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["run_id"], first["run_id"])
        self.assertEqual(len(body["windows"]), 60)
        self.assertTrue(all(item["label"] == UNCERTAIN for item in body["windows"]))
        self.assertEqual(body["preparation"]["samples_per_symbol"], 8)
        self.assertEqual(body["preparation"]["ratio"], [2, 1])
        self.assertTrue(body["preparation"]["band_note"])
        # The same run reads back with the new preparation.
        again = self.client.get(f"/runs/{first['run_id']}").json()
        self.assertEqual(len(again["windows"]), 60)

    def test_bad_settings_are_422_with_the_prepare_message(self):
        run = self.analyze_example()
        response = self.client.post(f"/runs/{run['run_id']}/prepare", json={"symbol_rate": 125, "factor": 2})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Give a symbol rate or a resampling factor, not both.")

    def test_a_window_view_has_the_decision_and_plots(self):
        run = self.analyze_example()
        response = self.client.get(f"/runs/{run['run_id']}/windows/5")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["number"], 5)
        self.assertEqual(body["start_sample"], 4 * SAMPLES_PER_WINDOW)
        self.assertEqual(len(body["view"]["samples"]["i"]), SAMPLES_PER_WINDOW)
        self.assertEqual(body["decision"]["label"], run["windows"][4]["label"])
        self.assertIn("median window", body["level_phrase"])
        self.assertEqual(self.client.get(f"/runs/{run['run_id']}/windows/31").status_code, 404)
        self.assertEqual(self.client.get(f"/runs/{run['run_id']}/windows/0").status_code, 404)

    def test_csv_carries_the_preparation_and_the_model_version(self):
        run = self.analyze_example(symbol_rate=125)
        response = self.client.get(f"/runs/{run['run_id']}/results.csv")
        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(response.text)))
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows[0]["model_version"], MODEL_VERSION)
        self.assertTrue(rows[0]["preparation"].startswith("Resampled by 2/1"))
        self.assertEqual(rows[0]["decision"], UNCERTAIN)

    def test_a_forgotten_run_is_404(self):
        response = self.client.get("/runs/not-a-run")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Analyze the recording again", response.json()["detail"])

    def test_an_uploaded_csv_from_explore_is_classified_as_in_the_lab(self):
        data = self.client.get("/explore/csv", params={"signal": "BPSK", "snr": 10}).content
        response = self.client.post("/analyze", files=[("files", ("one.csv", data, "text/csv"))])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["source"], "upload")
        self.assertIsNone(body["example"])
        self.assertEqual(body["recording"]["source_format"], "CSV text (I,Q columns)")
        self.assertEqual(len(body["windows"]), 1)
        self.assertEqual(body["windows"][0]["label"], "BPSK")

    def test_later_uploads_win_and_earlier_ones_are_counted(self):
        first = self.client.get("/explore/csv", params={"signal": "noise"}).content
        second = self.client.get("/explore/csv", params={"signal": "QPSK", "snr": 12}).content
        response = self.client.post(
            "/analyze",
            files=[("files", ("a.csv", first, "text/csv")), ("files", ("b.csv", second, "text/csv"))],
        )
        body = response.json()
        self.assertEqual(body["recording"]["ignored_uploads"], 1)
        self.assertEqual(body["windows"][0]["label"], "QPSK")

    def test_an_all_zero_window_is_not_classifiable_without_nan(self):
        zeros = iq_csv.to_csv(np.zeros(SAMPLES_PER_WINDOW, dtype=complex)).encode()
        response = self.client.post("/analyze", files=[("files", ("z.csv", zeros, "text/csv"))])
        self.assertEqual(response.status_code, 200, response.text)
        window = response.json()["windows"][0]
        self.assertEqual(window["label"], NOT_CLASSIFIABLE)
        self.assertEqual(window["scores"], [None, None, None])
        self.assertIsNone(window["distance"])
        run_id = response.json()["run_id"]
        view = self.client.get(f"/runs/{run_id}/windows/1").json()
        self.assertEqual(view["decision"]["label"], NOT_CLASSIFIABLE)
        self.assertTrue(_finite_everywhere(view))

    def test_bad_files_are_422_and_never_echo_their_contents(self):
        secret = "SECRET-TOKEN-123"
        cases = [
            ("x.exe", b"anything"),
            ("short.csv", f"I,Q\n1,{secret}\n".encode()),
            ("bad.npy", b"\x93NUMPY" + secret.encode()),
        ]
        for name, data in cases:
            with self.subTest(name=name):
                response = self.client.post("/analyze", files=[("files", (name, data, "application/octet-stream"))])
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(secret, response.text)
                self.assertTrue(response.json()["detail"])

    def test_oversized_uploads_are_413(self):
        data = b"0" * (config.MAX_UPLOAD_BYTES + 1)
        response = self.client.post("/analyze", files=[("files", ("big.cf32", data, "application/octet-stream"))])
        self.assertEqual(response.status_code, 413)

    def test_a_large_content_length_is_refused_before_the_body_is_read(self):
        data = b"0" * (2 * config.MAX_UPLOAD_BYTES)
        with patch.object(MultiPartParser, "parse", side_effect=AssertionError("body was read")):
            response = self.client.post(
                "/analyze", files=[("files", ("big.cf32", data, "application/octet-stream"))]
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn(f"{config.MAX_UPLOAD_BYTES:,} bytes", response.json()["detail"])

    def test_an_upload_without_a_content_length_is_411(self):
        response = self.client.post(
            "/analyze",
            content=iter([b"--x\r\n"]),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
        self.assertEqual(response.status_code, 411)

    def test_uploads_under_the_limit_are_never_written_to_disk(self):
        # Starlette moves a file part over its spool size to a temporary file.
        samples = generate_capture("QPSK", np.random.default_rng(2), 40)[0].ravel()
        data = iq_csv.to_csv(samples).encode()
        self.assertGreater(len(data), 1024 * 1024)
        self.assertLessEqual(len(data), config.MAX_UPLOAD_BYTES)
        with patch.object(SpooledTemporaryFile, "rollover", side_effect=AssertionError("on disk")):
            response = self.client.post(
                "/analyze", files=[("files", ("big.csv", data, "text/csv"))]
            )
        self.assertEqual(response.status_code, 200, response.text)

    def test_run_cache_evicts_the_oldest(self):
        ids = [self.analyze_example()["run_id"] for _ in range(config.RUN_CACHE_SIZE + 1)]
        self.assertEqual(self.client.get(f"/runs/{ids[0]}").status_code, 404)
        self.assertEqual(self.client.get(f"/runs/{ids[-1]}").status_code, 200)


class ExampleTests(ApiTestCase):
    def test_example_credit_and_file(self):
        info = self.client.get("/example").json()
        self.assertEqual(info["author"], real_example.AUTHOR)
        self.assertIn("CC BY 4.0", info["credit_markdown"])
        response = self.client.get("/example/file")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, real_example.EXAMPLE_PATH.read_bytes())
        self.assertIn(real_example.EXAMPLE_PATH.name, response.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
