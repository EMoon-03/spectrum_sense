"""Offline tests for tools/fetch_real_iq.py. Every network request is faked."""

from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
import hashlib
import importlib.util
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse

import numpy as np

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, to_csv
from iq_formats import load_recording, load_sigmf_pair
from signal_config import SAMPLES_PER_WINDOW


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fetch_real_iq.py"


def load_tool():
    """Import the script by path: tools/ is a folder of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("fetch_real_iq", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()

RECORD = "13371136"
RECORD_URL = f"https://zenodo.org/api/records/{RECORD}"
STEM = "gr4-packet-modem-intelsat37e-test"
TITLE = "Test of gr4-packet-modem through an Intelsat 37e C-band transponder"
AUTHOR = "Daniel Estévez <daniel@destevez.net>"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
SAMPLE_RATE = 500
FREQUENCY = 3_671_055_830
W = SAMPLES_PER_WINDOW
TOTAL = 12 * W + 100  # twelve whole windows and a partial one
LOUD_WINDOWS = (3, 9)  # numbered from 1, like the app
QUIET_WINDOWS = (4, 7)
# A link the API answer carries; the tool must build its own download addresses.
TRAP_LINK = "file:///etc/passwd"


def file_url(key: str) -> str:
    return f"{RECORD_URL}/files/{urllib.parse.quote(key, safe='')}/content"


def bursty_samples(count: int = TOTAL, seed: int = 5) -> np.ndarray:
    """Complex64 noise with two loud and two quiet windows."""
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(count) + 1j * rng.standard_normal(count)).astype(np.complex64)
    for window, gain in [(n, 10.0) for n in LOUD_WINDOWS] + [(n, 0.1) for n in QUIET_WINDOWS]:
        samples[(window - 1) * W : window * W] *= gain
    return samples


def sigmf_meta(data: bytes, datatype: str = "cf32_le", captures=None, **fields) -> dict:
    section = {
        "core:author": AUTHOR,
        "core:datatype": datatype,
        "core:description": TITLE,
        "core:hw": "USRP B200mini connected to the L-band IF of a teleport",
        "core:license": CC_BY,
        "core:recorder": "GNU Radio 3.10",
        "core:sample_rate": SAMPLE_RATE,
        "core:sha512": hashlib.sha512(data).hexdigest(),
        "core:version": "1.0.0",
    }
    section.update(fields)
    section = {key: value for key, value in section.items() if value is not None}
    if captures is None:
        captures = [
            {"core:datetime": "2024-08-14T20:46:08Z", "core:frequency": FREQUENCY,
             "core:sample_start": 0}
        ]
    return {"global": section, "captures": captures, "annotations": []}


def zenodo_record(files: dict[str, bytes], sizes=None, md5s=None, **metadata) -> dict:
    """A record like the Zenodo API's answer, trimmed to the fields the tool reads."""
    sizes, md5s = sizes or {}, md5s or {}
    info = {
        "title": TITLE,
        "creators": [{"name": "Estévez, Daniel", "affiliation": None}],
        "license": {"id": "cc-by-4.0"},
    }
    info.update(metadata)
    return {
        "id": int(RECORD),
        "doi_url": f"https://doi.org/10.5281/zenodo.{RECORD}",
        "metadata": info,
        "files": [
            {
                "key": key,
                "size": sizes.get(key, len(payload)),
                "checksum": "md5:" + md5s.get(key, hashlib.md5(payload).hexdigest()),
                "links": {"self": TRAP_LINK},
            }
            for key, payload in files.items()
        ],
    }


def http_error(code: int, reason: str, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://zenodo.org/", code, reason, headers, None)


class FakeZenodo:
    """Stands in for urllib.request.urlopen. Each URL answers with bytes or an
    exception; a list gives one answer per request and then repeats the last."""

    def __init__(self, routes: dict):
        self.routes = {url: list(a) if isinstance(a, list) else [a] for url, a in routes.items()}
        self.requests = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.requests.append((url, request.get_header("User-agent"), timeout))
        answers = self.routes.get(url)
        if not answers:
            raise http_error(404, "Not Found")
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return BytesIO(answer)

    @property
    def urls(self) -> list[str]:
        return [url for url, _, _ in self.requests]


def recording_record(data: bytes, meta: dict | None = None, stem: str = STEM, **changes) -> dict:
    """A record holding one SigMF recording; changes go to zenodo_record."""
    meta_bytes = json.dumps(meta or sigmf_meta(data)).encode()
    files = {stem + ".sigmf-meta": meta_bytes, stem + ".sigmf-data": data}
    return zenodo_record(files, **changes)


def recording_routes(
    data: bytes, meta: dict | None = None, stem: str = STEM, record: dict | None = None
) -> dict:
    meta_bytes = json.dumps(meta or sigmf_meta(data)).encode()
    return {
        RECORD_URL: json.dumps(record or recording_record(data, meta, stem)).encode(),
        file_url(stem + ".sigmf-meta"): meta_bytes,
        file_url(stem + ".sigmf-data"): data,
    }


def archive_contents(path: Path) -> tuple[list[str], dict]:
    """Member names and parsed metadata of a .sigmf archive."""
    with tarfile.open(path, mode="r:") as archive:
        names = archive.getnames()
        member = next(m for m in archive.getmembers() if m.name.endswith(".sigmf-meta"))
        return names, json.load(archive.extractfile(member))


def listed_windows(output: str, label: str) -> set[int]:
    line = next(line for line in output.splitlines() if line.strip().startswith(label))
    return {int(number) for number in line.rpartition(": ")[2].split(", ")}


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.out = self.root / "out"

    def run_tool(self, routes, *args: str, out: Path | None = None):
        """Run the command line offline; return (exit code, stdout, stderr, fake, sleep mock)."""
        fake = routes if isinstance(routes, FakeZenodo) else FakeZenodo(routes)
        stdout, stderr = StringIO(), StringIO()
        argv = ["--out-dir", str(out or self.out), *args]
        with patch("urllib.request.urlopen", fake), patch("time.sleep") as sleep:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = tool.main(argv)
        return code, stdout.getvalue(), stderr.getvalue(), fake, sleep

    def assert_fails(self, routes, *args: str, message: str, out: Path | None = None):
        """Check for a one-line error, no traceback, and no files written."""
        out = out or self.out
        code, _, stderr, fake, sleep = self.run_tool(routes, *args, out=out)
        self.assertEqual(code, 1)
        self.assertTrue(stderr.startswith("error: "), stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertIn(message, stderr)
        self.assertFalse(out.exists() and any(out.iterdir()), "no files on failure")
        return stderr, fake, sleep


class HappyPathTests(ToolTestCase):
    def test_writes_files_that_the_app_loads_with_their_metadata(self):
        samples = bursty_samples()
        code, stdout, stderr, _, _ = self.run_tool(recording_routes(samples.tobytes()))
        self.assertEqual(code, 0, stderr)

        name = f"{STEM}_0-{TOTAL}"
        expected_files = {f"{name}.sigmf", f"{name}.cf32", f"{name}.csv", f"{name}.ATTRIBUTION.txt"}
        self.assertEqual({path.name for path in self.out.iterdir()}, expected_files)
        expected = samples.astype(np.complex128)
        for extension in ("sigmf", "cf32", "csv"):
            with self.subTest(extension=extension):
                path = self.out / f"{name}.{extension}"
                recording = load_recording(path.name, path.read_bytes())
                np.testing.assert_array_equal(recording.samples, expected)
        archive = load_recording(f"{name}.sigmf", (self.out / f"{name}.sigmf").read_bytes())
        self.assertEqual(archive.sample_rate, SAMPLE_RATE)
        self.assertEqual(archive.center_frequency, FREQUENCY)
        self.assertEqual(archive.source_format, "SigMF archive (cf32_le)")
        self.assertEqual((self.out / f"{name}.csv").read_bytes().decode(), to_csv(expected))

        names, meta = archive_contents(self.out / f"{name}.sigmf")
        self.assertEqual(names, [name, f"{name}/{name}.sigmf-meta", f"{name}/{name}.sigmf-data"])
        section = meta["global"]
        self.assertEqual(section["core:datatype"], "cf32_le")
        self.assertEqual(section["core:sample_rate"], SAMPLE_RATE)
        self.assertEqual(section["core:author"], AUTHOR)
        self.assertEqual(section["core:license"], CC_BY)
        self.assertEqual(section["core:sha512"], hashlib.sha512(samples.tobytes()).hexdigest())
        self.assertIn(f'Excerpt of "{TITLE}"', section["core:description"])
        self.assertIn(f"samples 0 to {TOTAL - 1:,}", section["core:description"])
        self.assertIn(f"of {TOTAL:,} samples", section["core:description"])
        self.assertEqual(
            meta["captures"],
            [{"core:sample_start": 0, "core:frequency": FREQUENCY,
              "core:datetime": "2024-08-14T20:46:08Z"}],
        )

        attribution = (self.out / f"{name}.ATTRIBUTION.txt").read_text(encoding="utf-8")
        for fragment in (TITLE, "Estévez, Daniel", f"https://doi.org/10.5281/zenodo.{RECORD}",
                         "cc-by-4.0", CC_BY, f"kept samples 0 to {TOTAL - 1}", f"{name}.sigmf"):
            self.assertIn(fragment, attribution)

        self.assertIn("MD5 matches", stdout)
        self.assertIn("SHA-512 matches", stdout)
        loud, quiet = listed_windows(stdout, "Loudest"), listed_windows(stdout, "Quietest")
        self.assertLessEqual(set(LOUD_WINDOWS), loud)
        self.assertLessEqual(set(QUIET_WINDOWS), quiet)
        self.assertTrue(loud.isdisjoint(QUIET_WINDOWS) and quiet.isdisjoint(LOUD_WINDOWS))

    def test_downloads_only_from_addresses_it_builds(self):
        _, _, _, fake, _ = self.run_tool(recording_routes(bursty_samples().tobytes()))
        self.assertEqual(
            fake.urls,
            [RECORD_URL, file_url(STEM + ".sigmf-meta"), file_url(STEM + ".sigmf-data")],
        )
        for _, user_agent, timeout in fake.requests:
            self.assertEqual(user_agent, tool.HEADERS["User-Agent"])
            self.assertEqual(timeout, tool.TIMEOUT_S)

    def test_start_and_samples_choose_the_excerpt(self):
        samples = bursty_samples()
        code, _, stderr, _, _ = self.run_tool(
            recording_routes(samples.tobytes()), "--start", "2048", "--samples", "1024"
        )
        self.assertEqual(code, 0, stderr)
        path = self.out / f"{STEM}_2048-3072.sigmf"
        recording = load_recording(path.name, path.read_bytes())
        np.testing.assert_array_equal(recording.samples, samples[2048:3072].astype(np.complex128))
        _, meta = archive_contents(path)
        self.assertIn("samples 2,048 to 3,071", meta["global"]["core:description"])
        # 2,048 samples at 500 samples/s is 4.096 s after the recording's start.
        self.assertEqual(meta["captures"][0]["core:datetime"], "2024-08-14T20:46:12.096Z")

    def test_an_excerpt_past_the_end_is_shortened(self):
        samples = bursty_samples()
        start = TOTAL - 1500
        code, _, stderr, _, _ = self.run_tool(
            recording_routes(samples.tobytes()), "--start", str(start), "--samples", "5000"
        )
        self.assertEqual(code, 0, stderr)
        path = self.out / f"{STEM}_{start}-{TOTAL}.cf32"
        np.testing.assert_array_equal(np.fromfile(path, "<c8"), samples[start:])

    def test_start_must_leave_one_whole_window(self):
        routes = recording_routes(bursty_samples().tobytes())
        self.assert_fails(
            routes, "--start", str(TOTAL - W + 1), message=f"at most {TOTAL - W:,}"
        )

    def test_integer_datatypes_are_scaled_like_the_app(self):
        rng = np.random.default_rng(8)
        for datatype, dtype in (("ci16_le", "<i2"), ("ci8", "i1"), ("cu8", "u1")):
            with self.subTest(datatype=datatype):
                info = np.iinfo(dtype)
                data = rng.integers(info.min, info.max, 2 * TOTAL, endpoint=True).astype(dtype)
                data = data.tobytes()
                meta = sigmf_meta(data, datatype=datatype)
                out = self.root / datatype
                code, _, stderr, _, _ = self.run_tool(
                    recording_routes(data, meta), out=out
                )
                self.assertEqual(code, 0, stderr)
                app_view = load_sigmf_pair(json.dumps(meta).encode(), data).samples
                saved = np.fromfile(out / f"{STEM}_0-{TOTAL}.cf32", "<c8")
                np.testing.assert_allclose(saved, app_view, rtol=1e-6, atol=1e-7)
                _, written = archive_contents(out / f"{STEM}_0-{TOTAL}.sigmf")
                self.assertIn(f"converted from {datatype}", written["global"]["core:description"])

    def test_license_falls_back_to_the_zenodo_record(self):
        data = bursty_samples().tobytes()
        meta = sigmf_meta(data, **{"core:license": None, "core:author": None})
        record = recording_record(data, meta, license={"id": "cc-by-sa-4.0"})
        code, _, stderr, _, _ = self.run_tool(recording_routes(data, meta, record=record))
        self.assertEqual(code, 0, stderr)
        _, written = archive_contents(self.out / f"{STEM}_0-{TOTAL}.sigmf")
        self.assertEqual(
            written["global"]["core:license"], "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        self.assertEqual(written["global"]["core:author"], "Estévez, Daniel")

    def test_differing_licenses_are_pointed_out(self):
        data = bursty_samples().tobytes()
        meta = sigmf_meta(data, **{"core:license": "https://creativecommons.org/licenses/by-nc/4.0/"})
        _, stdout, _, _, _ = self.run_tool(recording_routes(data, meta))
        self.assertIn("the two license fields differ", stdout)

    def test_uses_the_capture_segment_in_effect_at_the_start(self):
        data = bursty_samples().tobytes()
        captures = [
            {"core:sample_start": 0, "core:frequency": 1e9, "core:datetime": "2024-01-01T00:00:00Z"},
            {"core:sample_start": 3000, "core:frequency": 2e9,
             "core:datetime": "2024-01-01T00:10:00Z"},
        ]
        meta = sigmf_meta(data, captures=captures)
        code, _, stderr, _, _ = self.run_tool(recording_routes(data, meta), "--start", "4096",
                                              "--samples", "1024")
        self.assertEqual(code, 0, stderr)
        _, written = archive_contents(self.out / f"{STEM}_4096-5120.sigmf")
        # (4,096 - 3,000) samples at 500 samples/s is 2.192 s into the second segment.
        self.assertEqual(
            written["captures"],
            [{"core:sample_start": 0, "core:frequency": 2_000_000_000,
              "core:datetime": "2024-01-01T00:10:02.192Z"}],
        )
        _, stdout, _, _, _ = self.run_tool(
            recording_routes(data, meta), "--samples", "4096", out=self.root / "spanning"
        )
        self.assertIn("a new SigMF capture segment starts at sample 3,000", stdout)

    def test_a_record_with_several_recordings(self):
        data = bursty_samples().tobytes()
        meta = json.dumps(sigmf_meta(data)).encode()
        files = {f"{stem}.sigmf-{kind}": payload for stem in ("a", "b")
                 for kind, payload in (("meta", meta), ("data", data))}
        routes = {RECORD_URL: json.dumps(zenodo_record(files)).encode()}
        routes.update({file_url(key): payload for key, payload in files.items()})

        _, stdout, _, _, _ = self.run_tool(routes)
        self.assertIn("Record holds 2 recordings; using 'a'", stdout)
        self.assertTrue((self.out / f"a_0-{TOTAL}.sigmf").exists())
        code, _, _, _, _ = self.run_tool(routes, "--recording", "b")
        self.assertEqual(code, 0)
        self.assertTrue((self.out / f"b_0-{TOTAL}.sigmf").exists())
        self.assert_fails(
            routes, "--recording", "c", message="No recording named 'c'. Choices: a, b",
            out=self.root / "unused",
        )


class SafetyTests(ToolTestCase):
    def test_file_names_from_the_api_stay_inside_the_output_folder(self):
        samples = bursty_samples()
        hostile_stems = (
            "../../escaped", "/tmp/absolute", "..\\..\\windows", "C:\\evil", ".hidden",
            "-rf", "a/b/../../../c", "\x1b[31mred\x07", "", "..", "名前",
        )
        for index, stem in enumerate(hostile_stems):
            with self.subTest(stem=stem):
                out = self.root / f"case{index}" / "out"
                code, stdout, stderr, fake, _ = self.run_tool(
                    recording_routes(samples.tobytes(), stem=stem), out=out
                )
                self.assertEqual(code, 0, stderr)
                written = [path for path in out.parent.rglob("*")]
                self.assertEqual(len(written), 5)  # out/ and its four files
                for path in written:
                    self.assertIn(path.parent, (out.parent, out))
                    if path.parent == out:
                        self.assertRegex(
                            path.name,
                            r"^[A-Za-z0-9][A-Za-z0-9_-]*_0-\d+\.(sigmf|cf32|csv|ATTRIBUTION\.txt)$",
                        )
                archive = next(out.glob("*.sigmf"))
                names, _ = archive_contents(archive)
                self.assertTrue(all(".." not in name and not name.startswith("/") for name in names))
                load_recording(archive.name, archive.read_bytes())
                for url in fake.urls:
                    self.assertTrue(url.startswith(RECORD_URL))
                    self.assertNotIn("/../", url)
                self.assertNotIn("\x1b", stdout)
                self.assertNotIn("\x07", stdout)
                cases = {path.name for path in self.root.iterdir()}
                self.assertEqual(cases, {f"case{number}" for number in range(index + 1)})

    def test_safe_stem(self):
        cases = {
            "gr4-packet-modem-intelsat37e-test": "gr4-packet-modem-intelsat37e-test",
            "../../etc/passwd": "etc_passwd",
            "..": "recording",
            "": "recording",
            "-rf": "rf",
            "a b.c": "a_b_c",
            "x" * 200: "x" * 80,
        }
        for stem, expected in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(tool.safe_stem(stem), expected)

    def test_checksum_mismatches_are_refused(self):
        data = bursty_samples().tobytes()
        record = recording_record(data, md5s={STEM + ".sigmf-data": "0" * 32})
        self.assert_fails(
            recording_routes(data, record=record), message="does not match Zenodo's MD5 checksum"
        )
        meta = sigmf_meta(data, **{"core:sha512": "0" * 128})
        self.assert_fails(recording_routes(data, meta), message="does not match core:sha512")

    def test_a_download_over_the_cap_is_refused_before_it_starts(self):
        data = bursty_samples().tobytes()
        record = recording_record(data, sizes={STEM + ".sigmf-data": 60_000_000})
        _, fake, _ = self.assert_fails(
            recording_routes(data, record=record), message="60.0 MB, over the 50 MB cap"
        )
        self.assertNotIn(file_url(STEM + ".sigmf-data"), fake.urls)
        _, fake, _ = self.assert_fails(
            recording_routes(data), "--max-mb", "0.05", message="over the 0.05 MB cap"
        )
        self.assertNotIn(file_url(STEM + ".sigmf-data"), fake.urls)

    def test_a_download_larger_than_stated_stops_at_the_cap(self):
        data = bursty_samples().tobytes()
        record = recording_record(data, sizes={STEM + ".sigmf-data": 1000})
        self.assert_fails(
            recording_routes(data, record=record), "--max-mb", "0.02",
            message="stopped at the 20,000-byte cap",
        )

    def test_a_short_download_is_retried(self):
        data = bursty_samples().tobytes()
        routes = recording_routes(data)
        routes[file_url(STEM + ".sigmf-data")] = [data[:1000], data]
        code, _, stderr, fake, sleep = self.run_tool(routes)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(fake.urls.count(file_url(STEM + ".sigmf-data")), 2)
        sleep.assert_called_once_with(2)

    def test_unsupported_datatypes_are_refused_before_the_download(self):
        data = bursty_samples().tobytes()
        cases = {
            "cf64_le": "The SigMF datatype cf64_le is not supported",
            "rf32_le": "The SigMF datatype rf32_le is not supported",
            "[x](https://e.example)": "The SigMF datatype of this recording is not supported",
            None: "does not state core:datatype",
        }
        for datatype, message in cases.items():
            with self.subTest(datatype=datatype):
                meta = sigmf_meta(data, **{"core:datatype": datatype})
                stderr, fake, _ = self.assert_fails(recording_routes(data, meta), message=message)
                self.assertNotIn("e.example", stderr)
                self.assertNotIn(file_url(STEM + ".sigmf-data"), fake.urls)

    def test_multichannel_and_non_finite_recordings_are_refused(self):
        data = bursty_samples().tobytes()
        meta = sigmf_meta(data, **{"core:num_channels": 2})
        self.assert_fails(recording_routes(data, meta), message="several channels")
        samples = bursty_samples()
        samples[10] = np.nan
        self.assert_fails(recording_routes(samples.tobytes()), message="NaN or infinite")

    def test_a_record_without_a_sigmf_pair(self):
        routes = {
            RECORD_URL: json.dumps(zenodo_record({"capture.zip": b"PK"})).encode(),
        }
        self.assert_fails(routes, message="no uncompressed .sigmf-meta/.sigmf-data pair")
        routes = {RECORD_URL: json.dumps({"metadata": {"title": "Closed"}}).encode()}
        self.assert_fails(routes, message="no public file list")


class NetworkErrorTests(ToolTestCase):
    def test_http_errors_give_clear_messages_without_needless_retries(self):
        cases = (
            (404, "Not Found",
             "Zenodo record 13371136: HTTP 404 Not Found (not found: check the record number)"),
            (410, "Gone", "HTTP 410 Gone (the record was removed)"),
            (403, "Forbidden", "HTTP 403 Forbidden"),
        )
        for code, reason, message in cases:
            with self.subTest(code=code):
                _, fake, sleep = self.assert_fails(
                    {RECORD_URL: http_error(code, reason)}, message=message
                )
                self.assertEqual(fake.urls, [RECORD_URL])
                sleep.assert_not_called()

    def test_server_errors_are_retried_then_reported(self):
        routes = {RECORD_URL: http_error(503, "Service Unavailable")}
        _, fake, sleep = self.assert_fails(
            routes, message="HTTP 503 Service Unavailable (server error: try again later"
        )
        self.assertEqual(len(fake.urls), tool.ATTEMPTS)
        self.assertEqual([c.args for c in sleep.call_args_list], [(2,), (4,)])

    def test_too_many_requests_waits_as_asked(self):
        data = bursty_samples().tobytes()
        routes = recording_routes(data)
        routes[RECORD_URL] = [http_error(429, "Too Many Requests", "7"), routes[RECORD_URL]]
        code, _, stderr, _, sleep = self.run_tool(routes)
        self.assertEqual(code, 0, stderr)
        sleep.assert_called_once_with(7)

    def test_connection_failures_suggest_checking_the_connection(self):
        failures = (
            urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
            TimeoutError("timed out"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                _, fake, _ = self.assert_fails(
                    {RECORD_URL: failure}, message="(check your internet connection)"
                )
                self.assertEqual(len(fake.urls), tool.ATTEMPTS)

    def test_answers_that_are_not_json_objects(self):
        for body in (b"<html>maintenance</html>", b"[1, 2]"):
            with self.subTest(body=body):
                self.assert_fails({RECORD_URL: body}, message="Zenodo record 13371136: ")

    def test_oversized_metadata_is_refused(self):
        data = bursty_samples().tobytes()
        routes = recording_routes(data)
        routes[file_url(STEM + ".sigmf-meta")] = b" " * (tool.METADATA_MAX_BYTES + 1)
        self.assert_fails(routes, message="SigMF metadata: larger than the expected")


class CommandLineTests(unittest.TestCase):
    def parse_error(self, *argv: str) -> str:
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            tool.parse_args(list(argv))
        self.assertEqual(caught.exception.code, 2)
        return stderr.getvalue()

    def test_sample_and_start_bounds(self):
        self.assertIn("--samples must be between", self.parse_error("--samples", str(W - 1)))
        self.assertIn("--samples must be between", self.parse_error("--samples", str(MAX_SAMPLES + 1)))
        self.assertIn("--start cannot be negative", self.parse_error("--start", "-1"))
        self.assertEqual(tool.parse_args(["--samples", str(W)]).samples, W)
        self.assertEqual(tool.parse_args([]).samples, MAX_SAMPLES)
        self.assertEqual(tool.parse_args([]).record, RECORD)

    def test_record_and_cap_must_be_plain_numbers(self):
        for record in ("abc", "../13371136", "１２３", "13371136?x=1"):
            with self.subTest(record=record):
                self.assertIn("--record must be", self.parse_error("--record", record))
        for cap in ("0", "-5", "nan", "inf"):
            with self.subTest(cap=cap):
                self.assertIn("--max-mb must be", self.parse_error("--max-mb", cap))

    def test_limits_come_from_the_app(self):
        self.assertEqual(tool.MAX_SAMPLES, MAX_SAMPLES)
        self.assertEqual(tool.csv_text.__defaults__, (MAX_UPLOAD_BYTES,))

    def test_csv_is_cut_at_a_whole_row(self):
        samples = bursty_samples(100)
        full = to_csv(samples.astype(np.complex128))
        content, rows = tool.csv_text(samples, max_bytes=len(full))
        self.assertEqual((content, rows), (full, 100))
        content, rows = tool.csv_text(samples, max_bytes=len(full) - 1)
        self.assertEqual(rows, 99)
        self.assertEqual(content, full[: len(content)])
        self.assertTrue(content.endswith("\r\n"))

    def test_shifted_datetime(self):
        cases = [
            (("2024-08-14T20:46:08Z", 4.096), "2024-08-14T20:46:12.096Z"),
            (("2024-08-14T20:46:08.5Z", 0.5), "2024-08-14T20:46:09Z"),
            (("2024-12-31T23:59:59Z", 1.0), "2025-01-01T00:00:00Z"),
            (("2024-08-14 20:46:08", 0.0), None),
            ((None, 0.0), None),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(tool.shifted_datetime(*args), expected)

    def test_creative_commons_url(self):
        self.assertEqual(tool.creative_commons_url("cc-by-4.0"), CC_BY)
        self.assertEqual(
            tool.creative_commons_url("cc-by-nc-sa-4.0"),
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        )
        self.assertEqual(
            tool.creative_commons_url("cc0-1.0"), "https://creativecommons.org/publicdomain/zero/1.0/"
        )
        self.assertIsNone(tool.creative_commons_url("mit"))

    def test_runs_as_a_script_from_another_folder(self):
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as folder:
            shown = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                cwd=folder, env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertIn("--out-dir", shown.stdout)
            self.assertIn(f"{MAX_SAMPLES:,}", shown.stdout)
            refused = subprocess.run(
                [sys.executable, str(SCRIPT), "--samples", "5"],
                cwd=folder, env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(refused.returncode, 2)
            self.assertIn("--samples must be between", refused.stderr)
            self.assertEqual(os.listdir(folder), [])


if __name__ == "__main__":
    unittest.main()
