"""Checks of the bundled real recording, its credit, and the script that rebuilds it.

No test uses the network: the rebuild test answers the script's Zenodo
requests from memory.
"""

from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

import numpy as np

from iq_formats import load_recording
from prepare import PrepareSettings, prepare_recording
from real_example import (
    ABOUT,
    AUTHOR,
    CREDIT,
    DOI,
    DOI_URL,
    EXAMPLE_PATH,
    FIRST_SAMPLE,
    LAST_SAMPLE,
    LICENSE,
    LICENSE_URL,
    MAX_EXAMPLE_BYTES,
    ORIGINAL_SAMPLES,
    RECORD,
    SAMPLE_COUNT,
    SAMPLE_RATE,
    TITLE,
)
from signal_config import (
    FEATURE_NAMES,
    REASON_UNFAMILIAR,
    SAMPLES_PER_SYMBOL,
    SAMPLES_PER_WINDOW,
    UNCERTAIN,
)
from spectrum_sense import build_demo_model, classify_recording, explain_features, extract_features


BUILDER = Path(__file__).resolve().parents[1] / "tools" / "make_real_example.py"
FREQUENCY = 3_671_055_830
# The original recording's SigMF metadata, as published on Zenodo.
ORIGINAL_GLOBAL = {
    "core:author": "Daniel Estévez <daniel@destevez.net>",
    "core:datatype": "cf32_le",
    "core:description": TITLE,
    "core:hw": "USRP B200mini connected to the L-band IF of a teleport in Usingen (19m antenna)",
    "core:license": LICENSE_URL,
    "core:recorder": "GNU Radio 3.10",
    "core:sample_rate": SAMPLE_RATE,
    "core:version": "1.0.0",
}
ORIGINAL_CAPTURE = {
    "core:datetime": "2024-08-14T20:46:08Z", "core:frequency": FREQUENCY, "core:sample_start": 0,
}
STEM = "gr4-packet-modem-intelsat37e-test"


def load_builder():
    """Import the script by path: tools/ is a folder of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("make_real_example", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def bundled() -> tuple[dict, bytes]:
    """The bundled archive's metadata and data."""
    with tarfile.open(EXAMPLE_PATH, mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
        stem = "intelsat37e_qpsk_excerpt"
        meta = json.load(archive.extractfile(members[f"{stem}/{stem}.sigmf-meta"]))
        data = archive.extractfile(members[f"{stem}/{stem}.sigmf-data"]).read()
    return meta, data


def bundled_samples() -> np.ndarray:
    return np.frombuffer(bundled()[1], dtype="<c8").astype(np.complex128)


def annotated_packets() -> list[tuple[int, int]]:
    return [
        (item["core:sample_start"], item["core:sample_start"] + item["core:sample_count"])
        for item in bundled()[0]["annotations"]
    ]


def zenodo_answers(data: bytes) -> dict[str, bytes]:
    """What the Zenodo API answers for record 13371136, if its recording held these samples."""
    meta = {
        "global": {**ORIGINAL_GLOBAL, "core:sha512": hashlib.sha512(data).hexdigest()},
        "captures": [ORIGINAL_CAPTURE],
        "annotations": [],
    }
    files = {f"{STEM}.sigmf-meta": json.dumps(meta).encode(), f"{STEM}.sigmf-data": data}
    record = {
        "doi_url": DOI_URL,
        "metadata": {
            "title": TITLE,
            "creators": [{"name": "Estévez, Daniel"}],
            "license": {"id": "cc-by-4.0"},
        },
        "files": [
            {"key": key, "size": len(payload), "checksum": "md5:" + hashlib.md5(payload).hexdigest()}
            for key, payload in files.items()
        ],
    }
    api = f"https://zenodo.org/api/records/{RECORD}"
    answers = {api: json.dumps(record).encode()}
    for key, payload in files.items():
        answers[f"{api}/files/{urllib.parse.quote(key, safe='')}/content"] = payload
    return answers


class BundledFileTests(unittest.TestCase):
    def test_loads_with_its_metadata_and_sample_count(self):
        recording = load_recording(EXAMPLE_PATH.name, EXAMPLE_PATH.read_bytes())
        self.assertEqual(len(recording.samples), SAMPLE_COUNT)
        self.assertEqual(recording.sample_rate, SAMPLE_RATE)
        self.assertEqual(recording.center_frequency, FREQUENCY)
        self.assertEqual(recording.source_format, "SigMF archive (cf32_le)")
        self.assertEqual(SAMPLE_COUNT % SAMPLES_PER_WINDOW, 0)

    def test_stays_small(self):
        self.assertLessEqual(SAMPLE_COUNT, 32 * SAMPLES_PER_WINDOW)
        self.assertLessEqual(EXAMPLE_PATH.stat().st_size, MAX_EXAMPLE_BYTES)
        self.assertLessEqual(MAX_EXAMPLE_BYTES, 300_000)

    def test_metadata_credits_the_source_and_describes_the_excerpt(self):
        meta, data = bundled()
        section = meta["global"]
        self.assertEqual(section["core:datatype"], "cf32_le")
        self.assertEqual(section["core:sample_rate"], SAMPLE_RATE)
        self.assertEqual(section["core:license"], LICENSE_URL)
        self.assertIn(AUTHOR, section["core:author"])
        self.assertEqual(section["core:hw"], ORIGINAL_GLOBAL["core:hw"])
        self.assertEqual(section["core:sha512"], hashlib.sha512(data).hexdigest())
        description = section["core:description"]
        for fragment in (
            f'Excerpt of "{TITLE}"', f"samples {FIRST_SAMPLE:,} to {LAST_SAMPLE:,}",
            f"of {ORIGINAL_SAMPLES:,} samples", f"Zenodo record {RECORD}", DOI_URL,
            "The sample values are unmodified.", "an estimate, not the author's labels",
        ):
            self.assertIn(fragment, description)
        # The capture starts FIRST_SAMPLE / SAMPLE_RATE = 408 seconds into the original.
        self.assertEqual(
            meta["captures"],
            [{"core:sample_start": 0, "core:frequency": FREQUENCY,
              "core:datetime": "2024-08-14T20:52:56Z"}],
        )

    def test_annotations_mark_the_packets_the_detector_finds(self):
        samples = bundled_samples()
        packets = annotated_packets()
        self.assertEqual(packets, builder.find_packets(samples))
        for item in bundled()[0]["annotations"]:
            self.assertEqual(item["core:label"], "packet")
            self.assertIn("BPSK syncword, then QPSK", item["core:comment"])
        # Ten complete packets; the excerpt starts and ends in a gap.
        self.assertEqual(len(packets), 10)
        self.assertEqual(packets, sorted(packets))
        self.assertGreater(packets[0][0], 0)
        self.assertLess(packets[-1][1], SAMPLE_COUNT)

    def test_packet_numbers_in_the_explanation_match_the_recording(self):
        samples = bundled_samples()
        packets = annotated_packets()
        lengths = [stop - start for start, stop in packets]
        gaps = [start - stop for (_, stop), (start, _) in zip(packets, packets[1:])]
        periods = np.diff([start for start, _ in packets])
        self.assertIn("each packet lasts about 2,200 samples (4.4 seconds)", ABOUT)
        self.assertTrue(all(2_150 <= length <= 2_250 for length in lengths), lengths)
        self.assertIn("about every 3,090 samples (6.2 seconds)", ABOUT)
        self.assertAlmostEqual(float(np.median(periods)), 3_090, delta=30)
        self.assertIn("gaps of about 900 samples", ABOUT)
        self.assertTrue(all(850 <= gap <= 950 for gap in gaps), gaps)
        self.assertIn("every gap is shorter than a 1,024-sample window", ABOUT)
        self.assertLess(max(gaps + [packets[0][0], SAMPLE_COUNT - packets[-1][1]]), SAMPLES_PER_WINDOW)
        self.assertIn("about 6 dB stronger than the gaps", ABOUT)
        self.assertAlmostEqual(builder.packet_levels(samples, packets), 6, delta=1)

    def test_each_packet_opens_with_a_two_phase_stretch(self):
        # BPSK squared is a steady tone (a strong line in the spectrum of x**2);
        # QPSK squared is not. Compare each packet's first 240 samples with a later stretch.
        def squared_line(part: np.ndarray) -> float:
            part = part / np.sqrt(np.mean(np.abs(part) ** 2))
            return float(np.abs(np.fft.fft(part**2, 4096)).max() / len(part))

        samples = bundled_samples()
        for start, _ in annotated_packets():
            with self.subTest(packet=start):
                opening = squared_line(samples[start + 8 : start + 248])
                later = squared_line(samples[start + 800 : start + 1040])
                self.assertGreater(opening, 2 * later)

    def test_model_answers_match_the_explanation(self):
        model = build_demo_model().model
        result = classify_recording(model, bundled_samples())
        packets = annotated_packets()
        inside, mixed = [], []
        for index, start in enumerate(result.starts):
            stop = start + SAMPLES_PER_WINDOW
            in_packet = any(first <= start and stop <= last for first, last in packets)
            (inside if in_packet else mixed).append(index)
        # Every window holds some packet: no window is gap alone.
        for index in mixed:
            start = result.starts[index]
            self.assertTrue(any(first < start + SAMPLES_PER_WINDOW and start < last
                                for first, last in packets))
        self.assertIn("most windows hold part of a gap and part of a packet", ABOUT)
        self.assertGreater(len(mixed), len(inside))
        self.assertEqual({result.labels[i] for i in mixed}, {UNCERTAIN})
        self.assertEqual({result.reasons[i] for i in mixed}, {REASON_UNFAMILIAR})
        self.assertIn('Windows that lie entirely inside a packet are answered "uncertain" too', ABOUT)
        self.assertEqual({result.labels[i] for i in inside}, {UNCERTAIN})
        self.assertEqual({result.reasons[i] for i in inside}, {REASON_UNFAMILIAR})
        # "Their fourth-power line ... is about half as strong as the model's own
        # QPSK would show": the evidence for each window inside a packet.
        self.assertIn("is about half as strong as the model's own QPSK would show", ABOUT)
        line = FEATURE_NAMES.index("fourth-power line")
        samples = bundled_samples()
        for index in inside:
            start = result.starts[index]
            features = extract_features(samples[start : start + SAMPLES_PER_WINDOW])
            evidence = explain_features(model, features)
            with self.subTest(window=index + 1):
                self.assertEqual(evidence.nearest, "QPSK")
                self.assertLess(evidence.values[line], 0.6 * evidence.expected[line])
                self.assertGreater(evidence.distance_shares[line], 0.3)

    def test_preparing_the_excerpt_does_not_help(self):
        model = build_demo_model().model
        recording = load_recording(EXAMPLE_PATH.name, EXAMPLE_PATH.read_bytes())
        prepared = prepare_recording(recording, PrepareSettings(symbol_rate=125.0))
        self.assertEqual(prepared.samples_per_symbol, SAMPLES_PER_SYMBOL)
        self.assertAlmostEqual(prepared.filled_share, 0.5)
        result = classify_recording(model, prepared.recording.samples)
        self.assertEqual(len(result.labels), 60)
        self.assertEqual(set(result.labels), {UNCERTAIN})
        self.assertEqual(set(result.reasons), {REASON_UNFAMILIAR})
        # The README says the windows lie well beyond the distance limit.
        self.assertGreater(float(result.distances.min()), 1.5 * model.distance_threshold)
        # Keeping 170 Hz around the signal first does not help either.
        narrowed = prepare_recording(
            recording, PrepareSettings(offset=-80.0, bandwidth=170.0, symbol_rate=125.0)
        )
        narrowed_result = classify_recording(model, narrowed.recording.samples)
        self.assertEqual(set(narrowed_result.labels), {UNCERTAIN})
        # A shift alone changes no answer.
        shifted = prepare_recording(recording, PrepareSettings(offset=-80.0))
        self.assertEqual(
            classify_recording(model, shifted.recording.samples).labels,
            classify_recording(model, recording.samples).labels,
        )
        for fragment in (
            "a symbol rate of 125 resamples the excerpt to 8 samples per symbol and 60 windows",
            'every one of them is answered "uncertain" because it is unlike the training data',
            "leaves half of the band without noise",
        ):
            self.assertIn(fragment, ABOUT)

    def test_credit_names_the_author_links_and_changes(self):
        self.assertIn(f'"{TITLE}" by {AUTHOR}', CREDIT)
        self.assertIn(f"[doi:{DOI}]({DOI_URL})", CREDIT)
        self.assertIn(f"[{LICENSE}]({LICENSE_URL})", CREDIT)
        self.assertIn(f"samples {FIRST_SAMPLE:,} to {LAST_SAMPLE:,}", CREDIT)
        self.assertIn("unchanged", CREDIT)
        attribution = (EXAMPLE_PATH.parent / "ATTRIBUTION.md").read_text(encoding="utf-8")
        for fragment in (
            TITLE, AUTHOR, DOI_URL, LICENSE, LICENSE_URL, f"{FIRST_SAMPLE:,} to {LAST_SAMPLE:,}",
            "tools/make_real_example.py", hashlib.sha256(EXAMPLE_PATH.read_bytes()).hexdigest(),
        ):
            self.assertIn(fragment, attribution)


class RebuildTests(unittest.TestCase):
    def test_rebuilding_from_the_record_gives_the_bundled_file_byte_for_byte(self):
        # Only the excerpt is known offline, so the rest of the original is zeros.
        original = np.zeros(ORIGINAL_SAMPLES, dtype="<c8")
        original[FIRST_SAMPLE : LAST_SAMPLE + 1] = np.frombuffer(bundled()[1], dtype="<c8")
        answers = zenodo_answers(original.tobytes())
        requested = []

        def urlopen(request, timeout=None):
            requested.append(request.full_url)
            return BytesIO(answers[request.full_url])

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / EXAMPLE_PATH.name
            with patch("urllib.request.urlopen", urlopen), patch("time.sleep"):
                with redirect_stdout(StringIO()) as output:
                    code = builder.main(["--out", str(out)])
            self.assertEqual(code, 0, output.getvalue())
            self.assertEqual(out.read_bytes(), EXAMPLE_PATH.read_bytes())
        self.assertEqual(set(requested), set(answers))
        self.assertIn("10 complete packets", output.getvalue())

    def test_find_packets_marks_complete_bursts_only(self):
        rng = np.random.default_rng(4)
        samples = (rng.standard_normal(12_000) + 1j * rng.standard_normal(12_000)) / np.sqrt(2)
        bursts = [(0, 700), (1_500, 3_700), (4_600, 6_800), (7_700, 9_900), (11_000, 12_000)]
        for start, stop in bursts:
            samples[start:stop] *= 2  # 6 dB stronger, like the recording's packets
        samples[5_000:5_030] = 0  # A short dip does not split a packet.
        found = builder.find_packets(samples)
        # The bursts cut by either end are left out.
        self.assertEqual(len(found), 3)
        for (start, stop), (true_start, true_stop) in zip(found, bursts[1:4]):
            self.assertLessEqual(abs(start - true_start), builder.SMOOTHING // 2)
            self.assertLessEqual(abs(stop - true_stop), builder.SMOOTHING // 2)


if __name__ == "__main__":
    unittest.main()
