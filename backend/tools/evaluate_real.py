#!/usr/bin/env python3
"""Measure the app's demo model on real recordings whose content is known.

The model is the one the app trains (spectrum_sense.build_demo_model), which
has seen only simulated signals. Every complete 1,024-sample window of each
recording is classified twice: as recorded, and after the app's Prepare step
(prepare.prepare_recording) resamples it to the model's 8 samples per symbol.
The answers are then counted against what each window should hold:

  Intelsat 37e (Daniel Estévez, Zenodo record 13371136, CC BY 4.0): QPSK
    packets relayed by a satellite, with gaps of noise between them. Which
    windows lie in a packet or a gap is estimated from signal power (see
    window_parts). That is a heuristic ground truth, not the author's labels,
    and each packet opens with a 64-symbol BPSK syncword, which it leaves out.
  MIGOU-MOD (Ramiro Utrilla, Mendeley Data fkwr8mzndr, CC BY 4.0): real
    BPSK sent over the air and recorded 1 m from the transmitter; the dataset
    labels it BPSK.

By default only the bundled excerpt of the Intelsat recording is used, which
takes a few seconds and needs no network. --fetch also downloads the whole
Intelsat recording (4.8 MB) from Zenodo and reads the first 4 MB of MIGOU-MOD's
3.3 GB zip with an HTTP range request; a source that cannot be fetched is
skipped with a message. The results go to real_data_results.md (a Markdown
table and notes) and real_data_results.csv in --output-dir.

Examples:
  python tools/evaluate_real.py
  python tools/evaluate_real.py --fetch --output-dir my_results
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import datetime
from io import StringIO
from pathlib import Path
import pickletools
import platform
import re
import struct
import sys
import urllib.request
import zlib

import numpy as np

# Make the other tools and the repository's modules importable from any folder.
TOOLS = Path(__file__).resolve().parent
for folder in (TOOLS, TOOLS.parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import fetch_real_iq
from fetch_real_iq import FetchError, printable
import make_real_example
from iq_formats import Recording, load_recording
from prepare import PrepareSettings, prepare_recording
import real_example
from signal_config import CLASSES, SAMPLES_PER_SYMBOL, SAMPLES_PER_WINDOW, UNCERTAIN
from spectrum_sense import DemoModel, build_demo_model, classify_recording, decision_order


DEFAULT_OUTPUT = Path("real_results")
REPORT_NAME = "real_data_results.md"
CSV_NAME = "real_data_results.csv"
DECISIONS = decision_order(CLASSES)  # noise, BPSK, QPSK, uncertain, not classifiable

# Parts of a recording. ALL covers every window; the others split them.
ALL = "all windows"
PACKET = "inside a packet"
GAP = "in a gap"
MIXED = "mixed"

EXCERPT_SOURCE = "Intelsat 37e excerpt (bundled)"
FULL_SOURCE = "Intelsat 37e, whole recording (Zenodo)"
MIGOU_SOURCE = "MIGOU-MOD BPSK at 1 m (Mendeley Data)"

# Intelsat 37e: 125 symbols per second recorded at 500 samples per second.
INTELSAT_SYMBOL_RATE = 125.0
# The 64-symbol BPSK syncword at the start of each packet, in recorded samples.
SYNCWORD_SAMPLES = 64 * round(real_example.SAMPLE_RATE / INTELSAT_SYMBOL_RATE)
# On simulated bursts, the edges that the power rule finds are within half its
# smoothing length of the true edges, so windows keep this far from them.
EDGE_MARGIN = make_real_example.SMOOTHING // 2
INTELSAT_TRUTH = {PACKET: "QPSK", GAP: "noise"}
INTELSAT_CONDITIONS = (
    ("as recorded (4 samples per symbol)", PrepareSettings()),
    (
        f"resampled by 2/1 to {SAMPLES_PER_SYMBOL} samples per symbol",
        PrepareSettings(symbol_rate=INTELSAT_SYMBOL_RATE),
    ),
)
# The whole Intelsat recording is 4.8 MB; refuse anything much larger.
INTELSAT_MAX_BYTES = 20_000_000

# MIGOU-MOD: one zip holding one pickle of float32 arrays, each (vectors, 2, 128):
# I then Q for 128 consecutive samples per vector. The first array holds BPSK
# recorded 1 m from the transmitter, which sent 8 samples per symbol at 200,000
# samples per second; the recording has about 20 samples per symbol (its
# symbol-rate line lies at 0.05 cycles per sample).
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/"
MENDELEY_FILES = "https://data.mendeley.com/public-files/datasets/"
MIGOU_DATASET = "fkwr8mzndr"
MIGOU_DOI_URL = "https://doi.org/10.17632/fkwr8mzndr.1"
MIGOU_ZIP = "MIGOU-MOD_dataset_19.08_400000x128.zip"
MIGOU_KEY = ("BPSK", "ota_1m")
MIGOU_SHAPE = (400_000, 2, 128)
# Bytes read from the start of the zip: about 8,700 vectors, 1.1 million samples.
MIGOU_READ_BYTES = 4_000_000
MIGOU_TRUTH = {ALL: "BPSK"}
MIGOU_CONDITIONS = (
    ("as recorded (about 20 samples per symbol)", PrepareSettings()),
    (f"resampled by 2/5 to {SAMPLES_PER_SYMBOL} samples per symbol", PrepareSettings(factor=0.4)),
)
# The inflated data may be at most this large, whatever the download holds.
MAX_INFLATED_BYTES = 64_000_000
# The pickle's opcodes before the first array's values fit in this many bytes.
PICKLE_HEADER_LIMIT = 4_096
FILE_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
# A zip file's local header: signature, version, flags, method, time, date,
# CRC-32, sizes, and the lengths of the name and extra field that follow it.
ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
ZIP_SIGNATURE = 0x04034B50
DEFLATE = 8


@dataclass(frozen=True, eq=False)
class Evaluated:
    """The model's answers for one recording under one condition."""

    source: str
    condition: str
    labels: tuple[str, ...]  # the decision for each window
    parts: tuple[str, ...]  # the part of the recording that each window lies in
    truth: dict[str, str]  # the right answer for each part that has one
    part_order: tuple[str, ...]  # the parts listed after ALL, in table order


@dataclass(frozen=True)
class Row:
    """Answer counts for the windows of one part of a recording under one condition."""

    source: str
    condition: str
    part: str
    truth: str  # the right answer, or "" when the part has no single one
    counts: tuple[int, ...]  # windows per decision, in DECISIONS order

    def count(self, decision: str) -> int:
        return self.counts[DECISIONS.index(decision)]

    @property
    def windows(self) -> int:
        return sum(self.counts)

    @property
    def confident(self) -> int:
        """Windows answered with a class rather than "uncertain"."""
        return sum(self.count(name) for name in CLASSES)

    @property
    def correct(self) -> int | None:
        return self.count(self.truth) if self.truth else None

    @property
    def correct_share(self) -> float | None:
        """Correct answers among confident ones; None without a right answer
        or a confident answer."""
        return self.correct / self.confident if self.truth and self.confident else None

    @property
    def uncertain_share(self) -> float | None:
        return self.count(UNCERTAIN) / self.windows if self.windows else None


def window_parts(
    stretches: Sequence[tuple[int, int]], bounds: Sequence[tuple[float, float]]
) -> tuple[str, ...]:
    """PACKET, GAP, or MIXED for each window.

    stretches are the recording's loud stretches (make_real_example.
    loud_stretches), and bounds give each window's first sample and one past
    its last on the same scale, the recording as recorded. A window is PACKET
    when it lies inside a loud stretch of at least MIN_PACKET_SAMPLES samples,
    at least EDGE_MARGIN samples before its end and SYNCWORD_SAMPLES +
    EDGE_MARGIN after its start, which leaves out the BPSK syncword; GAP when
    it stays at least EDGE_MARGIN samples away from every loud stretch; and
    MIXED otherwise.
    """
    packets = [
        (start, stop)
        for start, stop in stretches
        if stop - start >= make_real_example.MIN_PACKET_SAMPLES
    ]
    parts = []
    for first, end in bounds:
        if any(
            start + SYNCWORD_SAMPLES + EDGE_MARGIN <= first and end <= stop - EDGE_MARGIN
            for start, stop in packets
        ):
            parts.append(PACKET)
        elif all(
            end <= start - EDGE_MARGIN or stop + EDGE_MARGIN <= first
            for start, stop in stretches
        ):
            parts.append(GAP)
        else:
            parts.append(MIXED)
    return tuple(parts)


def classify_prepared(model, recording: Recording, settings: PrepareSettings):
    """The whole-file classification of the prepared recording, and the
    resampling ratio (up, down) that preparing it used."""
    prepared = prepare_recording(recording, settings, limit_samples=False)
    return classify_recording(model, prepared.recording.samples), prepared.ratio


def evaluate_intelsat(model, recording: Recording, source: str) -> list[Evaluated]:
    """Each condition of an Intelsat 37e recording (500 samples per second),
    with each window's part found on the recording as recorded."""
    if recording.sample_rate != real_example.SAMPLE_RATE:
        raise ValueError(
            f"Expected {real_example.SAMPLE_RATE} samples per second, the Intelsat 37e "
            "recording's rate."
        )
    stretches = make_real_example.loud_stretches(recording.samples)
    evaluated = []
    for condition, settings in INTELSAT_CONDITIONS:
        result, (up, down) = classify_prepared(model, recording, settings)
        # Resampling keeps the timing, so prepared sample k lies at recorded sample k * down / up.
        bounds = [
            (start * down / up, (start + SAMPLES_PER_WINDOW) * down / up)
            for start in result.starts
        ]
        evaluated.append(
            Evaluated(
                source, condition, result.labels, window_parts(stretches, bounds),
                INTELSAT_TRUTH, (PACKET, GAP, MIXED),
            )
        )
    return evaluated


def evaluate_migou(model, samples: np.ndarray) -> list[Evaluated]:
    """Each condition of MIGOU-MOD's BPSK samples; every window should be BPSK."""
    recording = Recording(samples, None, None, "MIGOU-MOD")
    evaluated = []
    for condition, settings in MIGOU_CONDITIONS:
        result, _ = classify_prepared(model, recording, settings)
        count = len(result.labels)
        evaluated.append(
            Evaluated(MIGOU_SOURCE, condition, result.labels, (ALL,) * count, MIGOU_TRUTH, ())
        )
    return evaluated


def summarize(item: Evaluated) -> list[Row]:
    """A row for all windows, then one per part in item.part_order."""
    rows = []
    for part in (ALL, *item.part_order):
        labels = [
            label for label, where in zip(item.labels, item.parts) if part in (ALL, where)
        ]
        counts = tuple(labels.count(name) for name in DECISIONS)
        rows.append(Row(item.source, item.condition, part, item.truth.get(part, ""), counts))
    return rows


# --- Fetching -------------------------------------------------------------


def fetch_intelsat() -> Recording:
    """The whole Intelsat 37e recording from Zenodo, checked against the
    checksums that Zenodo and the SigMF metadata publish."""
    record_id = real_example.RECORD
    record = fetch_real_iq.get_json(fetch_real_iq.API + record_id, f"Zenodo record {record_id}")
    files = fetch_real_iq.record_files(record)
    stem = fetch_real_iq.pick_recording(files, None)
    meta = fetch_real_iq.get_json(
        fetch_real_iq.file_url(record_id, stem + ".sigmf-meta"), "SigMF metadata"
    )
    sigmf = meta.get("global") if isinstance(meta.get("global"), dict) else {}
    datatype = fetch_real_iq.text(sigmf.get("core:datatype"))
    rate = fetch_real_iq.as_number(sigmf.get("core:sample_rate"))
    if datatype != "cf32_le" or rate != real_example.SAMPLE_RATE:
        raise FetchError(
            "The recording is no longer cf32_le at "
            f"{real_example.SAMPLE_RATE} samples per second, which this evaluation assumes."
        )
    data_key = stem + ".sigmf-data"
    size = files[data_key].get("size")
    size = size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 0
    data = fetch_real_iq.stream_download(
        fetch_real_iq.file_url(record_id, data_key), size, INTELSAT_MAX_BYTES, printable(data_key)
    )
    fetch_real_iq.check_checksums(data, files[data_key], sigmf)
    samples = fetch_real_iq.to_complex64(data, datatype).astype(np.complex128)
    if len(samples) < SAMPLES_PER_WINDOW or not np.isfinite(samples).all():
        raise FetchError("The recording is too short or holds values that are not finite.")
    return Recording(samples, rate, None, f"SigMF ({datatype})")


def get_range(url: str, count: int, what: str) -> bytes:
    """The first count bytes of a file, with an HTTP Range request. A server
    that ignores the range would send the whole file, so its answer is refused."""
    headers = {**fetch_real_iq.HEADERS, "Range": f"bytes=0-{count - 1}"}

    def action() -> bytes:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=fetch_real_iq.TIMEOUT_S) as response:
            if getattr(response, "status", None) != 206:
                raise FetchError(f"{what}: the server did not answer the range request")
            body = response.read(count + 1)
        if len(body) != count:  # A dropped connection can end a read early.
            raise ConnectionError(f"received {len(body):,} of {count:,} bytes")
        return body

    print(f"Downloading {what}: the first {count:,} bytes")
    return fetch_real_iq.with_retries(what, action)


def inflate_first_entry(head: bytes, limit: int = MAX_INFLATED_BYTES) -> bytes:
    """As much of a zip file's first entry as the file's first bytes hold,
    decompressed: at most limit bytes."""
    if len(head) < ZIP_LOCAL_HEADER.size:
        raise FetchError("The download is too short to be a zip file.")
    fields = ZIP_LOCAL_HEADER.unpack_from(head)
    signature, flags, method, name_length, extra_length = (
        fields[0], fields[2], fields[3], fields[9], fields[10]
    )
    if signature != ZIP_SIGNATURE:
        raise FetchError("The download is not a zip file.")
    if method != DEFLATE or flags & 1:
        raise FetchError("The zip file's first entry is not deflate-compressed, or is encrypted.")
    start = ZIP_LOCAL_HEADER.size + name_length + extra_length
    try:
        return zlib.decompressobj(-15).decompress(head[start:], limit)
    except zlib.error as exc:
        raise FetchError("The zip file's first entry could not be decompressed.") from exc


def _pickle_header(header: bytes) -> tuple[list[str], list[int], str]:
    """The texts and small integers in a pickle's opcodes, and the name of the
    last opcode. The opcodes are only read, never run."""
    texts: list[str] = []
    numbers: list[int] = []
    last = ""
    try:
        for opcode, arg, _ in pickletools.genops(header):
            last = opcode.name
            if isinstance(arg, str):
                texts.append(arg)
            elif isinstance(arg, bytes) and len(arg) % 4 == 0:
                # NumPy stores a string scalar's characters as UTF-32.
                texts.append(arg.decode("utf-32-le", errors="replace"))
            elif opcode.name in ("BININT", "BININT1", "BININT2"):
                numbers.append(arg)
    except ValueError:  # The header ends where the array's values begin.
        pass
    return texts, numbers, last


def migou_samples(raw: bytes, shape: tuple[int, int, int] = MIGOU_SHAPE) -> np.ndarray:
    """The complex samples of the first array in the start of MIGOU-MOD's
    pickle, vector after vector in stored order.

    The pickle is never loaded. Its opcodes up to the array's values must
    describe the array of key MIGOU_KEY with this shape, stored as C-ordered,
    little-endian float32; only the complete vectors that arrived are kept.
    """
    vectors, rows, length = shape
    vector_bytes = rows * length * 4
    # BINBYTES and the array's size in bytes: its values follow.
    marker = b"B" + (vectors * vector_bytes).to_bytes(4, "little")
    position = raw.find(marker, 0, PICKLE_HEADER_LIMIT)
    texts, numbers, last = _pickle_header(raw[: max(position, 0)])
    key = [text for text in texts if text in MIGOU_KEY]
    shape_found = any(
        tuple(numbers[index : index + 3]) == shape for index in range(len(numbers))
    )
    if not (
        position >= 0 and key[:2] == list(MIGOU_KEY) and shape_found and "f4" in texts
        and ">" not in texts and last == "NEWFALSE"
    ):
        raise FetchError(
            "The data do not start with the expected array (BPSK recorded at 1 m, "
            f"{vectors:,} vectors of {rows} x {length} float32 values)."
        )
    start = position + len(marker)
    usable = min(len(raw) - start, vectors * vector_bytes) // vector_bytes
    values = np.frombuffer(raw, "<f4", count=usable * rows * length, offset=start)
    values = values.astype(np.float64).reshape(usable, rows, length)
    samples = (values[:, 0, :] + 1j * values[:, 1, :]).ravel()
    if len(samples) < SAMPLES_PER_WINDOW:
        raise FetchError("Too few samples arrived to fill one window.")
    if not np.isfinite(samples).all():
        raise FetchError("The samples hold values that are not finite.")
    return samples


def fetch_migou(read_bytes: int = MIGOU_READ_BYTES) -> np.ndarray:
    """The BPSK samples at the start of MIGOU-MOD's zip, read with one range request."""
    dataset = fetch_real_iq.get_json(
        MENDELEY_API + MIGOU_DATASET, f"Mendeley Data record {MIGOU_DATASET}"
    )
    files = dataset.get("files") if isinstance(dataset.get("files"), list) else []
    entry = next(
        (item for item in files if isinstance(item, dict) and item.get("filename") == MIGOU_ZIP),
        {},
    )
    file_id = entry.get("id")
    if not (isinstance(file_id, str) and FILE_ID.fullmatch(file_id)):
        raise FetchError(f"The record no longer lists {MIGOU_ZIP}.")
    licence = dataset.get("data_licence") if isinstance(dataset.get("data_licence"), dict) else {}
    shown = fetch_real_iq.text(licence.get("short_name")) or fetch_real_iq.NOT_STATED
    print(f"License:  {printable(shown, 80)}")
    # The address is built here, not taken from the answer, like fetch_real_iq's.
    url = f"{MENDELEY_FILES}{MIGOU_DATASET}/files/{file_id}/file_downloaded"
    head = get_range(url, read_bytes, "the start of MIGOU-MOD's zip")
    samples = migou_samples(inflate_first_entry(head))
    print(f"  {len(samples) // MIGOU_SHAPE[2]:,} vectors, {len(samples):,} samples")
    return samples


# --- Reports --------------------------------------------------------------


def csv_fields() -> tuple[str, ...]:
    return (
        "source", "condition", "part", "ground_truth", "windows",
        *(name.replace(" ", "_") for name in DECISIONS),
        "confident", "correct", "correct_share", "uncertain_share",
    )


def results_csv(rows: Sequence[Row]) -> str:
    """One line per row; blank cells where a value does not apply."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(csv_fields())
    for row in rows:
        shares = (row.correct_share, row.uncertain_share)
        writer.writerow(
            (row.source, row.condition, row.part, row.truth, row.windows, *row.counts,
             row.confident, "" if row.correct is None else row.correct,
             *("" if share is None else f"{share:.4f}" for share in shares))
        )
    return output.getvalue()


def _share(part: int, whole: int) -> str:
    return f"{part:,} of {whole:,}" + (f" ({part / whole:.0%})" if whole else "")


def markdown_table(rows: Sequence[Row]) -> str:
    """The rows as a Markdown table. The "not classifiable" column appears
    only when some window is not classifiable."""
    decisions = [name for name in DECISIONS if name in CLASSES or name == UNCERTAIN]
    if any(count for row in rows for name, count in zip(DECISIONS, row.counts)
           if name not in decisions):
        decisions = list(DECISIONS)
    header = (
        "Recording", "Condition", "Windows", "Right answer", "Count", *decisions,
        "Correct when not uncertain", "Uncertain",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in rows:
        correct = "n/a" if row.correct is None else _share(row.correct, row.confident)
        cells = (
            row.source, row.condition, row.part, row.truth or "none", f"{row.windows:,}",
            *(f"{row.count(name):,}" for name in decisions),
            correct, _share(row.count(UNCERTAIN), row.windows),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def versions() -> str:
    import scipy
    import sklearn

    return (
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}"
    )


def report(
    rows: Sequence[Row],
    demo: DemoModel,
    *,
    command: str,
    skipped: Sequence[tuple[str, str]] = (),
    date: datetime.date | None = None,
) -> str:
    """The Markdown report: when and how it was measured, the table, and how
    to read it."""
    date = date or datetime.datetime.now(datetime.timezone.utc).date()
    sources = {row.source for row in rows}
    lines = [
        "# SpectrumSense on real recordings",
        "",
        f"Measured on {date.isoformat()} with `{command}` ({versions()}). The model is the "
        "app's demo model, trained only on simulated signals. On held-out simulated "
        f'windows it answers "uncertain" for {demo.false_reject_rate:.0%} of them and is '
        f"right about {demo.accepted_accuracy:.0%} of the rest.",
        "",
        markdown_table(rows),
        "How to read it:",
        "",
        f"- **Windows:** every complete {SAMPLES_PER_WINDOW:,}-sample window, as the app "
        "classifies an upload. *Correct when not uncertain* counts the answers that name "
        "a class and are right; it is n/a where the windows have no single right answer.",
        f"- **Intelsat 37e, a heuristic ground truth.** Power averaged over "
        f"{make_real_example.SMOOTHING} samples is compared with the midpoint of its quiet "
        "(10th percentile) and loud (90th percentile) levels, as tools/make_real_example.py "
        "does to mark the packets. A window is *inside a packet* (right answer QPSK) when "
        f"it lies within a loud stretch of at least {make_real_example.MIN_PACKET_SAMPLES} "
        f"samples, at least {EDGE_MARGIN} samples from its end and "
        f"{SYNCWORD_SAMPLES + EDGE_MARGIN} from its start: each packet opens with a "
        f"64-symbol BPSK syncword ({SYNCWORD_SAMPLES} samples as recorded), which this leaves "
        f"out. It is *in a gap* (right answer noise) when it stays at least {EDGE_MARGIN} "
        "samples from every loud stretch, and *mixed* otherwise. These are estimates "
        "from power, not the author's labels. Windows of the resampled recording are "
        "placed on the samples as recorded.",
    ]
    if MIGOU_SOURCE in sources:
        lines.append(
            "- **MIGOU-MOD:** the right answer is the dataset's label, BPSK. The dataset "
            f"stores {MIGOU_SHAPE[2]}-sample vectors, each normalized on its own; "
            f"{SAMPLES_PER_WINDOW // MIGOU_SHAPE[2]} consecutive vectors form one window. "
            f"These are the first {MIGOU_READ_BYTES / 1e6:g} MB of its "
            f"{MIGOU_ZIP}, read without unpickling."
        )
    lines += [
        "",
        "Sources, used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/):",
        "",
        f'- "{real_example.TITLE}" by {real_example.AUTHOR}, Zenodo, {real_example.DOI_URL}. '
        "The bundled excerpt keeps samples "
        f"{real_example.FIRST_SAMPLE:,} to {real_example.LAST_SAMPLE:,} unchanged.",
        '- "MIGOU-MOD: A dataset of modulated radio signals acquired with MIGOU, a '
        f'low-power IoT experimental platform" by Ramiro Utrilla, Mendeley Data, '
        f"{MIGOU_DOI_URL}.",
    ]
    if skipped:
        lines += ["", "Skipped because they could not be fetched:", ""]
        lines += [f"- {name}: {printable(reason, 300)}" for name, reason in skipped]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the demo model on real recordings whose content is known."
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="also download the whole Intelsat 37e recording (4.8 MB, Zenodo) and "
        f"{MIGOU_READ_BYTES / 1e6:g} MB of MIGOU-MOD's BPSK (Mendeley Data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"where to write {REPORT_NAME} and {CSV_NAME} (default ./{DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    """Evaluate every source, write the report and CSV, and return their paths."""
    print("Training the demo model on simulated signals, as the app does.")
    demo = build_demo_model()
    model = demo.model
    excerpt = load_recording(real_example.EXAMPLE_PATH.name, real_example.EXAMPLE_PATH.read_bytes())
    evaluated = evaluate_intelsat(model, excerpt, EXCERPT_SOURCE)
    skipped: list[tuple[str, str]] = []
    if args.fetch:
        fetchers = (
            (FULL_SOURCE, lambda: evaluate_intelsat(model, fetch_intelsat(), FULL_SOURCE)),
            (MIGOU_SOURCE, lambda: evaluate_migou(model, fetch_migou())),
        )
        for name, evaluate in fetchers:
            print(f"\n{name}")
            try:
                evaluated += evaluate()
            except FetchError as exc:
                print(f"skipped {name}: {exc}", file=sys.stderr)
                skipped.append((name, str(exc)))
    rows = [row for item in evaluated for row in summarize(item)]
    command = "python tools/evaluate_real.py" + (" --fetch" if args.fetch else "")
    text = report(rows, demo, command=command, skipped=skipped)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report_path, csv_path = out / REPORT_NAME, out / CSV_NAME
    report_path.write_text(text, encoding="utf-8")
    csv_path.write_text(results_csv(rows), encoding="utf-8", newline="")
    print("\n" + markdown_table(rows))
    print(f"Wrote {report_path} and {csv_path}")
    return [report_path, csv_path]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except OSError as exc:  # Network errors become FetchError, so this is a local file.
        print(f"error: could not write the results: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
