#!/usr/bin/env python3
"""Fetch a real radio recording from Zenodo and save it for SpectrumSense.

By default this downloads Zenodo record 13371136, "Test of gr4-packet-modem
through an Intelsat 37e C-band transponder" by Daniel Estévez (CC BY 4.0,
https://doi.org/10.5281/zenodo.13371136). It is a real over-the-air recording
of QPSK packets relayed by a geostationary satellite. The author's blog and the
modem's README describe it as 125 symbols/s with root-raised-cosine pulses,
stored at 500 samples/s (4 samples per symbol). SpectrumSense's demo model was
trained on rectangular pulses at 8 samples per symbol, so its answers on this
file are an experiment, not a check against ground truth.

No account, API key, or radio hardware is needed. The script uses only the
standard library, NumPy, and two small modules of this repository, and runs on
Python 3.10 or newer from any folder.

What it does:
  1. Asks the Zenodo REST API for the record (title, license, file list).
  2. Downloads the SigMF metadata, then streams the SigMF data file with a size
     cap and checks it against the checksums Zenodo and SigMF publish.
  3. Keeps at most 100,000 samples and writes four files to --out-dir:
       NAME.sigmf             SigMF archive: the samples with their sample rate,
                              center frequency, author, and license
       NAME.cf32              complex64 little-endian samples, no metadata
       NAME.csv               I,Q text in the app's CSV export format
       NAME.ATTRIBUTION.txt   credit, license, and changes, as CC BY requires
     NAME is the recording's name and the sample range, for example
     gr4-packet-modem-intelsat37e-test_0-100000.

Examples:
  python tools/fetch_real_iq.py                     # first 100,000 samples
  python tools/fetch_real_iq.py --start 200000 --out-dir data
  python tools/fetch_real_iq.py --record 13143544   # another Zenodo SigMF record
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

# Make the repository's modules importable when run as "python tools/fetch_real_iq.py".
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, to_csv
from signal_config import SAMPLES_PER_WINDOW


API = "https://zenodo.org/api/records/"
DEFAULT_RECORD = "13371136"
HEADERS = {"User-Agent": "fetch_real_iq.py (SpectrumSense teaching project)"}
TIMEOUT_S = 30  # per connection attempt and per network read
ATTEMPTS = 3  # retries cover timeouts, dropped connections, HTTP 429 and 5xx
METADATA_MAX_BYTES = 1_000_000  # API answers and .sigmf-meta files
DEFAULT_MAX_MB = 50.0
SIGMF_VERSION = "1.2.0"
NOT_STATED = "not stated"

# SigMF datatype -> (NumPy type of one I or Q value, offset, full scale).
# Integers are scaled to about [-1, 1] the same way the app's loader does.
DATATYPES = {
    "cf32_le": ("<f4", 0.0, 1.0),
    "ci16_le": ("<i2", 0.0, 32768.0),
    "ci8": ("i1", 0.0, 128.0),
    "cu8": ("u1", 127.5, 127.5),
}

HTTP_HINTS = {
    404: " (not found: check the record number)",
    410: " (the record was removed)",
    429: " (too many requests: wait a minute and try again)",
    500: " (server error: try again later, and check the record number in a browser)",
}


class FetchError(Exception):
    """A problem worth a one-line message rather than a traceback."""


def with_retries(what: str, action):
    """Run action(), retrying network failures that are likely to be temporary."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except urllib.error.HTTPError as exc:  # must come before OSError
            exc.close()  # An error answer holds its connection open until closed.
            if (exc.code != 429 and exc.code < 500) or attempt == ATTEMPTS:
                hint = HTTP_HINTS.get(exc.code, HTTP_HINTS[500] if exc.code > 500 else "")
                reason = printable(str(exc.reason), 80)
                raise FetchError(f"{what}: HTTP {exc.code} {reason}{hint}") from exc
            # Zenodo sends Retry-After on every answer; only a 429 (too many requests) needs it.
            retry_after = str((exc.headers or {}).get("Retry-After", ""))
            use_header = exc.code == 429 and retry_after.isdigit()
            delay = min(int(retry_after), 60) if use_header else 2**attempt
        except (OSError, http.client.HTTPException) as exc:  # timeouts, DNS, resets
            if attempt == ATTEMPTS:
                reason = printable(str(getattr(exc, "reason", None) or exc) or type(exc).__name__)
                raise FetchError(f"{what}: {reason} (check your internet connection)") from exc
            delay = 2**attempt
        print(f"  {what} failed (attempt {attempt} of {ATTEMPTS}); retrying in {delay} s")
        time.sleep(delay)
    raise AssertionError("unreachable")


def get_bytes(url: str, what: str, max_bytes: int = METADATA_MAX_BYTES) -> bytes:
    """Download a small file whole (API answers, metadata)."""

    def action() -> bytes:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise FetchError(f"{what}: larger than the expected {max_bytes:,} bytes")
        return body

    return with_retries(what, action)


def get_json(url: str, what: str) -> dict:
    try:
        answer = json.loads(get_bytes(url, what))
    except (ValueError, RecursionError) as exc:
        raise FetchError(f"{what}: the server did not return valid JSON") from exc
    if not isinstance(answer, dict):
        raise FetchError(f"{what}: expected a JSON object")
    return answer


def stream_download(url: str, size: int, max_bytes: int, what: str) -> bytes:
    """Stream a file in chunks, printing progress and refusing to pass max_bytes.
    A size of 0 means the server did not state one."""
    if size > max_bytes:
        raise FetchError(
            f"{what} is {size / 1e6:.1f} MB, over the {max_bytes / 1e6:g} MB cap. "
            "Raise --max-mb if you really want it."
        )
    stated = f"{size:,} bytes ({size / 1e6:.1f} MB)" if size else "size not stated"
    print(f"Downloading {what}: {stated}")

    def action() -> bytes:
        request = urllib.request.Request(url, headers=HEADERS)
        chunks, received, next_report = [], 0, 0.2
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            while chunk := response.read(256 * 1024):
                received += len(chunk)
                if received > max_bytes:
                    raise FetchError(f"{what}: stopped at the {max_bytes:,}-byte cap")
                chunks.append(chunk)
                if size and received / size >= next_report:
                    done = f"{received / 1e6:5.1f} of {size / 1e6:.1f} MB"
                    print(f"  {done} ({received / size:4.0%})")
                    next_report += 0.2
        if size and received != size:  # A dropped connection can end a read early.
            raise ConnectionError(f"received {received:,} of {size:,} bytes")
        seconds = max(time.monotonic() - started, 1e-3)
        print(f"  done in {seconds:.1f} s ({received / 1e6 / seconds:.1f} MB/s)")
        return b"".join(chunks)

    return with_retries(what, action)


def text(value) -> str:
    """A metadata value if it is text, else an empty string."""
    return value.strip() if isinstance(value, str) else ""


def printable(value: str, limit: int = 200) -> str:
    """Text from the server for the terminal or a text file: control characters
    replaced and the length capped."""
    shown = "".join(char if char.isprintable() else "?" for char in value)
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


def as_number(value) -> float | None:
    """A finite number from SigMF metadata, which sometimes stores numbers as text."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def plain_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def license_key(name: str) -> str:
    """'https://creativecommons.org/licenses/by/4.0/' and 'cc-by-4.0' -> 'cc-by-4.0'."""
    match = re.search(r"licenses/([a-z-]+)/(\d\.\d)", name.lower())
    if match:
        return f"cc-{match[1]}-{match[2]}"
    return re.sub(r"[\s_]+", "-", name.strip().lower())


def creative_commons_url(license_id: str) -> str | None:
    """Zenodo's 'cc-by-4.0' -> 'https://creativecommons.org/licenses/by/4.0/'."""
    match = re.fullmatch(r"cc-(by(?:-nc)?(?:-sa|-nd)?)-(\d\.\d)", license_id.lower())
    if match:
        return f"https://creativecommons.org/licenses/{match[1]}/{match[2]}/"
    if license_id.lower() == "cc0-1.0":
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    return None


def safe_stem(stem: str) -> str:
    """A file-name stem from a name the server chose. Only letters, digits, "_"
    and "-" remain, so the result names a file directly inside the output folder
    on any operating system and cannot look like a command-line option."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:80]
    return safe or "recording"


def file_url(record: str, key: str) -> str:
    """The download address of one file. It is built here rather than taken
    from the API's answer, so a changed answer cannot point anywhere else."""
    return f"{API}{record}/files/{urllib.parse.quote(key, safe='')}/content"


def record_files(record: dict) -> dict[str, dict]:
    """The record's files by name. Zenodo lists each with its size and checksum."""
    files = record.get("files")
    if not isinstance(files, list):
        raise FetchError("The record has no public file list (it may be restricted).")
    return {f["key"]: f for f in files if isinstance(f, dict) and isinstance(f.get("key"), str)}


def pick_recording(names, wanted: str | None) -> str:
    """The name, without extension, of one .sigmf-meta / .sigmf-data pair."""
    names = set(names)
    stems = sorted(k[: -len(".sigmf-meta")] for k in names if k.endswith(".sigmf-meta"))
    stems = [s for s in stems if s + ".sigmf-data" in names]
    if not stems:
        listed = ", ".join(printable(n, 80) for n in sorted(names)[:20]) or "none"
        raise FetchError(
            "This record has no uncompressed .sigmf-meta/.sigmf-data pair "
            f"(files: {listed}). Compressed archives would need unpacking first."
        )
    if wanted is None:
        if len(stems) > 1:
            print(f"Record holds {len(stems)} recordings; using {printable(stems[0])!r} "
                  "(choose with --recording).")
        return stems[0]
    if wanted not in stems:
        raise FetchError(
            f"No recording named {printable(wanted)!r}. "
            f"Choices: {', '.join(printable(s, 80) for s in stems[:20])}"
        )
    return wanted


def check_datatype(datatype: str) -> None:
    """Refuse a SigMF datatype this script cannot convert."""
    if datatype in DATATYPES:
        return
    handled = f"only {', '.join(DATATYPES)} are handled here"
    if not datatype:
        raise FetchError(f"The SigMF metadata does not state core:datatype ({handled}).")
    shown = datatype if re.fullmatch(r"[a-z0-9_]{1,16}", datatype) else "of this recording"
    raise FetchError(f"The SigMF datatype {shown} is not supported ({handled}).")


def check_checksums(data: bytes, data_file: dict, sigmf_global: dict) -> None:
    algorithm, _, expected = text(data_file.get("checksum")).partition(":")
    checked = False
    if algorithm.lower() == "md5":
        if hashlib.md5(data, usedforsecurity=False).hexdigest() != expected.lower():
            raise FetchError("The download does not match Zenodo's MD5 checksum. Try again.")
        print("  MD5 matches the checksum Zenodo publishes.")
        checked = True
    if "core:sha512" in sigmf_global:
        if hashlib.sha512(data).hexdigest() != text(sigmf_global["core:sha512"]).lower():
            raise FetchError("The download does not match core:sha512 in the SigMF metadata.")
        print("  SHA-512 matches core:sha512 in the SigMF metadata.")
        checked = True
    if not checked:
        print("  note: neither Zenodo nor the SigMF metadata gives a checksum to compare.")


def to_complex64(data: bytes, datatype: str) -> np.ndarray:
    check_datatype(datatype)
    component, offset, full_scale = DATATYPES[datatype]
    width = 2 * np.dtype(component).itemsize
    if len(data) % width:
        print(f"  warning: dropping {len(data) % width} trailing bytes (not a whole sample)")
    values = np.frombuffer(data[: len(data) - len(data) % width], dtype=component)
    values = (values.astype(np.float32) - offset) / full_scale
    return values.view(np.complex64)


def capture_at(captures, sample: int) -> tuple[dict, int | None]:
    """The SigMF capture segment in effect at this sample (the first segment if
    none has begun yet), and where the next segment starts (None if none does)."""
    segments = []
    for segment in captures if isinstance(captures, list) else []:
        first = segment.get("core:sample_start", 0) if isinstance(segment, dict) else None
        if isinstance(first, int) and not isinstance(first, bool) and first >= 0:
            segments.append((first, segment))
    segments.sort(key=lambda item: item[0])
    current: dict = segments[0][1] if segments else {}
    for first, segment in segments[1:]:
        if first > sample:
            return current, first
        current = segment
    return current, None


def shifted_datetime(value, seconds: float) -> str | None:
    """A SigMF core:datetime such as "2024-08-14T20:46:08Z" moved later by
    seconds, or None if it is missing or unreadable."""
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", text(value))
    if match is None:
        return None
    try:
        moment = datetime.datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S")
        moment += datetime.timedelta(seconds=float(match[2] or 0) + seconds)
    except (ValueError, OverflowError):
        return None
    if moment.microsecond == 0:
        return f"{moment:%Y-%m-%dT%H:%M:%S}Z"
    return f"{moment:%Y-%m-%dT%H:%M:%S.%f}".rstrip("0") + "Z"


def csv_text(samples: np.ndarray, max_bytes: int = MAX_UPLOAD_BYTES) -> tuple[str, int]:
    """The app's CSV export of the samples, cut after the last whole row that
    fits in max_bytes, and the number of rows it keeps."""
    content = to_csv(samples.astype(np.complex128))
    if len(content) > max_bytes:  # The export is ASCII: one byte per character.
        content = content[: content.rindex("\r\n", 0, max_bytes) + 2]
    return content, content.count("\n") - 1


def write_sigmf_archive(
    path: Path, name: str, metadata: dict, data: bytes, mtime: int | None = None
) -> None:
    """An uncompressed SigMF archive: a folder NAME holding NAME.sigmf-meta and
    NAME.sigmf-data, as the SigMF specification lays it out. Its entries are
    dated now unless mtime (seconds since 1970) is given."""
    meta = (json.dumps(metadata, indent=4) + "\n").encode("utf-8")
    mtime = int(time.time()) if mtime is None else mtime
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        folder = tarfile.TarInfo(name)
        folder.type, folder.mode, folder.mtime = tarfile.DIRTYPE, 0o755, mtime
        archive.addfile(folder)
        for suffix, payload in ((".sigmf-meta", meta), (".sigmf-data", data)):
            member = tarfile.TarInfo(f"{name}/{name}{suffix}")
            member.size, member.mode, member.mtime = len(payload), 0o644, mtime
            archive.addfile(member, io.BytesIO(payload))


def window_list(numbers) -> str:
    return ", ".join(str(n) for n in sorted(numbers))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a real I/Q recording from Zenodo and save it for SpectrumSense."
    )
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help=f"Zenodo record number (default {DEFAULT_RECORD}, Intelsat 37e QPSK)")
    parser.add_argument("--recording",
                        help="SigMF file name without extension, if the record holds several")
    parser.add_argument("--start", type=int, default=0,
                        help="first sample to keep, counting from 0 (default 0)")
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES,
                        help=f"samples to keep, {SAMPLES_PER_WINDOW:,} to {MAX_SAMPLES:,} "
                        f"(default {MAX_SAMPLES:,})")
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB,
                        help=f"download size cap in MB (default {DEFAULT_MAX_MB:g})")
    parser.add_argument("--out-dir", type=Path, default=Path("real_iq"),
                        help="where to write the files (default ./real_iq)")
    args = parser.parse_args(argv)
    if not (args.record.isascii() and args.record.isdigit()):
        parser.error("--record must be a Zenodo record number, such as 13371136")
    if not SAMPLES_PER_WINDOW <= args.samples <= MAX_SAMPLES:
        parser.error(f"--samples must be between {SAMPLES_PER_WINDOW:,} and {MAX_SAMPLES:,}")
    if args.start < 0:
        parser.error("--start cannot be negative")
    if not 0 < args.max_mb < math.inf:
        parser.error("--max-mb must be a positive number")
    return args


def excerpt_metadata(
    source: dict, sigmf: dict, capture: dict, start: int, end: int, total: int, data: bytes
) -> dict:
    """New SigMF metadata for samples start to end - 1 of a recording, saved as cf32_le."""
    about = text(sigmf.get("core:description")) or source["title"]
    datatype = source["datatype"]
    converted = "" if datatype == "cf32_le" else f" (converted from {datatype})"
    description = (
        f'Excerpt of "{about}": samples {start:,} to {end - 1:,} (counting from 0; '
        f"{end - start:,} of {total:,} samples) of {source['file']} in Zenodo record "
        f"{source['record']} ({source['doi_url']}), saved as cf32_le{converted} by "
        "SpectrumSense's fetch_real_iq.py."
    )
    if source["annotated"]:
        description += " The original's annotations are not included."
    section = {
        "core:datatype": "cf32_le",
        "core:version": SIGMF_VERSION,
        "core:num_channels": 1,
        "core:description": description,
        "core:author": text(sigmf.get("core:author")) or source["creators"],
        "core:recorder": "fetch_real_iq.py (SpectrumSense)",
        "core:sha512": hashlib.sha512(data).hexdigest(),  # of the excerpt, not the original
    }
    sample_rate = source["sample_rate"]
    if sample_rate:
        section["core:sample_rate"] = plain_number(sample_rate)
    license_url = source["sigmf_license"]
    if not license_url.startswith(("https://", "http://")):
        license_url = creative_commons_url(source["record_license"])
    if license_url:
        section["core:license"] = license_url
    if text(sigmf.get("core:hw")):
        section["core:hw"] = text(sigmf.get("core:hw"))

    segment: dict = {"core:sample_start": 0}
    if source["frequency"] is not None:
        segment["core:frequency"] = plain_number(source["frequency"])
    # The time of the excerpt's first sample: the segment's time plus the samples between.
    first = capture.get("core:sample_start", 0)
    offset = start - first if isinstance(first, int) else -1
    if offset == 0 or (offset > 0 and sample_rate):
        moment = shifted_datetime(capture.get("core:datetime"), offset / (sample_rate or 1))
        if moment:
            segment["core:datetime"] = moment
    return {"global": section, "captures": [segment], "annotations": []}


def fetch(args: argparse.Namespace) -> list[Path]:
    """Download, check, cut, and save one recording; return the files written."""
    # 1. The record: title, creators, license, DOI and file list.
    record = get_json(API + args.record, f"Zenodo record {args.record}")
    info = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    people = info.get("creators") if isinstance(info.get("creators"), list) else []
    names = [text(person.get("name")) for person in people if isinstance(person, dict)]
    license_info = info.get("license") if isinstance(info.get("license"), dict) else {}
    source = {
        "record": args.record,
        "title": printable(text(info.get("title")) or "untitled", 500),
        "creators": printable("; ".join(name for name in names if name) or NOT_STATED, 500),
        "doi_url": printable(
            text(record.get("doi_url")) or f"https://zenodo.org/records/{args.record}", 500
        ),
        "record_license": printable(text(license_info.get("id")) or NOT_STATED),
    }
    print(f"Record:   {printable(source['title'])}")
    print(f"By:       {printable(source['creators'])}   DOI: {printable(source['doi_url'])}")

    # 2. SigMF metadata, then the samples themselves.
    files = record_files(record)
    stem = pick_recording(files, args.recording)
    data_key = stem + ".sigmf-data"
    meta = get_json(file_url(args.record, stem + ".sigmf-meta"), "SigMF metadata")
    sigmf = meta.get("global") if isinstance(meta.get("global"), dict) else {}
    sample_rate = as_number(sigmf.get("core:sample_rate"))
    source.update(
        file=printable(data_key, 500),
        datatype=text(sigmf.get("core:datatype")),
        sample_rate=sample_rate if sample_rate and sample_rate > 0 else None,
        sigmf_license=printable(text(sigmf.get("core:license")) or NOT_STATED),
        annotated=bool(meta.get("annotations")),
    )
    print(f"License:  Zenodo record: {source['record_license']}   "
          f"SigMF core:license: {source['sigmf_license']}")
    if NOT_STATED not in (source["record_license"], source["sigmf_license"]) and (
        license_key(source["record_license"]) != license_key(source["sigmf_license"])
    ):
        print("  note: the two license fields differ; follow the stricter one.")
    rate_text = (
        f"{source['sample_rate']:,.10g} samples/s" if source["sample_rate"] else NOT_STATED
    )
    print(f"Format:   {printable(source['datatype'] or NOT_STATED)}, sample rate {rate_text}")
    print(f"Hardware: {printable(text(sigmf.get('core:hw')) or NOT_STATED)}")
    if as_number(sigmf.get("core:num_channels", 1)) != 1:
        raise FetchError("This recording has several channels; SpectrumSense expects one.")
    check_datatype(source["datatype"])  # before downloading the samples

    data_file = files[data_key]
    size = data_file.get("size")
    size = size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 0
    data = stream_download(
        file_url(args.record, data_key), size, int(args.max_mb * 1e6), printable(data_key)
    )
    check_checksums(data, data_file, sigmf)
    samples = to_complex64(data, source["datatype"])
    rate = source["sample_rate"]
    duration = f" = {len(samples) / rate:,.1f} s" if rate else ""
    print(f"  {len(samples):,} samples{duration}")

    # 3. Keep at most MAX_SAMPLES samples.
    if len(samples) < SAMPLES_PER_WINDOW:
        raise FetchError(f"This recording has fewer than {SAMPLES_PER_WINDOW:,} samples.")
    if args.start + SAMPLES_PER_WINDOW > len(samples):
        raise FetchError(
            f"--start can be at most {len(samples) - SAMPLES_PER_WINDOW:,} for this recording "
            f"({len(samples):,} samples)."
        )
    end = min(args.start + args.samples, len(samples))
    kept = samples[args.start:end]
    if not np.isfinite(kept).all():
        raise FetchError("These samples contain NaN or infinite values, which the app rejects. "
                         "Try another --start.")
    capture, next_segment = capture_at(meta.get("captures"), args.start)
    source["frequency"] = as_number(capture.get("core:frequency"))
    frequency = source["frequency"]
    frequency_text = NOT_STATED
    if frequency is not None:
        frequency_text = f"{frequency:,.0f} Hz ({frequency / 1e6:,.3f} MHz)"
    print(f"Center frequency: {frequency_text}")
    if next_segment is not None and next_segment < end:
        print(f"  note: a new SigMF capture segment starts at sample {next_segment:,}; the "
              "files keep the center frequency and time of the segment at --start.")

    # 4. Write the files.
    name = f"{safe_stem(stem)}_{args.start}-{end}"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_bytes = kept.astype("<c8").tobytes()
    sigmf_path = out_dir / f"{name}.sigmf"
    metadata = excerpt_metadata(source, sigmf, capture, args.start, end, len(samples), data_bytes)
    write_sigmf_archive(sigmf_path, name, metadata, data_bytes)
    cf32_path = out_dir / f"{name}.cf32"
    cf32_path.write_bytes(data_bytes)
    csv_path = out_dir / f"{name}.csv"
    content, csv_rows = csv_text(kept)
    csv_path.write_text(content, encoding="ascii", newline="")
    attribution = out_dir / f"{name}.ATTRIBUTION.txt"
    now = datetime.datetime.now(datetime.timezone.utc)
    attribution.write_text(
        f'Source: "{source["title"]}" by {source["creators"]}, Zenodo, {source["doi_url"]}\n'
        f"License: {source['record_license']} (Zenodo record); "
        f"core:license {source['sigmf_license']} (SigMF)\n"
        f"Original file: {source['file']} ({printable(source['datatype'])}, sample rate "
        f"{rate_text}, center frequency {frequency_text})\n"
        f"Changes: kept samples {args.start} to {end - 1} (counting from 0); saved them as a "
        f"SigMF archive with new metadata ({sigmf_path.name}), as complex64 ({cf32_path.name}), "
        f"and as I,Q CSV text ({csv_path.name}, first {csv_rows} samples).\n"
        f"Made by fetch_real_iq.py on {now:%Y-%m-%d %H:%M} UTC.\n",
        encoding="utf-8",
    )

    print(f"\nKept samples {args.start:,} to {end - 1:,} ({len(kept):,} samples):")
    print(f"  {sigmf_path}  {sigmf_path.stat().st_size:,} bytes, samples and metadata")
    print(f"  {cf32_path}  {cf32_path.stat().st_size:,} bytes, samples only")
    capped = f" (stops before {MAX_UPLOAD_BYTES:,} bytes)" if csv_rows < len(kept) else ""
    print(f"  {csv_path}  {len(content):,} bytes, first {csv_rows:,} samples{capped}")
    print(f"  {attribution}  (credit and license to keep with the data)")

    # A hint for the app's window slider: loud windows are likely inside a burst.
    count = len(kept) // SAMPLES_PER_WINDOW
    power = np.mean(np.abs(kept[: count * SAMPLES_PER_WINDOW].reshape(count, -1)) ** 2, axis=1)
    order = np.argsort(power, kind="stable") + 1  # the app numbers windows from 1
    shown = min(5, count)
    print(f"\nUpload {sigmf_path.name} in the app's \"Try your own recording\" tab; it keeps "
          "the sample rate and center frequency.")
    print(f"The app classifies each whole {SAMPLES_PER_WINDOW:,}-sample window; this excerpt "
          f"has {count}.")
    print(f"  Loudest windows (in a bursty recording, likely inside a burst): "
          f"{window_list(order[-shown:])}")
    print(f"  Quietest windows (in a bursty recording, likely between bursts): "
          f"{window_list(order[:shown])}")
    return [sigmf_path, cf32_path, csv_path, attribution]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fetch(args)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # Network errors become FetchError, so this is a local file.
        print(f"error: could not write the files: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
