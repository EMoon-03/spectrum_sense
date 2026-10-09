#!/usr/bin/env python3
"""Rebuild examples/intelsat37e_qpsk_excerpt.sigmf, the real recording that
comes with the app.

The file holds samples 204,000 to 234,719 (30 windows of 1,024 samples) of
Zenodo record 13371136, "Test of gr4-packet-modem through an Intelsat 37e
C-band transponder" by Daniel Estévez (CC BY 4.0,
https://doi.org/10.5281/zenodo.13371136). The excerpt starts and ends in a gap
between packets and holds ten complete packets.

What it does:
  1. Runs fetch_real_iq.py, which downloads the record, checks it against the
     published checksums, and keeps those samples as cf32_le, unchanged.
  2. Finds the packets with a smoothed-power detector and marks each one as a
     SigMF annotation.
  3. Writes the SigMF archive with fixed dates, so running this again with the
     same download gives the same bytes.

Examples:
  python tools/make_real_example.py                      # replace the bundled file
  python tools/make_real_example.py --out /tmp/check.sigmf
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np

# Make fetch_real_iq and the repository's modules importable from any folder.
TOOLS = Path(__file__).resolve().parent
for folder in (TOOLS, TOOLS.parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import fetch_real_iq
from real_example import EXAMPLE_PATH, FIRST_SAMPLE, LAST_SAMPLE, SAMPLE_COUNT


# Samples in the moving average of power.
SMOOTHING = 64
# Shorter stretches above the threshold are noise, not packets.
MIN_PACKET_SAMPLES = 500
PACKET_COMMENT = (
    "A packet: a 64-symbol BPSK syncword, then QPSK. Its edges are estimated from "
    "signal power."
)


def loud_stretches(samples: np.ndarray, smoothing: int = SMOOTHING) -> list[tuple[int, int]]:
    """(first sample, one past the last sample) of each loud stretch.

    Power is averaged over `smoothing` samples centered on each sample. A
    sample is loud when that average is above the midpoint of the quiet (10th
    percentile) and loud (90th percentile) levels. Dips shorter than twice
    `smoothing` are closed. Stretches cut off by either end are included.
    """
    power = np.abs(np.asarray(samples, dtype=np.complex128)) ** 2
    kernel = np.ones(smoothing)
    # Dividing by the number of samples averaged keeps the ends unbiased.
    smoothed = np.convolve(power, kernel, "same") / np.convolve(
        np.ones(len(power)), kernel, "same"
    )
    quiet, loud = np.percentile(smoothed, [10, 90])
    edges = np.diff((smoothed > (quiet + loud) / 2).astype(np.int8), prepend=0, append=0)
    spans: list[list[int]] = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if spans and start - spans[-1][1] < 2 * smoothing:
            spans[-1][1] = int(stop)
        else:
            spans.append([int(start), int(stop)])
    return [(start, stop) for start, stop in spans]


def find_packets(
    samples: np.ndarray, smoothing: int = SMOOTHING, min_length: int = MIN_PACKET_SAMPLES
) -> list[tuple[int, int]]:
    """(first sample, one past the last sample) of each complete packet: each
    loud stretch (see loud_stretches) of at least `min_length` samples, except
    those that may be cut off because they come within `smoothing` samples of
    either end.
    """
    count = len(samples)
    return [
        (start, stop)
        for start, stop in loud_stretches(samples, smoothing)
        if stop - start >= min_length and smoothing <= start and stop <= count - smoothing
    ]


def packet_levels(samples: np.ndarray, packets: list[tuple[int, int]]) -> float:
    """How much stronger the packets are than the gaps, in dB."""
    inside = np.zeros(len(samples), dtype=bool)
    for start, stop in packets:
        inside[start:stop] = True
    power = np.abs(np.asarray(samples, dtype=np.complex128)) ** 2
    return float(10 * np.log10(power[inside].mean() / power[~inside].mean()))


def example_metadata(fetched: dict, packets: list[tuple[int, int]]) -> dict:
    """fetch_real_iq's metadata for the excerpt, with a fuller description and
    one annotation per packet."""
    section = dict(fetched["global"])
    section["core:description"] += (
        " The sample values are unmodified. Each annotation marks a packet found "
        "from signal power by SpectrumSense's make_real_example.py (an estimate, "
        "not the author's labels)."
    )
    section["core:recorder"] = "make_real_example.py and fetch_real_iq.py (SpectrumSense)"
    annotations = [
        {
            "core:sample_start": start,
            "core:sample_count": stop - start,
            "core:label": "packet",
            "core:comment": PACKET_COMMENT,
            "core:generator": "make_real_example.py (SpectrumSense)",
        }
        for start, stop in packets
    ]
    return {"global": section, "captures": fetched["captures"], "annotations": annotations}


def capture_time(metadata: dict) -> int:
    """The excerpt's first-sample time in seconds since 1970, used to date the
    archive's entries; 0 if the metadata gives no whole-second time."""
    moment = metadata["captures"][0].get("core:datetime", "")
    try:
        parsed = datetime.datetime.strptime(moment, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return 0
    return int(parsed.replace(tzinfo=datetime.timezone.utc).timestamp())


def read_archive(path: Path) -> tuple[dict, bytes]:
    """The metadata and data of a SigMF archive that fetch_real_iq wrote."""
    with tarfile.open(path, mode="r:") as archive:
        members = {member.name.rpartition(".")[2]: member for member in archive.getmembers()}
        meta = json.load(archive.extractfile(members["sigmf-meta"]))
        data = archive.extractfile(members["sigmf-data"]).read()
    return meta, data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and rebuild the real recording bundled with SpectrumSense."
    )
    parser.add_argument("--out", type=Path, default=EXAMPLE_PATH,
                        help=f"where to write the archive (default {EXAMPLE_PATH})")
    return parser.parse_args(argv)


def build(out: Path) -> None:
    with tempfile.TemporaryDirectory() as folder:
        fetch_args = fetch_real_iq.parse_args(
            ["--start", str(FIRST_SAMPLE), "--samples", str(SAMPLE_COUNT), "--out-dir", folder]
        )
        fetched, data = read_archive(fetch_real_iq.fetch(fetch_args)[0])
    samples = np.frombuffer(data, dtype="<c8")
    if len(samples) != SAMPLE_COUNT:
        raise fetch_real_iq.FetchError(
            f"Expected {SAMPLE_COUNT:,} samples but received {len(samples):,}."
        )
    packets = find_packets(samples)
    if not packets:
        raise fetch_real_iq.FetchError("No packets were found in these samples.")
    metadata = example_metadata(fetched, packets)
    name = out.name.removesuffix(".sigmf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fetch_real_iq.write_sigmf_archive(out, name, metadata, data, mtime=capture_time(metadata))

    lengths = [stop - start for start, stop in packets]
    periods = np.diff([start for start, _ in packets])
    gaps = [start - stop for (_, stop), (start, _) in zip(packets, packets[1:])]
    print(f"\nWrote {out}: samples {FIRST_SAMPLE:,} to {LAST_SAMPLE:,}, "
          f"{out.stat().st_size:,} bytes")
    print(f"  {len(packets)} complete packets; first starts at {packets[0][0]:,}, last ends at "
          f"{packets[-1][1]:,} of {len(samples):,} (counting from the excerpt's start)")
    print(f"  median packet {np.median(lengths):,.0f} samples, median start-to-start "
          f"{np.median(periods):,.0f}, median gap {np.median(gaps):,.0f}")
    print(f"  packets are {packet_levels(samples, packets):.1f} dB stronger than the gaps")
    print(f"  SHA-256 of the archive: {hashlib.sha256(out.read_bytes()).hexdigest()}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build(args.out)
    except fetch_real_iq.FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write the archive: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
