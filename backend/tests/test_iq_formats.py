"""Regression tests for reading raw SDR, NumPy, WAV, and SigMF recordings."""

import gzip
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

import iq_formats
from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, from_csv, to_csv
from iq_formats import (
    MAX_FILE_BYTES,
    NPY_MAX_BYTES,
    SIGMF_ARCHIVE_MAX_BYTES,
    SIGMF_META_MAX_BYTES,
    SUPPORTED_EXTENSIONS,
    WAV_MAX_BYTES,
    load_recording,
    load_sigmf_pair,
    load_uploads,
    select_uploads,
)
from signal_config import SAMPLES_PER_WINDOW


# Extension: NumPy type of one I or Q component.
RAW_COMPONENTS = {
    "cf32": "<f4",
    "fc32": "<f4",
    "cfile": "<f4",
    "cs16": "<i2",
    "sc16": "<i2",
    "cs8": "i1",
    "cu8": "u1",
}
SIGMF_COMPONENTS = {"cf32_le": "<f4", "ci16_le": "<i2", "ci8": "i1", "cu8": "u1"}
# Text a shared file could carry to make the app's error box show a link.
HOSTILE_TEXT = "![](https://e.example/p.png)[Get the fixed file](https://e.example)"
UNREADABLE = "could not be read as an uncompressed tar archive"

WAV_PCM, WAV_FLOAT, WAV_EXTENSIBLE = 0x0001, 0x0003, 0xFFFE
# (WAV format code, bits): NumPy type of one I or Q value ("<i4" holds 24-bit
# values before packing), values that reach both ends of the range, and the
# offset and full scale that map them to about [-1, 1].
WAV_SAMPLE_TYPES = {
    (WAV_PCM, 8): ("u1", [0, 255, 128, 127, 1, 200], 128, 128),
    (WAV_PCM, 16): ("<i2", [-32768, 32767, 0, 1, -1, 12345], 0, 2**15),
    (WAV_PCM, 24): ("<i4", [-(2**23), 2**23 - 1, 0, 1, -1, 1234567], 0, 2**23),
    (WAV_PCM, 32): ("<i4", [-(2**31), 2**31 - 1, 0, 1, -1, 123456789], 0, 2**31),
    (WAV_FLOAT, 32): ("<f4", [-1.5, 1.0, 0.0, 0.25, -0.125, 3.0e38], 0, 1),
    (WAV_FLOAT, 64): ("<f8", [-1.5, 1.0, 0.0, 0.25, -0.125, 1.0e300], 0, 1),
}
# The fixed part of every WAVE_FORMAT_EXTENSIBLE sub-format GUID.
WAV_GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")
# A SpectraVue-style "auxi" start or stop time: 2026-09-25 (a Friday) 16:45:56.250.
AUXI_TIME = (2026, 9, 5, 25, 16, 45, 56, 250)


def interleaved(components, dtype: str, samples: int = SAMPLES_PER_WINDOW) -> bytes:
    """Repeat I,Q component values to fill the given number of samples."""
    return np.resize(np.asarray(components, dtype=dtype), 2 * samples).tobytes()


def as_complex(components, samples: int = SAMPLES_PER_WINDOW) -> np.ndarray:
    values = np.resize(np.asarray(components, dtype=np.float64), 2 * samples)
    return values[0::2] + 1j * values[1::2]


def npy(array: np.ndarray, allow_pickle: bool = False) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


def random_complex(count: int = SAMPLES_PER_WINDOW, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(count) + 1j * rng.standard_normal(count)


def sigmf_meta(
    datatype: str = "cf32_le",
    sample_rate: float | None = None,
    frequency: float | None = None,
    **global_fields,
) -> bytes:
    section = {"core:datatype": datatype, "core:version": "1.2.0", **global_fields}
    if sample_rate is not None:
        section["core:sample_rate"] = sample_rate
    capture = {"core:sample_start": 0}
    if frequency is not None:
        capture["core:frequency"] = frequency
    return json.dumps({"global": section, "captures": [capture], "annotations": []}).encode()


def regular(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    return info, payload


def special(name: str, kind: bytes, linkname: str = "") -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.mode = 0o755
    return info, None


def tar_bytes(entries) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for info, payload in entries:
            archive.addfile(info, None if payload is None else BytesIO(payload))
    return buffer.getvalue()


def fix_checksum(raw: bytearray, header: int) -> None:
    """Recompute the checksum of the tar header at this byte offset."""
    raw[header + 148 : header + 156] = b" " * 8
    raw[header + 148 : header + 156] = b"%06o\0 " % sum(raw[header : header + 512])


def set_gnu_size(archive: bytes, header: int, size: int) -> bytes:
    """Give the header at this byte offset a GNU base-256 size field (no PAX
    record needed) and a valid checksum."""
    raw = bytearray(archive)
    raw[header + 124 : header + 136] = bytes([0x80]) + size.to_bytes(11, "big")
    fix_checksum(raw, header)
    return bytes(raw)


def assert_plain_message(test: unittest.TestCase, error: BaseException) -> str:
    """The message of a ValueError, checked to repeat no hostile text and to be
    sendable to a browser as UTF-8."""
    test.assertIsInstance(error, ValueError)
    message = str(error)
    message.encode("utf-8")  # A lone surrogate would raise here.
    for fragment in ("e.example", "](", "\ud800", "\udcff"):
        test.assertNotIn(fragment, message)
    return message


def sigmf_archive(meta: bytes, data: bytes, name: str = "recording", extra=()) -> bytes:
    """Laid out like a sigmf-python archive: one directory holding both files."""
    return tar_bytes(
        [
            special(name, tarfile.DIRTYPE),
            regular(f"{name}/{name}.sigmf-meta", meta),
            regular(f"{name}/{name}.sigmf-data", data),
            *extra,
        ]
    )


def wav_chunk(chunk_id: bytes, contents: bytes, pad: bool = True) -> bytes:
    """One RIFF chunk; odd-sized contents are followed by a padding byte."""
    padding = b"\0" * (len(contents) % 2) if pad else b""
    return struct.pack("<4sI", chunk_id, len(contents)) + contents + padding


def wav_format(
    code: int = WAV_PCM,
    bits: int = 16,
    channels: int = 2,
    rate: int = 12_000,
    extensible: bool = False,
    block: int | None = None,
) -> bytes:
    """A "fmt " chunk. An extensible one names the format code in its GUID."""
    block = channels * bits // 8 if block is None else block
    tag = WAV_EXTENSIBLE if extensible else code
    contents = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    if extensible:
        # Extension size, valid bits, channel mask (front left and right), GUID.
        contents += struct.pack("<HHIH", 22, bits, 0x3, code) + WAV_GUID_TAIL
    elif code != WAV_PCM:
        contents += struct.pack("<H", 0)  # Formats other than PCM state an empty extension.
    return wav_chunk(b"fmt ", contents)


def wav_file(*chunks: bytes, form: bytes = b"RIFF", riff_size: int | None = None) -> bytes:
    """A RIFF WAVE file holding these chunks; the RIFF size is correct by default."""
    body = b"WAVE" + b"".join(chunks)
    return struct.pack("<4sI", form, len(body) if riff_size is None else riff_size) + body


def wav_values(values, code: int = WAV_PCM, bits: int = 16, samples: int = SAMPLES_PER_WINDOW) -> bytes:
    """Interleaved I,Q values of a WAV sample type, repeated to fill the samples."""
    dtype = WAV_SAMPLE_TYPES[(code, bits)][0]
    array = np.resize(np.asarray(values, dtype=dtype), 2 * samples)
    if (code, bits) == (WAV_PCM, 24):
        return array.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()  # Low three bytes of each.
    return array.tobytes()


def wav_expected(values, code: int = WAV_PCM, bits: int = 16, samples: int = SAMPLES_PER_WINDOW):
    dtype, _, offset, full_scale = WAV_SAMPLE_TYPES[(code, bits)]
    stored = np.asarray(values, dtype=dtype).astype(np.float64)
    return as_complex((stored - offset) / full_scale, samples)


def pcm16_wav(*extra: bytes, values=(1000, -2000), samples: int = SAMPLES_PER_WINDOW) -> bytes:
    """A 16-bit PCM I/Q file, with any extra chunks between the format and the data."""
    return wav_file(wav_format(), *extra, wav_chunk(b"data", wav_values(values, samples=samples)))


def auxi_chunk(frequency: int, time=AUXI_TIME, size: int = 164) -> bytes:
    """An "auxi" chunk in SpectraVue's layout, as SDRangel writes it: start and
    stop times, center frequency, eight more 32-bit fields, and a 96-byte
    next-file name."""
    contents = struct.pack("<8H8H9I96s", *time, *time, frequency, 12_000, *[0] * 7, b"")
    return wav_chunk(b"auxi", contents[:size])


def kiwirecorder_wav(blocks, sample_rate: float, kiwi_wav: bool) -> bytes:
    """A file laid out as kiwirecorder.py (kiwiclient) writes it: the header
    from its _write_wav_header, then each block of int16 I,Q values. With
    --kiwi-wav, each block is a data chunk after a "kiwi" GNSS time stamp."""
    body = b""
    for number, block in enumerate(blocks):
        samples = np.asarray(block, dtype="<i2")
        if kiwi_wav:
            body += struct.pack("<4sIBBII", b"kiwi", 10, 255, 0, 1_000 + number, 0)
            body += struct.pack("<4sI", b"data", samples.nbytes)
        body += samples.tobytes()
    rate = int(sample_rate + 0.5)
    filesize = 12 + 24 + (0 if kiwi_wav else 8) + len(body)
    header = struct.pack("<4sI4s", b"RIFF", filesize - 8, b"WAVE")
    header += struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, rate, rate * 4, 4, 16)
    if not kiwi_wav:
        header += struct.pack("<4sI", b"data", filesize - 12 - 8 - 16 - 8)
    return header + body


unpickled = []


def record_unpickling():
    unpickled.append(True)


class Tripwire:
    """Records any attempt to unpickle it."""

    def __reduce__(self):
        return record_unpickling, ()


class RawFormatTests(unittest.TestCase):
    def test_complex_float32_round_trip_is_exact(self):
        samples = random_complex().astype(np.complex64)
        for extension in ("cf32", "fc32", "cfile"):
            with self.subTest(extension=extension):
                recording = load_recording(f"capture.{extension}", samples.tobytes())
                self.assertEqual(recording.samples.dtype, np.dtype(np.complex128))
                np.testing.assert_array_equal(recording.samples, samples.astype(np.complex128))
                self.assertIsNone(recording.sample_rate)
                self.assertIsNone(recording.center_frequency)
                self.assertIn(f".{extension}", recording.source_format)

    def test_integer_formats_scale_known_values(self):
        cases = {
            "cs16": ([-32768, 32767, 0, 16384, -16384, 1], 32768, 0),
            "sc16": ([-32768, 32767, 0, 16384, -16384, 1], 32768, 0),
            "cs8": ([-128, 127, 0, 64, -64, 1], 128, 0),
            "cu8": ([0, 255, 127, 128, 64, 191], 127.5, 127.5),
        }
        for extension, (components, full_scale, offset) in cases.items():
            with self.subTest(extension=extension):
                data = interleaved(components, RAW_COMPONENTS[extension])
                expected = as_complex((np.array(components) - offset) / full_scale)
                recording = load_recording(f"capture.{extension}", data)
                np.testing.assert_array_equal(recording.samples, expected)

    def test_integer_formats_map_every_code_into_the_unit_range(self):
        for extension, codes, low, high in (
            ("cs16", np.arange(-32768, 32768), -1.0, 32767 / 32768),
            ("cs8", np.arange(-128, 128), -1.0, 127 / 128),
            ("cu8", np.arange(256), -1.0, 1.0),
        ):
            with self.subTest(extension=extension):
                count = max(len(codes), SAMPLES_PER_WINDOW)
                data = interleaved(codes, RAW_COMPONENTS[extension], samples=count)
                samples = load_recording(f"capture.{extension}", data).samples
                components = np.concatenate([samples.real, samples.imag])
                self.assertEqual(components.min(), low)
                self.assertEqual(components.max(), high)
                self.assertEqual(len(np.unique(components)), len(codes))

    def test_multibyte_formats_are_little_endian_on_every_platform(self):
        cs16 = (b"\x00\x80" + b"\xff\x7f") * SAMPLES_PER_WINDOW
        np.testing.assert_array_equal(
            load_recording("x.cs16", cs16).samples,
            np.full(SAMPLES_PER_WINDOW, -1 + 1j * 32767 / 32768),
        )
        cf32 = (b"\x00\x00\x80\x3f" + b"\x00\x00\x00\xc0") * SAMPLES_PER_WINDOW
        np.testing.assert_array_equal(
            load_recording("x.cf32", cf32).samples, np.full(SAMPLES_PER_WINDOW, 1 - 2j)
        )

    def test_extensions_are_case_insensitive(self):
        data = interleaved([1, 2], "<i2")
        for name in ("REC.CS16", "Rec.Sc16", "capture.CS16"):
            with self.subTest(name=name):
                self.assertEqual(len(load_recording(name, data).samples), SAMPLES_PER_WINDOW)
        self.assertEqual(
            len(load_recording("GNURADIO.CFILE", random_complex().astype(np.complex64).tobytes()).samples),
            SAMPLES_PER_WINDOW,
        )

    def test_truncated_data_is_rejected(self):
        for extension, component in RAW_COMPONENTS.items():
            data = interleaved([1, 2], component, samples=SAMPLES_PER_WINDOW + 1)
            width = 2 * np.dtype(component).itemsize
            for cut in range(1, width):
                with self.subTest(extension=extension, cut=cut):
                    with self.assertRaisesRegex(ValueError, "whole number .*truncated"):
                        load_recording(f"x.{extension}", data[:-cut])

    def test_sample_count_limits(self):
        for extension, component in RAW_COMPONENTS.items():
            with self.subTest(extension=extension):
                width = 2 * np.dtype(component).itemsize
                for count in (SAMPLES_PER_WINDOW, MAX_SAMPLES):
                    recording = load_recording(f"x.{extension}", interleaved([1, 2], component, count))
                    self.assertEqual(len(recording.samples), count)
                with self.assertRaisesRegex(ValueError, "at least 1,024"):
                    load_recording(f"x.{extension}", interleaved([1, 2], component, SAMPLES_PER_WINDOW - 1))
                with self.assertRaisesRegex(ValueError, "more than 100,000 samples") as caught:
                    load_recording(f"x.{extension}", interleaved([1, 2], component, MAX_SAMPLES + 1))
                self.assertIn(f"{MAX_SAMPLES * width:,} bytes", str(caught.exception))
                with self.assertRaisesRegex(ValueError, "at least"):
                    load_recording(f"x.{extension}", b"")

    def test_nonfinite_float_values_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for components in ([value, 0.0], [0.0, value]):
                with self.subTest(components=components):
                    data = np.resize(np.array([1.0, 2.0], dtype="<f4"), 2 * SAMPLES_PER_WINDOW)
                    data[2 * 100 : 2 * 100 + 2] = components
                    with self.assertRaisesRegex(ValueError, "not finite"):
                        load_recording("x.cf32", data.tobytes())

    def test_largest_float32_values_are_accepted(self):
        largest = np.finfo(np.float32).max
        data = interleaved([largest, -largest], "<f4")
        samples = load_recording("x.cf32", data).samples
        np.testing.assert_array_equal(samples, np.full(SAMPLES_PER_WINDOW, complex(largest, -largest)))


class NpyTests(unittest.TestCase):
    def test_complex_arrays_round_trip_exactly(self):
        samples = random_complex()
        for dtype in (np.complex64, np.complex128, ">c16", ">c8"):
            with self.subTest(dtype=dtype):
                array = samples.astype(dtype)
                recording = load_recording("x.npy", npy(array))
                self.assertEqual(recording.samples.dtype, np.dtype(np.complex128))
                np.testing.assert_array_equal(recording.samples, array.astype(np.complex128))
                self.assertIn(np.dtype(dtype).name, recording.source_format)

    def test_two_column_real_arrays_hold_i_and_q(self):
        samples = random_complex()
        pairs = np.stack([samples.real, samples.imag], axis=1)
        for array in (
            pairs,
            pairs.astype(np.float32),
            np.asfortranarray(pairs),
            np.round(pairs * 1000).astype(np.int16),
            np.round(pairs * 10 + 100).astype(np.uint8),
        ):
            with self.subTest(dtype=array.dtype, fortran=array.flags.f_contiguous):
                recording = load_recording("x.NPY", npy(array))
                expected = array[:, 0].astype(np.float64) + 1j * array[:, 1].astype(np.float64)
                np.testing.assert_array_equal(recording.samples, expected)
                self.assertIn("I and Q columns", recording.source_format)

    def test_wrong_shapes_and_dtypes_are_rejected(self):
        n = SAMPLES_PER_WINDOW
        arrays = [
            np.zeros(n),
            np.zeros((n, 3)),
            np.zeros((2, n)),
            np.zeros((n, 2), dtype=np.complex128),
            np.zeros((n, 1), dtype=np.complex128),
            np.zeros((), dtype=np.complex128),
            np.zeros((n, 2, 1)),
            np.zeros((n, 2), dtype=bool),
            np.zeros(n, dtype="U3"),
            np.zeros((n, 2), dtype="datetime64[s]"),
            np.zeros(n, dtype=[("i", "<f8"), ("q", "<f8")]),
        ]
        if np.dtype(np.longdouble).itemsize > 8:
            arrays.append(np.zeros((n, 2), dtype=np.longdouble))
            arrays.append(np.zeros(n, dtype=np.clongdouble))
        for array in arrays:
            with self.subTest(dtype=array.dtype, shape=array.shape):
                with self.assertRaisesRegex(ValueError, "holds a .* array with shape"):
                    load_recording("x.npy", npy(array))

    def test_structured_field_names_are_not_repeated(self):
        array = np.zeros(SAMPLES_PER_WINDOW, dtype=[(HOSTILE_TEXT, "<f4"), ("q", "<f4")])
        with self.assertRaises(ValueError) as caught:
            load_recording("x.npy", npy(array))
        message = assert_plain_message(self, caught.exception)
        self.assertIn("holds a structured array with shape (1024,)", message)

    def test_object_arrays_are_rejected_without_unpickling(self):
        unpickled.clear()
        array = np.array([Tripwire()] * SAMPLES_PER_WINDOW, dtype=object)
        data = npy(array, allow_pickle=True)
        with patch("numpy.load", wraps=np.load) as loader:
            with self.assertRaisesRegex(ValueError, "Python objects"):
                load_recording("x.npy", data)
        loader.assert_not_called()
        self.assertEqual(unpickled, [])

    def test_header_describing_a_huge_array_is_rejected_before_loading(self):
        for shape, message in (((10**12,), "more than 100,000"), ((2_000,), "truncated")):
            with self.subTest(shape=shape):
                buffer = BytesIO()
                np.lib.format.write_array_header_1_0(
                    buffer, {"descr": "<c16", "fortran_order": False, "shape": shape}
                )
                data = buffer.getvalue() + bytes(16 * SAMPLES_PER_WINDOW)
                with patch("numpy.load", wraps=np.load) as loader:
                    with self.assertRaisesRegex(ValueError, message):
                        load_recording("x.npy", data)
                loader.assert_not_called()

    def test_truncated_or_padded_files_are_rejected(self):
        data = npy(random_complex())
        for damaged in (data[:-1], data[:-16], data + b"\0", data + bytes(16)):
            with self.subTest(size=len(damaged)):
                with self.assertRaisesRegex(ValueError, "truncated or has extra bytes"):
                    load_recording("x.npy", damaged)

    def test_files_that_are_not_npy_are_rejected(self):
        data = npy(random_complex())
        for damaged in (b"", b"hello", b"PK\x03\x04" + data, data[:8], data[:60]):
            with self.subTest(prefix=damaged[:12]):
                with self.assertRaisesRegex(ValueError, "not a readable NumPy"):
                    load_recording("x.npy", damaged)

    def test_sample_count_limits(self):
        self.assertEqual(len(load_recording("x.npy", npy(random_complex(MAX_SAMPLES))).samples), MAX_SAMPLES)
        self.assertLessEqual(len(npy(random_complex(MAX_SAMPLES))), NPY_MAX_BYTES)
        with self.assertRaisesRegex(ValueError, "more than 100,000"):
            load_recording("x.npy", npy(random_complex(MAX_SAMPLES + 1).astype(np.complex64)))
        with self.assertRaisesRegex(ValueError, "at least 1,024"):
            load_recording("x.npy", npy(random_complex(SAMPLES_PER_WINDOW - 1)))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_recording("x.npy", bytes(NPY_MAX_BYTES + 1))

    def test_nonfinite_values_are_rejected(self):
        for value in (complex(np.nan, 0), complex(0, np.inf)):
            samples = random_complex()
            samples[5] = value
            for array in (samples, np.stack([samples.real, samples.imag], axis=1)):
                with self.subTest(value=value, ndim=array.ndim):
                    with self.assertRaisesRegex(ValueError, "not finite"):
                        load_recording("x.npy", npy(array))


class SigmfArchiveTests(unittest.TestCase):
    def test_each_datatype_matches_the_raw_reader(self):
        raw_extension = {"cf32_le": "cf32", "ci16_le": "cs16", "ci8": "cs8", "cu8": "cu8"}
        for datatype, component in SIGMF_COMPONENTS.items():
            with self.subTest(datatype=datatype):
                values = [0, 255, 127, 128, 60, 200] if component == "u1" else [-100, 100, 0, 27, 55, -3]
                data = interleaved(values, component)
                meta = sigmf_meta(datatype, sample_rate=2.4e6, frequency=433.92e6)
                recording = load_recording("capture.sigmf", sigmf_archive(meta, data))
                expected = load_recording(f"x.{raw_extension[datatype]}", data).samples
                np.testing.assert_array_equal(recording.samples, expected)
                self.assertEqual(recording.sample_rate, 2.4e6)
                self.assertEqual(recording.center_frequency, 433.92e6)
                self.assertEqual(recording.source_format, f"SigMF archive ({datatype})")

    def test_flat_archive_without_directory_is_accepted(self):
        samples = random_complex().astype(np.complex64)
        archive = tar_bytes(
            [regular("a.sigmf-meta", sigmf_meta()), regular("a.sigmf-data", samples.tobytes())]
        )
        recording = load_recording("A.SIGMF", archive)
        np.testing.assert_array_equal(recording.samples, samples.astype(np.complex128))

    def test_optional_metadata_may_be_missing(self):
        data = random_complex().astype(np.complex64).tobytes()
        without_captures = json.dumps({"global": {"core:datatype": "cf32_le"}}).encode()
        for meta in (sigmf_meta(), without_captures):
            with self.subTest(meta=meta[:40]):
                recording = load_recording("x.sigmf", sigmf_archive(meta, data))
                self.assertIsNone(recording.sample_rate)
                self.assertIsNone(recording.center_frequency)

    def test_unsupported_datatypes_are_named(self):
        data = interleaved([1, 2], "<f4")
        for datatype in ("rf32_le", "cf32_be", "ci16_be", "cu16_le", "cf64_le", "ri8"):
            with self.subTest(datatype=datatype):
                with self.assertRaisesRegex(ValueError, f'datatype "{datatype}" is not supported'):
                    load_recording("x.sigmf", sigmf_archive(sigmf_meta(datatype), data))

    def test_malformed_datatypes_are_described_without_repeating_them(self):
        # JSON's "\ud800" decodes to a lone surrogate, which cannot be sent to
        # a browser, and Markdown would turn a link into a clickable one.
        data = interleaved([1, 2], "<f4")
        for datatype in ("mystery", "\ud800", HOSTILE_TEXT, "CF32_LE", "cf32_le ", "_cf32_le_"):
            with self.subTest(datatype=datatype):
                for load in (
                    lambda meta: load_recording("x.sigmf", sigmf_archive(meta, data)),
                    lambda meta: load_uploads([("x.sigmf-meta", meta), ("x.sigmf-data", data)]),
                ):
                    with self.assertRaises(ValueError) as caught:
                        load(sigmf_meta(datatype))
                    message = assert_plain_message(self, caught.exception)
                    self.assertIn("is not a valid SigMF datatype", message)
                    self.assertNotIn(datatype, message)

    def test_invalid_metadata_is_explained(self):
        data = interleaved([1, 2], "<f4")
        for meta, message in (
            (b"{not json", "not valid UTF-8 JSON"),
            (b"\xff\xfe{}", "not valid UTF-8 JSON"),
            (b"[" * 100_000, "not valid UTF-8 JSON"),
            (b"[]", 'no "global" section'),
            (b'{"global": 1}', 'no "global" section'),
            (b'{"global": {}}', '"core:datatype"'),
            (sigmf_meta(**{"core:num_channels": 2}), "exactly one channel"),
            (sigmf_meta(**{"core:num_channels": "1"}), "exactly one channel"),
            (sigmf_meta(sample_rate=-1.0), "positive"),
            (sigmf_meta(sample_rate=0), "positive"),
            (sigmf_meta(sample_rate="fast"), "finite number"),
            (sigmf_meta(sample_rate=True), "finite number"),
            (sigmf_meta(sample_rate=float("nan")), "finite number"),
            (sigmf_meta(sample_rate=10**400), "finite number"),
            (sigmf_meta(frequency="433 MHz"), "finite number"),
        ):
            with self.subTest(meta=meta[:60]):
                with self.assertRaisesRegex(ValueError, message):
                    load_recording("x.sigmf", sigmf_archive(meta, data))

    def test_missing_or_extra_recordings_are_rejected(self):
        meta, data = sigmf_meta(), interleaved([1, 2], "<f4")
        for entries, message in (
            ([regular("a.sigmf-data", data)], "no .sigmf-meta"),
            ([regular("a.sigmf-meta", meta)], "no .sigmf-data"),
            ([regular("a.sigmf-meta", meta), regular("b.sigmf-data", data)], "different names"),
            (
                [regular(f"{name}.sigmf-{kind}", meta if kind == "meta" else data)
                 for name in ("a", "b") for kind in ("meta", "data")],
                "more than one recording",
            ),
            ([regular("notes.txt", b"hello")], "no .sigmf-meta"),
            ([], "no .sigmf-meta"),
        ):
            with self.subTest(names=[info.name for info, _ in entries]):
                with self.assertRaisesRegex(ValueError, message):
                    load_recording("x.sigmf", tar_bytes(entries))

    def test_unsafe_member_paths_are_rejected(self):
        meta, data = sigmf_meta(), interleaved([1, 2], "<f4")
        for name in ("../escape.txt", "/etc/passwd", "rec/../../x", "..", "C:/x", "..\\x"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "unsafe path"):
                    load_recording("x.sigmf", sigmf_archive(meta, data, extra=[regular(name, b"x")]))
        traversing = tar_bytes(
            [regular("../x.sigmf-meta", meta), regular("../x.sigmf-data", data)]
        )
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            load_recording("x.sigmf", traversing)

    def test_member_names_are_never_repeated(self):
        meta, data = sigmf_meta(), interleaved([1, 2], "<f4")
        link = special(HOSTILE_TEXT, tarfile.SYMTYPE, "recording/recording.sigmf-meta")
        absolute = regular("/" + HOSTILE_TEXT, b"x")
        # tarfile reads a name that is not UTF-8 with lone surrogates in it.
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
            archive.addfile(regular("r.sigmf-meta", meta)[0], BytesIO(meta))
            archive.addfile(special("r\udcff.sigmf-data", tarfile.SYMTYPE, "r.sigmf-meta")[0])
        self.assertIn(b"r\xff.sigmf-data", buffer.getvalue())
        for archive, message in (
            (sigmf_archive(meta, data, extra=[link]), "link or special file"),
            (sigmf_archive(meta, data, extra=[absolute]), "unsafe path"),
            (buffer.getvalue(), "link or special file"),
        ):
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    load_recording("x.sigmf", archive)
                self.assertIn(message, assert_plain_message(self, caught.exception))

    def test_macos_metadata_entries_are_ignored(self):
        # macOS tar adds an AppleDouble "._" file for each file with extended
        # attributes, such as a download's quarantine flag.
        samples = random_complex().astype(np.complex64)
        meta = sigmf_meta(sample_rate=1e6)
        apple_double = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + bytes(40)
        archive = tar_bytes(
            [
                special("rec", tarfile.DIRTYPE),
                regular("rec/._rec.sigmf-meta", apple_double),
                regular("rec/rec.sigmf-meta", meta),
                regular("rec/._rec.sigmf-data", apple_double),
                regular("rec/rec.sigmf-data", samples.tobytes()),
                regular("__MACOSX/rec/._rec.sigmf-data", apple_double),
            ]
        )
        recording = load_recording("rec.sigmf", archive)
        np.testing.assert_array_equal(recording.samples, samples.astype(np.complex128))
        self.assertEqual(recording.sample_rate, 1e6)
        # Finder information alone is not a recording.
        only_finder_info = tar_bytes(
            [regular("rec/._rec.sigmf-meta", apple_double), regular("rec/rec.sigmf-data", samples.tobytes())]
        )
        with self.assertRaisesRegex(ValueError, "no .sigmf-meta"):
            load_recording("rec.sigmf", only_finder_info)

    def test_hostile_headers_give_one_plain_error(self):
        meta, data = sigmf_meta(), interleaved([1, 2], "<f4")

        def pax_archive(target: str, pax_headers: dict) -> bytes:
            entries = [regular("r.sigmf-meta", meta), regular("r.sigmf-data", data)]
            for info, _ in entries:
                if info.name.endswith(target):
                    info.pax_headers = pax_headers
            return tar_bytes(entries)

        plain = tar_bytes([regular("r.sigmf-meta", meta), regular("r.sigmf-data", data)])
        # The data member's header follows the metadata header and its block.
        data_header = 512 + -(-len(meta) // 512) * 512
        cases = {
            # A size too large for a C integer makes tarfile raise OverflowError.
            "PAX size 2**63": pax_archive("data", {"size": str(2**63)}),
            "PAX size 2**63 - 1": pax_archive("data", {"size": str(2**63 - 1)}),
            "PAX size 2**64 on the metadata": pax_archive("meta", {"size": str(2**64)}),
            "GNU base-256 size": set_gnu_size(plain, data_header, 2**70),
            # tarfile's own message would repeat the PAX value.
            "bad sparse map": pax_archive(
                "meta", {"GNU.sparse.map": f"**Damaged.** {HOSTILE_TEXT}", "GNU.sparse.size": "8"}
            ),
            "bad sparse count": pax_archive(
                "meta", {"GNU.sparse.map": "0.0", "GNU.sparse.size": "8"}
            ),
        }
        for name, archive in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError) as caught:
                    load_recording("x.sigmf", archive)
                message = assert_plain_message(self, caught.exception)
                self.assertRegex(message, f"{UNREADABLE}|truncated or damaged")

    def test_damaged_headers_only_ever_raise_value_errors(self):
        # Change random header bytes, sometimes turning a size field into a
        # huge or negative GNU base-256 number (first byte 0x80 or 0xFF), and
        # fix the checksums so that tarfile parses the damaged fields. Each
        # archive either loads or fails with a plain ValueError.
        meta = sigmf_meta(sample_rate=1e6)
        archive = sigmf_archive(meta, interleaved([1, 2], "<f4"))
        headers = [0, 512, 1024 + -(-len(meta) // 512) * 512]
        rng = np.random.default_rng(0)
        for trial in range(400):
            damaged = bytearray(archive)
            for _ in range(int(rng.integers(1, 6))):
                header = int(rng.choice(headers))
                if rng.random() < 0.25:
                    damaged[header + 124] = int(rng.choice([0x80, 0xFF]))
                else:
                    damaged[header + int(rng.integers(0, 512))] = int(rng.integers(0, 256))
                fix_checksum(damaged, header)
            try:
                load_recording("x.sigmf", bytes(damaged))
            except ValueError as error:
                assert_plain_message(self, error)
            except Exception as error:  # noqa: BLE001 - any other type is the bug.
                self.fail(f"trial {trial}: {type(error).__name__}: {error!r}")

    def test_links_and_special_files_are_rejected(self):
        meta = sigmf_meta()
        for kind, linkname in (
            (tarfile.SYMTYPE, "/etc/passwd"),
            (tarfile.LNKTYPE, "recording/recording.sigmf-meta"),
            (tarfile.FIFOTYPE, ""),
            (tarfile.CHRTYPE, ""),
        ):
            with self.subTest(kind=kind):
                archive = tar_bytes(
                    [
                        regular("recording/recording.sigmf-meta", meta),
                        special("recording/recording.sigmf-data", kind, linkname),
                    ]
                )
                with self.assertRaisesRegex(ValueError, "link or special file"):
                    load_recording("x.sigmf", archive)

    def test_oversized_members_are_rejected_before_reading(self):
        extract = patch.object(
            tarfile.TarFile, "extractfile", autospec=True, side_effect=tarfile.TarFile.extractfile
        )
        too_long = bytes(2 * (MAX_SAMPLES + 1))
        with extract as extracted:
            with self.assertRaisesRegex(ValueError, "more than 100,000 samples"):
                load_recording("x.sigmf", sigmf_archive(sigmf_meta("cu8"), too_long))
        self.assertEqual(
            [call.args[1].name for call in extracted.call_args_list],
            ["recording/recording.sigmf-meta"],
        )

        padded_meta = sigmf_meta() + b" " * SIGMF_META_MAX_BYTES
        with extract as extracted:
            with self.assertRaisesRegex(ValueError, "metadata exceeds"):
                load_recording("x.sigmf", sigmf_archive(padded_meta, interleaved([1, 2], "<f4")))
        extracted.assert_not_called()

    def test_largest_archive_fits_the_limits(self):
        meta = sigmf_meta() + b" " * (SIGMF_META_MAX_BYTES - len(sigmf_meta()))
        data = interleaved([1, 2], "<f4", MAX_SAMPLES)
        archive = sigmf_archive(meta, data)
        self.assertLessEqual(len(archive), SIGMF_ARCHIVE_MAX_BYTES)
        self.assertEqual(len(load_recording("x.sigmf", archive).samples), MAX_SAMPLES)

    def test_oversized_archive_is_rejected_before_opening(self):
        with patch("tarfile.open", wraps=tarfile.open) as opened:
            with self.assertRaisesRegex(ValueError, "archive exceeds"):
                load_recording("x.sigmf", bytes(SIGMF_ARCHIVE_MAX_BYTES + 1))
        opened.assert_not_called()

    def test_damaged_or_compressed_archives_are_rejected(self):
        archive = sigmf_archive(sigmf_meta(), interleaved([1, 2], "<f4"))
        # The archive has 512-byte blocks: a directory header, the metadata
        # header and contents, then the data header and contents.
        for damaged in (
            b"",
            b"hello" * 1000,
            gzip.compress(archive),
            archive[:600],
            archive[:1100],
            archive[:2000],
            archive[:4000],
            archive + b"trailing bytes",
        ):
            with self.subTest(size=len(damaged)):
                with self.assertRaisesRegex(ValueError, "could not be read|truncated or damaged"):
                    load_recording("x.sigmf", damaged)

    def test_invalid_sample_data_is_rejected(self):
        for data, message in (
            (interleaved([1, 2], "<f4")[:-3], "whole number"),
            (interleaved([1, 2], "<f4", SAMPLES_PER_WINDOW - 1), "at least 1,024"),
            (interleaved([np.nan, 2], "<f4"), "not finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    load_recording("x.sigmf", sigmf_archive(sigmf_meta(), data))


class WavTests(unittest.TestCase):
    def assert_rejected(self, data: bytes, message: str) -> str:
        with self.assertRaises(ValueError) as caught:
            load_recording("capture.wav", data)
        text = assert_plain_message(self, caught.exception)
        self.assertIn(message, text)
        return text

    def test_each_sample_type_is_scaled_to_the_unit_range(self):
        for (code, bits), (_, values, _, _) in WAV_SAMPLE_TYPES.items():
            for extensible in (False, True):
                with self.subTest(code=code, bits=bits, extensible=extensible):
                    data = wav_file(
                        wav_format(code, bits, extensible=extensible),
                        wav_chunk(b"data", wav_values(values, code, bits)),
                    )
                    recording = load_recording("capture.wav", data)
                    self.assertEqual(recording.samples.dtype, np.dtype(np.complex128))
                    np.testing.assert_array_equal(recording.samples, wav_expected(values, code, bits))
                    self.assertEqual(recording.sample_rate, 12_000.0)
                    self.assertIsNone(recording.center_frequency)
                    kind = "float" if code == WAV_FLOAT else "PCM"
                    self.assertRegex(
                        recording.source_format, rf"^WAV \({bits}-bit .*{kind}, I and Q channels\)$"
                    )

    def test_integer_extremes_reach_minus_one(self):
        for (code, bits), (_, values, _, _) in WAV_SAMPLE_TYPES.items():
            if code == WAV_PCM:
                with self.subTest(bits=bits):
                    samples_data = wav_chunk(b"data", wav_values(values, code, bits))
                    samples = load_recording("x.wav", wav_file(wav_format(code, bits), samples_data)).samples
                    components = np.concatenate([samples.real, samples.imag])
                    self.assertEqual(components.min(), -1.0)
                    self.assertLess(components.max(), 1.0)
                    self.assertGreater(components.max(), 0.99)

    def test_files_written_by_the_standard_wave_module(self):
        # An independent writer of 8-, 16-, 24-, and 32-bit PCM files.
        rng = np.random.default_rng(5)
        for width, dtype in ((1, "u1"), (2, "<i2"), (3, "<i4"), (4, "<i4")):
            with self.subTest(bits=8 * width):
                high = 2 ** (8 * width - 1)
                low, high = (0, 2 * high) if width == 1 else (-high, high)
                values = rng.integers(low, high, 2 * SAMPLES_PER_WINDOW, dtype=np.int64)
                buffer = BytesIO()
                with wave.open(buffer, "wb") as writer:
                    writer.setnchannels(2)
                    writer.setsampwidth(width)
                    writer.setframerate(250_000)
                    writer.writeframes(wav_values(values.astype(dtype), WAV_PCM, 8 * width))
                recording = load_recording("sdr.WAV", buffer.getvalue())
                np.testing.assert_array_equal(recording.samples, wav_expected(values, WAV_PCM, 8 * width))
                self.assertEqual(recording.sample_rate, 250_000.0)

    def test_kiwirecorder_files_with_one_or_many_data_chunks(self):
        # kiwirecorder rounds the sample rate to whole hertz for the header, and
        # --kiwi-wav writes a "kiwi" time stamp chunk before each data chunk.
        rng = np.random.default_rng(8)
        values = rng.integers(-32768, 32768, (3 * 512, 2), dtype=np.int64).astype("<i2")
        expected = (values[:, 0] + 1j * values[:, 1].astype(np.float64)) / 32768
        blocks = [values[:512], values[512:1000], values[1000:]]
        for kiwi_wav in (False, True):
            with self.subTest(kiwi_wav=kiwi_wav):
                data = kiwirecorder_wav(blocks, 12_001.2, kiwi_wav)
                self.assertEqual(data.count(b"data"), 3 if kiwi_wav else 1)
                recording = load_recording("20260925T164556Z_23400_kiwi_iq.wav", data)
                np.testing.assert_array_equal(recording.samples, expected)
                self.assertEqual(recording.sample_rate, 12_001.0)
                self.assertEqual(recording.source_format, "WAV (16-bit integer PCM, I and Q channels)")

    def test_center_frequency_comes_from_an_auxi_chunk(self):
        data = wav_values([100, -100])
        for label, chunks, frequency in (
            ("before the data", [auxi_chunk(7_100_000), wav_chunk(b"data", data)], 7.1e6),
            ("after the data", [wav_chunk(b"data", data), auxi_chunk(144_800_000)], 144.8e6),
            ("largest value", [auxi_chunk(2**32 - 1), wav_chunk(b"data", data)], float(2**32 - 1)),
            ("header part only", [auxi_chunk(1_000, size=36), wav_chunk(b"data", data)], 1_000.0),
            ("first of two", [auxi_chunk(5), wav_chunk(b"data", data), auxi_chunk(6)], 5.0),
            ("zero means unknown", [auxi_chunk(0), wav_chunk(b"data", data)], None),
            ("too short", [auxi_chunk(7_100_000, size=35), wav_chunk(b"data", data)], None),
            (
                "impossible time",
                [auxi_chunk(7_100_000, time=(2026, 13, 5, 25, 16, 45, 56, 250)), wav_chunk(b"data", data)],
                None,
            ),
            (
                # SDR Console stores XML text in its "auxi" chunk.
                "UTF-16 XML",
                [wav_chunk(b"auxi", '<?xml version="1.0"?><SDR-XML-Root/>'.encode("utf-16-le")),
                 wav_chunk(b"data", data)],
                None,
            ),
        ):
            with self.subTest(label):
                recording = load_recording("x.wav", wav_file(wav_format(), *chunks))
                self.assertEqual(recording.center_frequency, frequency)
                self.assertEqual(len(recording.samples), SAMPLES_PER_WINDOW)

    def test_sample_rate_comes_from_the_header(self):
        for rate, expected in ((11_999, 11_999.0), (2_400_000, 2_400_000.0), (0, None)):
            with self.subTest(rate=rate):
                data = wav_file(wav_format(rate=rate), wav_chunk(b"data", wav_values([1, 2])))
                self.assertEqual(load_recording("x.wav", data).sample_rate, expected)

    def test_other_chunks_and_padding_are_skipped(self):
        values = [300, -300]
        expected = wav_expected(values)
        samples = wav_values(values)
        odd_text = HOSTILE_TEXT.encode()[:63]
        for label, data in (
            (
                "odd-sized chunks with padding",
                wav_file(
                    wav_chunk(b"LIST", odd_text), wav_format(), wav_chunk(b"JUNK", b"x"),
                    wav_chunk(b"fact", struct.pack("<I", SAMPLES_PER_WINDOW)), wav_chunk(b"data", samples),
                ),
            ),
            (
                "odd-sized last chunk without padding",
                wav_file(wav_format(), wav_chunk(b"data", samples), wav_chunk(b"LIST", odd_text, pad=False)),
            ),
            ("empty chunks", wav_file(wav_chunk(b"JUNK", b""), wav_format(), wav_chunk(b"data", b""),
                                      wav_chunk(b"data", samples))),
            # An ID3 tag appended after the RIFF data is outside the WAV file.
            ("bytes after the RIFF end", pcm16_wav(values=values) + b"TAG" + HOSTILE_TEXT.encode()),
            # A file cut exactly after its data chunk is complete.
            ("RIFF size too large", wav_file(wav_format(), wav_chunk(b"data", samples), riff_size=10**6)),
            # The data chunk is complete, but the RIFF size ends partway through it.
            (
                "RIFF size too small",
                wav_file(wav_format(), wav_chunk(b"data", samples), riff_size=4 + 24 + 8 + 4),
            ),
        ):
            with self.subTest(label):
                np.testing.assert_array_equal(load_recording("x.wav", data).samples, expected)

    def test_many_small_chunks_are_walked_quickly(self):
        chunks = [wav_chunk(b"JUNK", b"")] * 100_000 + [wav_chunk(b"data", b"")] * 100_000
        data = pcm16_wav(*chunks)
        started = time.perf_counter()
        self.assertEqual(len(load_recording("x.wav", data).samples), SAMPLES_PER_WINDOW)
        self.assertLess(time.perf_counter() - started, 5)

    def test_one_channel_is_audio_and_other_counts_are_rejected(self):
        for code, bits in ((WAV_PCM, 16), (WAV_FLOAT, 32), (0x0055, 0)):
            with self.subTest(code=code, bits=bits):
                mono = wav_file(wav_format(code, bits, channels=1), wav_chunk(b"data", bytes(4096)))
                self.assert_rejected(mono, "one channel, so it looks like ordinary audio, not I/Q")
        for channels in (0, 3, 4, 65535):
            with self.subTest(channels=channels):
                data = wav_file(wav_format(channels=channels, block=4), wav_chunk(b"data", bytes(4096)))
                self.assert_rejected(data, f"has {channels:,} channels. An I/Q recording has exactly two")

    def test_unsupported_sample_formats_are_explained(self):
        data = wav_chunk(b"data", bytes(8 * SAMPLES_PER_WINDOW))
        guid = wav_format(WAV_PCM, 16, extensible=True)
        for label, fmt, message in (
            ("12-bit PCM", wav_format(WAV_PCM, 12, block=4), "holds 12-bit integer samples"),
            ("16-bit float", wav_format(WAV_FLOAT, 16), "holds 16-bit float samples"),
            ("24-bit float", wav_format(WAV_FLOAT, 24, extensible=True), "holds 24-bit float samples"),
            ("IMA ADPCM", wav_format(0x0011, 4, block=2048), "compressed or other unsupported"),
            ("MP3", wav_format(0x0055, 0, block=1), "compressed or other unsupported"),
            ("A-law", wav_format(0x0006, 8), "compressed or other unsupported"),
            ("unknown GUID", guid[:-1] + b"\x72", "compressed or other unsupported"),
            ("extensible without GUID", wav_chunk(b"fmt ", guid[8:26]), "compressed or other unsupported"),
            ("block size", wav_format(WAV_PCM, 16, block=8), "gives 8 bytes per I/Q sample, but two"),
            ("short format", wav_chunk(b"fmt ", guid[8:22]), "format chunk is too short"),
        ):
            with self.subTest(label):
                text = self.assert_rejected(wav_file(fmt, data), message)
                if "unsupported" in message or "holds" in message:
                    self.assertIn("8-, 16-, 24-, or 32-bit integer PCM, or 32- or 64-bit float", text)

    def test_other_riff_variants_and_other_files_are_rejected(self):
        valid = pcm16_wav()
        for label, data, message in (
            ("big-endian", b"RIFX" + valid[4:], "big-endian (RIFX)"),
            ("RF64", b"RF64" + b"\xff" * 4 + valid[8:], "RF64 WAV file"),
            ("BW64", b"BW64" + b"\xff" * 4 + valid[8:], "RF64 WAV file"),
            ("AVI", valid[:8] + b"AVI " + valid[12:], "does not hold WAVE audio"),
            ("raw samples", wav_values([1, 2]), "does not start with a RIFF header"),
            ("text", HOSTILE_TEXT.encode(), "does not start with a RIFF header"),
            ("empty", b"", "too short to be a WAV file"),
            ("header only", valid[:11], "too short to be a WAV file"),
        ):
            with self.subTest(label):
                self.assert_rejected(data, message)

    def test_missing_or_repeated_chunks_are_rejected(self):
        samples = wav_chunk(b"data", wav_values([1, 2]))
        for label, data, message in (
            ("no format", wav_file(samples), "no format chunk"),
            ("no chunks", wav_file(), "no format chunk"),
            ("two formats", wav_file(wav_format(), wav_format(), samples), "more than one format chunk"),
            ("no data", wav_file(wav_format(), wav_chunk(b"LIST", b"INFO")), "no data chunk"),
            ("empty data", wav_file(wav_format(), wav_chunk(b"data", b"")), "at least 1,024 I/Q samples"),
            # A data chunk inside a LIST chunk is not sample data.
            ("data inside LIST", wav_file(wav_format(), wav_chunk(b"LIST", samples)), "no data chunk"),
        ):
            with self.subTest(label):
                self.assert_rejected(data, message)

    def test_truncated_and_huge_chunks_are_rejected_before_decoding(self):
        valid = pcm16_wav(auxi_chunk(7_100_000))
        header_end = 12 + 24 + 8 + 164  # RIFF header, fmt chunk, auxi chunk
        # Cutting 4,096 + 8 bytes would leave a whole file without a data chunk.
        cases = {
            f"cut {cut} bytes": valid[:-cut] for cut in (1, 2, 4, 4096, 4096 + 5, 4096 + 7)
        }
        cases["cut inside the fmt chunk"] = valid[:30]
        cases["cut inside the auxi chunk"] = valid[:100]
        cases["cut inside a data chunk header"] = valid[: header_end + 4]
        for size in (2**32 - 1, 2**32 - 2, 2**31, 4097):
            cases[f"data size {size:#x}"] = (
                valid[:header_end] + struct.pack("<4sI", b"data", size) + valid[header_end + 8 :]
            )
            cases[f"LIST size {size:#x}"] = wav_file(
                wav_format(), wav_chunk(b"data", wav_values([1, 2])), struct.pack("<4sI", b"LIST", size)
            )
        # The kiwirecorder header of a stream sent to standard output (--nc-wav).
        cases["streamed"] = (
            struct.pack("<4sI4s", b"RIFF", 0x7FFFFFF0 - 8, b"WAVE") + wav_format()
            + struct.pack("<4sI", b"data", 0x7FFFFFF0 - 44) + wav_values([1, 2])
        )
        with patch("iq_formats._decode", wraps=iq_formats._decode) as decode:
            for label, data in cases.items():
                with self.subTest(label):
                    self.assert_rejected(data, "The WAV file is truncated")
        decode.assert_not_called()

    def test_sample_count_limits(self):
        for (code, bits), (_, values, _, _) in WAV_SAMPLE_TYPES.items():
            with self.subTest(code=code, bits=bits):
                width = 2 * bits // 8

                def wav(count: int, cut: int = 0) -> bytes:
                    """A file of this many samples, less the last cut bytes."""
                    contents = wav_values(values, code, bits, count)
                    contents = contents[: len(contents) - cut]
                    return wav_file(wav_format(code, bits), wav_chunk(b"data", contents))

                for count in (SAMPLES_PER_WINDOW, MAX_SAMPLES):
                    self.assertEqual(len(load_recording("x.wav", wav(count)).samples), count)
                self.assert_rejected(wav(SAMPLES_PER_WINDOW - 1), "needs at least 1,024 I/Q samples")
                too_long = self.assert_rejected(wav(MAX_SAMPLES + 1), "more than 100,000 samples")
                self.assertIn(f"at most {MAX_SAMPLES * width:,} bytes", too_long)
                self.assert_rejected(wav(SAMPLES_PER_WINDOW + 1, cut=1), "whole number")

    def test_too_many_samples_are_rejected_before_decoding(self):
        # Samples split over several data chunks count together.
        half = wav_chunk(b"data", wav_values([1, 2], samples=MAX_SAMPLES // 2 + 1))
        with patch("iq_formats._decode", wraps=iq_formats._decode) as decode:
            self.assert_rejected(wav_file(wav_format(), half, half), "more than 100,000 samples")
            limit = f"exceeds the {WAV_MAX_BYTES // 10**5 / 10:g} MB limit"
            self.assert_rejected(b"RIFF" + bytes(WAV_MAX_BYTES), limit)
        decode.assert_not_called()

    def test_largest_file_fits_the_limit(self):
        metadata = wav_chunk(b"LIST", bytes(900_000))
        data = wav_file(
            wav_format(WAV_FLOAT, 64, extensible=True), metadata,
            wav_chunk(b"data", wav_values([0.5, -0.5], WAV_FLOAT, 64, MAX_SAMPLES)),
        )
        self.assertLessEqual(len(data), WAV_MAX_BYTES)
        self.assertEqual(len(load_recording("x.wav", data).samples), MAX_SAMPLES)

    def test_nonfinite_float_values_are_rejected(self):
        for bits in (32, 64):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(bits=bits, value=value):
                    samples = wav_chunk(b"data", wav_values([0.5, value], WAV_FLOAT, bits))
                    self.assert_rejected(wav_file(wav_format(WAV_FLOAT, bits), samples), "not finite")

    def test_damaged_headers_only_ever_raise_value_errors(self):
        # Change random bytes among the chunk headers and the format and auxi
        # chunks, sometimes writing a huge size. Each file either loads or
        # fails with a plain ValueError.
        values = [1000, -1000]
        valid = wav_file(
            wav_chunk(b"LIST", HOSTILE_TEXT.encode() + b"!"), wav_format(), auxi_chunk(7_100_000),
            struct.pack("<4sIBBII", b"kiwi", 10, 255, 0, 1, 0), wav_chunk(b"data", wav_values(values)),
        )
        header_bytes = len(valid) - 4 * SAMPLES_PER_WINDOW
        rng = np.random.default_rng(0)
        for trial in range(400):
            damaged = bytearray(valid)
            for _ in range(int(rng.integers(1, 5))):
                position = int(rng.integers(0, header_bytes))
                if rng.random() < 0.2:
                    size = int(rng.choice([2**32 - 1, 2**31, 7]))
                    damaged[position : position + 4] = struct.pack("<I", size)
                else:
                    damaged[position] = int(rng.integers(0, 256))
            try:
                load_recording("x.wav", bytes(damaged))
            except ValueError as error:
                assert_plain_message(self, error)
            except Exception as error:  # noqa: BLE001 - any other type is the bug.
                self.fail(f"trial {trial}: {type(error).__name__}: {error!r}")

    def test_file_text_is_never_repeated(self):
        hostile = HOSTILE_TEXT.encode()
        for label, data in (
            ("form", hostile[:4] + pcm16_wav()[4:]),
            ("RIFF type", pcm16_wav()[:8] + hostile[:4] + pcm16_wav()[12:]),
            ("chunk IDs", wav_file(wav_chunk(hostile[:4], hostile), wav_chunk(b"data", bytes(4096)))),
            ("metadata", wav_file(wav_chunk(b"LIST", hostile), wav_format(channels=1))),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    load_recording("x.wav", data)
                assert_plain_message(self, caught.exception)


class SelectUploadsTests(unittest.TestCase):
    """The uploader keeps earlier files when more are added; the newest is read."""

    @staticmethod
    def names(files) -> list[str]:
        return [name for name, _ in files]

    def select(self, *names: str) -> tuple[list[str], int]:
        files, ignored = select_uploads([(name, name.encode()) for name in names])
        self.assertEqual([data for _, data in files], [name.encode() for name, _ in files])
        return self.names(files), ignored

    def test_the_last_file_is_read(self):
        for names, expected in (
            (["a.csv"], (["a.csv"], 0)),
            (["a.csv", "b.csv"], (["b.csv"], 1)),
            (["a.sigmf-meta", "a.sigmf-data", "b.cu8"], (["b.cu8"], 2)),
            ([], ([], 0)),
        ):
            with self.subTest(names=names):
                self.assertEqual(self.select(*names), expected)

    def test_the_last_file_brings_the_other_half_of_its_sigmf_pair(self):
        for names, expected in (
            (["rec.sigmf-data", "rec.sigmf-meta"], (["rec.sigmf-data", "rec.sigmf-meta"], 0)),
            (["x.csv", "rec.sigmf-meta", "rec.sigmf-data"], (["rec.sigmf-meta", "rec.sigmf-data"], 1)),
            (
                ["a.sigmf-meta", "a.sigmf-data", "b.sigmf-meta", "b.SIGMF-DATA"],
                (["b.sigmf-meta", "b.SIGMF-DATA"], 2),
            ),
            (
                ["b.sigmf-data", "a.sigmf-meta", "a.sigmf-data", "b.sigmf-meta"],
                (["b.sigmf-data", "b.sigmf-meta"], 2),
            ),
            (["a.sigmf-meta", "a.sigmf-data", "a.sigmf-data"], (["a.sigmf-meta", "a.sigmf-data"], 1)),
            # Without a match, the other kind is paired so that the names can be reported.
            (["a.sigmf-meta", "b.sigmf-data"], (["a.sigmf-meta", "b.sigmf-data"], 0)),
            (["x.csv", "rec.sigmf-meta"], (["rec.sigmf-meta"], 1)),
        ):
            with self.subTest(names=names):
                self.assertEqual(self.select(*names), expected)

    def test_selected_files_load(self):
        first = to_csv(random_complex(seed=1)).encode()
        second = random_complex(2 * SAMPLES_PER_WINDOW, seed=2)
        files, ignored = select_uploads([("a.csv", first), ("b.csv", to_csv(second).encode())])
        self.assertEqual(ignored, 1)
        np.testing.assert_array_equal(load_uploads(files).samples, second)
        meta, data = sigmf_meta(), random_complex().astype(np.complex64).tobytes()
        for names, message in (
            (["x.csv", "rec.sigmf-meta"], "needs both files"),
            (["a.sigmf-meta", "b.sigmf-data"], "different names"),
        ):
            with self.subTest(names=names):
                uploads = [(name, meta if name.endswith("meta") else data) for name in names]
                with self.assertRaisesRegex(ValueError, message):
                    load_uploads(select_uploads(uploads)[0])


class SigmfPairTests(unittest.TestCase):
    def setUp(self):
        self.samples = random_complex().astype(np.complex64)
        self.meta = sigmf_meta("cf32_le", sample_rate=1e6, frequency=-2.5e3)

    def test_pair_round_trip(self):
        recording = load_sigmf_pair(self.meta, self.samples.tobytes())
        np.testing.assert_array_equal(recording.samples, self.samples.astype(np.complex128))
        self.assertEqual((recording.sample_rate, recording.center_frequency), (1e6, -2.5e3))
        self.assertEqual(recording.source_format, "SigMF recording (cf32_le)")

    def test_uploads_pair_in_either_order(self):
        meta = ("rec.sigmf-meta", self.meta)
        data = ("rec.sigmf-data", self.samples.tobytes())
        for files in ((meta, data), (data, meta)):
            with self.subTest(first=files[0][0]):
                recording = load_uploads(files)
                np.testing.assert_array_equal(recording.samples, self.samples.astype(np.complex128))

    def test_pair_names_must_match(self):
        with self.assertRaisesRegex(ValueError, "different names"):
            load_uploads([("a.sigmf-meta", self.meta), ("b.sigmf-data", self.samples.tobytes())])

    def test_half_of_a_pair_explains_both_files_are_needed(self):
        for name in ("rec.sigmf-meta", "rec.sigmf-data"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "needs both files"):
                    load_uploads([(name, self.meta)])

    def test_other_multiple_uploads_are_rejected(self):
        csv = to_csv(np.ones(SAMPLES_PER_WINDOW)).encode()
        for files in (
            [("a.csv", csv), ("b.csv", csv)],
            [("a.sigmf-meta", self.meta), ("a.csv", csv)],
            [("a.sigmf-meta", self.meta), ("a.sigmf-data", b""), ("a.csv", csv)],
            [],
        ):
            with self.subTest(names=[name for name, _ in files]):
                with self.assertRaisesRegex(ValueError, f"{len(files)} files were uploaded"):
                    load_uploads(files)

    def test_pair_metadata_and_data_limits(self):
        with self.assertRaisesRegex(ValueError, "metadata exceeds"):
            load_sigmf_pair(self.meta + b" " * SIGMF_META_MAX_BYTES, self.samples.tobytes())
        with self.assertRaisesRegex(ValueError, "whole number"):
            load_sigmf_pair(self.meta, self.samples.tobytes()[:-1])


class DispatchTests(unittest.TestCase):
    def test_csv_is_read_by_the_csv_module(self):
        data = to_csv(random_complex()).encode()
        recording = load_recording("Example.CSV", data)
        np.testing.assert_array_equal(recording.samples, from_csv(data))
        self.assertIsNone(recording.sample_rate)
        with self.assertRaisesRegex(ValueError, "columns named I,Q"):
            load_recording("x.csv", b"real,imaginary\n")

    def test_every_supported_extension_has_a_reader(self):
        for extension in SUPPORTED_EXTENSIONS:
            with self.subTest(extension=extension):
                with self.assertRaises(ValueError) as caught:
                    load_recording(f"x.{extension}", b"")
                self.assertNotIn("not supported", str(caught.exception))

    def test_unknown_extensions_list_the_supported_ones(self):
        for name in ("recording.flac", "recording", "recording.npz", "archive.tar"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "is not supported") as caught:
                    load_recording(name, b"\0" * 4096)
                for extension in SUPPORTED_EXTENSIONS:
                    self.assertIn(f".{extension}", str(caught.exception))

    def test_only_a_plain_extension_of_the_name_is_repeated(self):
        for name, start in (
            ("Recording.FLAC", "The .flac format is not supported."),
            ("recording", "This file's format is not supported."),
            (f"x.{HOSTILE_TEXT}", "This file's format is not supported."),
            ("x.w\ud800v", "This file's format is not supported."),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    load_recording(name, b"\0" * 4096)
                self.assertTrue(assert_plain_message(self, caught.exception).startswith(start))

    def test_upload_widget_limit_covers_every_reader(self):
        largest_raw = MAX_SAMPLES * 8
        for limit in (MAX_UPLOAD_BYTES, NPY_MAX_BYTES, SIGMF_ARCHIVE_MAX_BYTES, WAV_MAX_BYTES, largest_raw):
            self.assertLessEqual(limit, MAX_FILE_BYTES)

    def test_standalone_import_does_not_load_app_or_training_modules(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import iq_formats; "
                "heavy = {'streamlit', 'sklearn', 'matplotlib', 'plotly', 'spectrum_sense'}; "
                "loaded = heavy & sys.modules.keys(); "
                "assert not loaded, loaded",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
