"""Regression tests for reading raw SDR, NumPy, and SigMF recordings."""

import gzip
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch

import numpy as np

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, from_csv, to_csv
from iq_formats import (
    MAX_FILE_BYTES,
    NPY_MAX_BYTES,
    SIGMF_ARCHIVE_MAX_BYTES,
    SIGMF_META_MAX_BYTES,
    SUPPORTED_EXTENSIONS,
    load_recording,
    load_sigmf_pair,
    load_uploads,
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
        for datatype in ("rf32_le", "cf32_be", "ci16_be", "cu16_le", "cf64_le", "mystery"):
            with self.subTest(datatype=datatype):
                with self.assertRaisesRegex(ValueError, f'datatype "{datatype}" is not supported'):
                    load_recording("x.sigmf", sigmf_archive(sigmf_meta(datatype), data))

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
        for name in ("recording.wav", "recording", "recording.npz", "archive.tar"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "is not supported") as caught:
                    load_recording(name, b"\0" * 4096)
                for extension in SUPPORTED_EXTENSIONS:
                    self.assertIn(f".{extension}", str(caught.exception))

    def test_upload_widget_limit_covers_every_reader(self):
        largest_raw = MAX_SAMPLES * 8
        for limit in (MAX_UPLOAD_BYTES, NPY_MAX_BYTES, SIGMF_ARCHIVE_MAX_BYTES, largest_raw):
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
