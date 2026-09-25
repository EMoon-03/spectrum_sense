"""Load one I/Q recording from CSV, raw SDR samples, NumPy, or SigMF files.

Only NumPy and the standard library are used, so this module stays light.
Every problem with a file is reported as a plain-language ValueError. A shared
file is untrusted input, so messages never repeat free text from it, such as
names inside an archive or metadata values; the app shows them as Markdown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import io
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
import tarfile
from typing import TypeVar

import numpy as np
from numpy.lib import format as npy_format

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, from_csv
from signal_config import SAMPLES_PER_WINDOW


Upload = TypeVar("Upload")  # Anything that stands for one uploaded file.


@dataclass(frozen=True, eq=False)
class Recording:
    """Complex128 samples plus the metadata a file provides, if any."""

    samples: np.ndarray
    sample_rate: float | None
    center_frequency: float | None
    source_format: str


@dataclass(frozen=True)
class _SampleType:
    """Interleaved I,Q values of one NumPy type, scaled by (value - offset) / full_scale."""

    name: str
    description: str
    component: str
    offset: float = 0.0
    full_scale: float = 1.0

    @property
    def bytes_per_sample(self) -> int:
        return 2 * np.dtype(self.component).itemsize


_CF32 = _SampleType("cf32", "complex float32, little-endian", "<f4")
_CS16 = _SampleType("cs16", "complex int16, little-endian", "<i2", full_scale=32768.0)
_CS8 = _SampleType("cs8", "complex int8", "i1", full_scale=128.0)
_CU8 = _SampleType("cu8", "complex uint8", "u1", offset=127.5, full_scale=127.5)

_RAW_TYPES = {
    "cf32": _CF32,
    "fc32": _CF32,
    "cfile": _CF32,
    "cs16": _CS16,
    "sc16": _CS16,
    "cs8": _CS8,
    "cu8": _CU8,
}
_SIGMF_TYPES = {"cf32_le": _CF32, "ci16_le": _CS16, "ci8": _CS8, "cu8": _CU8}

SUPPORTED_EXTENSIONS = ("csv", *_RAW_TYPES, "npy", "sigmf", "sigmf-meta", "sigmf-data")
_SIGMF_PAIR = {"sigmf-meta": "sigmf-data", "sigmf-data": "sigmf-meta"}
# Text from a file is repeated in a message only when it has one of these
# forms, which hold nothing that Markdown could turn into links or formatting.
_SIGMF_DATATYPE = re.compile(r"[rc](?:f64|f32|i32|i16|u32|u16|i8|u8)(?:_le|_be)?")
_PLAIN_EXTENSION = re.compile(r"[a-z0-9]{1,16}")

# np.save writes a header of about 128 bytes; NumPy refuses headers over 10,000.
_NPY_HEADER_BYTES = 16_384
NPY_MAX_BYTES = MAX_SAMPLES * np.dtype(np.complex128).itemsize + _NPY_HEADER_BYTES
SIGMF_META_MAX_BYTES = 1_000_000
# Tar headers, extended headers, directory entries, and end-of-archive padding.
_TAR_OVERHEAD_BYTES = 65_536
SIGMF_ARCHIVE_MAX_BYTES = (
    SIGMF_META_MAX_BYTES
    + MAX_SAMPLES * max(kind.bytes_per_sample for kind in _SIGMF_TYPES.values())
    + _TAR_OVERHEAD_BYTES
)
# The largest file any reader accepts; the upload widget must admit it.
MAX_FILE_BYTES = max(
    MAX_UPLOAD_BYTES,
    NPY_MAX_BYTES,
    SIGMF_ARCHIVE_MAX_BYTES,
    *(MAX_SAMPLES * kind.bytes_per_sample for kind in _RAW_TYPES.values()),
)


def _extension(name: str) -> str:
    return PurePosixPath(name).suffix.lower().removeprefix(".")


def _check_size(size: int, kind: _SampleType, what: str) -> None:
    width = kind.bytes_per_sample
    if size > MAX_SAMPLES * width:
        raise ValueError(
            f"{what} has more than {MAX_SAMPLES:,} samples. Upload a shorter recording: "
            f"at most {MAX_SAMPLES * width:,} bytes of {kind.name} data."
        )
    if size % width:
        raise ValueError(
            f"{what} is {size:,} bytes, which is not a whole number of {width}-byte "
            f"{kind.name} I/Q samples. It may be truncated or in a different format."
        )
    if size < SAMPLES_PER_WINDOW * width:
        raise ValueError(
            f"{what} needs at least {SAMPLES_PER_WINDOW:,} I/Q samples "
            f"({SAMPLES_PER_WINDOW * width:,} bytes of {kind.name} data)."
        )


def _decode(data: bytes, kind: _SampleType, what: str) -> np.ndarray:
    """Interleaved I,Q values to complex128, scaled to about [-1, 1] for integers."""
    _check_size(len(data), kind, what)
    components = np.frombuffer(data, dtype=kind.component).astype(np.float64)
    if not np.isfinite(components).all():
        raise ValueError(f"{what} contains values that are not finite numbers (NaN or infinity).")
    components -= kind.offset
    components /= kind.full_scale
    return components.view(np.complex128)


def _read_npy(data: bytes) -> Recording:
    if len(data) > NPY_MAX_BYTES:
        raise ValueError(
            f"The .npy file exceeds the {format_size(NPY_MAX_BYTES)} limit. "
            f"Save a shorter recording (at most {MAX_SAMPLES:,} samples)."
        )
    stream = io.BytesIO(data)
    readers = {
        (1, 0): npy_format.read_array_header_1_0,
        (2, 0): npy_format.read_array_header_2_0,
    }
    try:
        shape, _, dtype = readers[npy_format.read_magic(stream)](stream)
    except Exception as exc:  # A damaged header can fail inside NumPy in many ways.
        raise ValueError(
            "This is not a readable NumPy .npy file (format version 1.0 or 2.0)."
        ) from exc

    # Check the header before np.load, which allocates the array it describes.
    if dtype.hasobject:
        raise ValueError(
            "This .npy file stores Python objects, which are never loaded for safety. "
            "Save a plain numeric array instead."
        )
    # Field names of a structured array come from the file, so only its kind is named.
    shown = "structured" if dtype.fields is not None else str(dtype)
    complex_column = len(shape) == 1 and dtype.kind == "c" and dtype.itemsize <= 16
    real_pairs = (
        len(shape) == 2 and shape[1] == 2 and dtype.kind in "iuf" and dtype.itemsize <= 8
    )
    if not (complex_column or real_pairs):
        raise ValueError(
            f"This .npy file holds a {shown} array with shape {shape}. Save a "
            "one-dimensional complex64 or complex128 array, or a real array with "
            "two columns (I and Q)."
        )
    if shape[0] > MAX_SAMPLES:
        raise ValueError(
            f"The .npy file has more than {MAX_SAMPLES:,} samples. Save a shorter recording."
        )
    if shape[0] < SAMPLES_PER_WINDOW:
        raise ValueError(f"The .npy file needs at least {SAMPLES_PER_WINDOW:,} samples.")
    if len(data) - stream.tell() != math.prod(shape) * dtype.itemsize:
        raise ValueError("The .npy file is truncated or has extra bytes after the array.")

    array = np.load(io.BytesIO(data), allow_pickle=False)
    if complex_column:
        samples = array.astype(np.complex128)
        description = dtype.name
    else:
        samples = np.empty(len(array), dtype=np.complex128)
        samples.real = array[:, 0]
        samples.imag = array[:, 1]
        description = f"{dtype.name}, I and Q columns"
    if not np.isfinite(samples).all():
        raise ValueError("The .npy file contains values that are not finite numbers.")
    return Recording(samples, None, None, f"NumPy array ({description})")


def _optional_number(section: dict, key: str) -> float | None:
    value = section.get(key)
    if value is None:
        return None
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    try:
        number = float(value) if is_number else math.nan
    except OverflowError:  # An integer too large for a float.
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(f'The SigMF field "{key}" must be a finite number.')
    return number


def _parse_sigmf_meta(meta: bytes) -> tuple[str, _SampleType, float | None, float | None]:
    """Datatype name, sample type, sample rate, and center frequency."""
    if len(meta) > SIGMF_META_MAX_BYTES:
        raise ValueError(
            f"The SigMF metadata exceeds the {format_size(SIGMF_META_MAX_BYTES)} limit."
        )
    try:
        document = json.loads(meta.decode("utf-8-sig"))
    except (ValueError, RecursionError) as exc:
        raise ValueError("The SigMF metadata is not valid UTF-8 JSON.") from exc
    section = document.get("global") if isinstance(document, dict) else None
    if not isinstance(section, dict):
        raise ValueError('The SigMF metadata has no "global" section.')

    datatype = section.get("core:datatype")
    if not isinstance(datatype, str):
        raise ValueError('The SigMF metadata does not state its "core:datatype".')
    kind = _SIGMF_TYPES.get(datatype)
    if kind is None:
        supported = f"Use complex samples in one of these datatypes: {', '.join(_SIGMF_TYPES)}."
        if _SIGMF_DATATYPE.fullmatch(datatype):
            raise ValueError(f'The SigMF datatype "{datatype}" is not supported. {supported}')
        raise ValueError(
            f'The SigMF metadata\'s "core:datatype" is not a valid SigMF datatype. {supported}'
        )
    channels = section.get("core:num_channels", 1)
    if isinstance(channels, bool) or channels != 1:
        raise ValueError(
            'The SigMF recording must have exactly one channel ("core:num_channels": 1).'
        )

    sample_rate = _optional_number(section, "core:sample_rate")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError('The SigMF field "core:sample_rate" must be positive.')
    captures = document.get("captures")
    frequency = None
    if isinstance(captures, list) and captures and isinstance(captures[0], dict):
        frequency = _optional_number(captures[0], "core:frequency")
    return datatype, kind, sample_rate, frequency


def load_sigmf_pair(meta: bytes, data: bytes) -> Recording:
    """Read a SigMF recording uploaded as separate .sigmf-meta and .sigmf-data files."""
    datatype, kind, sample_rate, frequency = _parse_sigmf_meta(meta)
    samples = _decode(data, kind, "The .sigmf-data file")
    return Recording(samples, sample_rate, frequency, f"SigMF recording ({datatype})")


def _is_unsafe_path(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return name.startswith(("/", "\\")) or bool(PureWindowsPath(name).drive) or ".." in parts


def _is_macos_metadata(name: str) -> bool:
    """An AppleDouble "._" file or a __MACOSX folder entry, which macOS adds to
    archives to keep Finder information; neither holds samples."""
    parts = PurePosixPath(name).parts
    return bool(parts) and (parts[-1].startswith("._") or parts[0] == "__MACOSX")


def _sigmf_members(members: list[tarfile.TarInfo]) -> tuple[tarfile.TarInfo, tarfile.TarInfo]:
    """The archive's metadata and data members, after checking every entry.

    Member names come from the file, so messages describe entries without
    repeating their names.
    """
    metas: list[tarfile.TarInfo] = []
    datas: list[tarfile.TarInfo] = []
    for member in members:
        if _is_unsafe_path(member.name):
            raise ValueError(
                'The SigMF archive contains an unsafe path: an absolute path or one with "..".'
            )
        if member.isdir():
            continue  # Directories hold no data; nothing is ever extracted to disk.
        if member.isreg() and _is_macos_metadata(member.name):
            continue
        if not member.isreg():
            raise ValueError(
                "The SigMF archive contains a link or special file. Only regular files are read."
            )
        if member.name.lower().endswith(".sigmf-meta"):
            metas.append(member)
        elif member.name.lower().endswith(".sigmf-data"):
            datas.append(member)
    if not metas:
        raise ValueError("The SigMF archive has no .sigmf-meta file.")
    if not datas:
        raise ValueError("The SigMF archive has no .sigmf-data file.")
    if len(metas) > 1 or len(datas) > 1:
        raise ValueError(
            "The SigMF archive holds more than one recording. Upload an archive "
            "with a single recording."
        )
    meta, data = metas[0], datas[0]
    if meta.name[: -len(".sigmf-meta")] != data.name[: -len(".sigmf-data")]:
        raise ValueError(
            "The SigMF archive's .sigmf-meta and .sigmf-data files have different names."
        )
    return meta, data


_DAMAGED_ARCHIVE = "The SigMF archive is truncated or damaged."


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    try:
        stream = archive.extractfile(member)
        content = b"" if stream is None else stream.read()
    except Exception as exc:  # tarfile's own errors describe the file's contents.
        raise ValueError(_DAMAGED_ARCHIVE) from exc
    if len(content) != member.size:
        raise ValueError(_DAMAGED_ARCHIVE)
    return content


def _read_sigmf_archive(data: bytes) -> Recording:
    if len(data) > SIGMF_ARCHIVE_MAX_BYTES:
        raise ValueError(
            f"The SigMF archive exceeds the {format_size(SIGMF_ARCHIVE_MAX_BYTES)} limit. "
            f"Save a shorter recording (at most {MAX_SAMPLES:,} samples)."
        )
    try:
        # Mode "r:" reads uncompressed tar only, entirely from memory.
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        members = archive.getmembers()
    except Exception as exc:
        # Hostile headers make tarfile fail in many ways (OverflowError for huge
        # sizes, ValueError for bad PAX records), and its messages can repeat
        # text from the file, so every failure gets this one message.
        raise ValueError(
            "This .sigmf file could not be read as an uncompressed tar archive. "
            "It may be damaged, truncated, or compressed."
        ) from exc
    with archive:
        # tarfile stops quietly at a cut-off header, so only end-of-archive
        # zero padding may follow the last member it read.
        if archive.offset > len(data) or data[archive.offset :].strip(b"\0"):
            raise ValueError(_DAMAGED_ARCHIVE)
        meta_member, data_member = _sigmf_members(members)
        # Check each size before reading the member into memory.
        if meta_member.size > SIGMF_META_MAX_BYTES:
            raise ValueError(
                f"The SigMF metadata exceeds the {format_size(SIGMF_META_MAX_BYTES)} limit."
            )
        datatype, kind, sample_rate, frequency = _parse_sigmf_meta(
            _read_member(archive, meta_member)
        )
        what = "The archive's .sigmf-data file"
        _check_size(data_member.size, kind, what)
        samples = _decode(_read_member(archive, data_member), kind, what)
    return Recording(samples, sample_rate, frequency, f"SigMF archive ({datatype})")


def load_recording(name: str, data: bytes) -> Recording:
    """Read one file, choosing the reader by its extension (case-insensitive)."""
    extension = _extension(name)
    if extension == "csv":
        return Recording(from_csv(data), None, None, "CSV text (I,Q columns)")
    if extension in _RAW_TYPES:
        kind = _RAW_TYPES[extension]
        samples = _decode(data, kind, f"This .{extension} file")
        return Recording(samples, None, None, f"Raw {kind.description} (.{extension})")
    if extension == "npy":
        return _read_npy(data)
    if extension == "sigmf":
        return _read_sigmf_archive(data)
    if extension in ("sigmf-meta", "sigmf-data"):
        raise ValueError(
            "A SigMF recording needs both files. Upload the .sigmf-meta and "
            ".sigmf-data files together, or a single .sigmf archive."
        )
    # Only a plain extension of the name is repeated.
    if _PLAIN_EXTENSION.fullmatch(extension):
        shown = f"The .{extension} format"
    else:
        shown = "This file's format"
    raise ValueError(
        f"{shown} is not supported. Use a file ending in "
        + ", ".join(f".{item}" for item in SUPPORTED_EXTENSIONS)
        + "."
    )


def select_uploads(
    files: Sequence[tuple[str, Upload]],
) -> tuple[tuple[tuple[str, Upload], ...], int]:
    """The (name, upload) pairs to read and how many earlier ones are ignored.

    Files are given in the order they were added; a multiple-file uploader keeps
    earlier choices when more are added. The last file is read, together with
    the other half of its SigMF pair when it is a .sigmf-meta or .sigmf-data
    file: the most recent one with the same name, or else the most recent one
    of the other kind, which load_uploads then reports as mismatched. Only the
    names are used, so the uploads can be read after choosing.
    """
    if not files:
        return (), 0
    *earlier, last = files
    partner = _SIGMF_PAIR.get(_extension(last[0]))
    selected = [last]
    if partner is not None:
        others = [item for item in earlier if _extension(item[0]) == partner]
        stem = PurePosixPath(last[0]).stem
        matching = [item for item in others if PurePosixPath(item[0]).stem == stem]
        selected = (matching or others)[-1:] + selected
    return tuple(selected), len(files) - len(selected)


def load_uploads(files: Sequence[tuple[str, bytes]]) -> Recording:
    """Read one (name, data) upload, or a matching SigMF metadata and data pair."""
    if len(files) == 1:
        return load_recording(*files[0])
    by_extension = {_extension(name): (name, data) for name, data in files}
    if len(files) == 2 and by_extension.keys() == {"sigmf-meta", "sigmf-data"}:
        meta_name, meta = by_extension["sigmf-meta"]
        data_name, data = by_extension["sigmf-data"]
        if PurePosixPath(meta_name).stem != PurePosixPath(data_name).stem:
            raise ValueError(
                "The .sigmf-meta and .sigmf-data files have different names. "
                "Upload the two files of one recording."
            )
        return load_sigmf_pair(meta, data)
    raise ValueError(
        f"{len(files)} files were uploaded. Remove all but one, or upload a "
        ".sigmf-meta file together with its matching .sigmf-data file."
    )
