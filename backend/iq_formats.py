"""Load one I/Q recording from CSV, raw SDR samples, NumPy, WAV, or SigMF files.

Only NumPy and the standard library are used, so this module stays light.
Every problem with a file is reported as a plain-language ValueError. A shared
file is untrusted input, so messages never repeat free text from it, such as
names inside an archive or metadata values; the app shows them as Markdown.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import io
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
import struct
import tarfile
from typing import TypeVar

import numpy as np
from numpy.lib import format as npy_format

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, format_size, from_csv
from signal_config import SAMPLES_PER_WINDOW


Upload = TypeVar("Upload")  # Anything that stands for one uploaded file.
# Packed little-endian 24-bit integers, for which NumPy has no type.
_INT24 = "<i3"


@dataclass(frozen=True, eq=False)
class Recording:
    """Complex128 samples plus the metadata a file provides, if any."""

    samples: np.ndarray
    sample_rate: float | None
    center_frequency: float | None
    source_format: str


@dataclass(frozen=True)
class _SampleType:
    """Interleaved I,Q values of one NumPy type (or packed 24-bit integers),
    scaled by (value - offset) / full_scale."""

    name: str
    description: str
    component: str
    offset: float = 0.0
    full_scale: float = 1.0

    @property
    def bytes_per_sample(self) -> int:
        width = 3 if self.component == _INT24 else np.dtype(self.component).itemsize
        return 2 * width


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

# WAV format codes (wFormatTag), and the sample type read for each code and bit depth.
_WAV_PCM = 0x0001
_WAV_FLOAT = 0x0003
_WAV_EXTENSIBLE = 0xFFFE
_WAV_TYPES = {
    (_WAV_PCM, 8): _SampleType(
        "8-bit PCM", "8-bit unsigned integer PCM", "u1", offset=128.0, full_scale=128.0
    ),
    (_WAV_PCM, 16): _SampleType("16-bit PCM", "16-bit integer PCM", "<i2", full_scale=2.0**15),
    (_WAV_PCM, 24): _SampleType("24-bit PCM", "24-bit integer PCM", _INT24, full_scale=2.0**23),
    (_WAV_PCM, 32): _SampleType("32-bit PCM", "32-bit integer PCM", "<i4", full_scale=2.0**31),
    (_WAV_FLOAT, 32): _SampleType("32-bit float", "32-bit float", "<f4"),
    (_WAV_FLOAT, 64): _SampleType("64-bit float", "64-bit float", "<f8"),
}
_WAV_TYPES_HINT = (
    "Save the recording as a two-channel WAV file of 8-, 16-, 24-, or 32-bit integer PCM, "
    "or 32- or 64-bit float."
)
# WAVE_FORMAT_EXTENSIBLE names its format with a GUID: the format code in the
# first two bytes, then these fixed bytes (Microsoft's KSDATAFORMAT_SUBTYPE_*).
_WAV_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_RIFF_HEADER = struct.Struct("<4sI4s")  # "RIFF", size of the rest, "WAVE"
_CHUNK_HEADER = struct.Struct("<4sI")  # ID, size of the contents
# Format code, channels, sample rate, bytes per second, bytes per sample, bits.
_WAV_FORMAT = struct.Struct("<HHIIHH")
# An "auxi" chunk as SpectraVue wrote it: start and stop times as Windows
# SYSTEMTIME values (year, month, weekday, day, hour, minute, second,
# millisecond), then the center frequency in hertz.
_AUXI = struct.Struct("<8H8HI")
_SYSTEMTIME_LIMITS = (65535, 12, 6, 31, 23, 59, 59, 999)

SUPPORTED_EXTENSIONS = ("csv", *_RAW_TYPES, "npy", "wav", "sigmf", "sigmf-meta", "sigmf-data")
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
# Chunks other than the samples, such as text metadata or the time stamps that
# kiwirecorder writes between blocks of samples, may add up to this much.
_WAV_OVERHEAD_BYTES = 1_000_000
WAV_MAX_BYTES = (
    MAX_SAMPLES * max(kind.bytes_per_sample for kind in _WAV_TYPES.values())
    + _WAV_OVERHEAD_BYTES
)
# The largest file any reader accepts; the upload widget must admit it.
MAX_FILE_BYTES = max(
    MAX_UPLOAD_BYTES,
    NPY_MAX_BYTES,
    SIGMF_ARCHIVE_MAX_BYTES,
    WAV_MAX_BYTES,
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


def _components(data: bytes, kind: _SampleType) -> np.ndarray:
    """Every I and Q value in the data as float64, before scaling."""
    if kind.component != _INT24:
        return np.frombuffer(data, dtype=kind.component).astype(np.float64)
    # Put each 3-byte value in the top of a 4-byte integer; shifting it back
    # down keeps its sign.
    packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    wide = np.zeros((len(packed), 4), dtype=np.uint8)
    wide[:, 1:] = packed
    return (wide.view("<i4")[:, 0] >> 8).astype(np.float64)


def _decode(data: bytes, kind: _SampleType, what: str) -> np.ndarray:
    """Interleaved I,Q values to complex128, scaled to about [-1, 1] for integers."""
    _check_size(len(data), kind, what)
    components = _components(data, kind)
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


_TRUNCATED_WAV = (
    "The WAV file is truncated: a chunk is longer than the rest of the file. It may have "
    "been cut short, or copied while it was still being recorded."
)


def _wav_chunks(data: bytes) -> Iterator[tuple[bytes, memoryview]]:
    """The ID and contents of each chunk of a RIFF WAVE file, in file order.

    Each size is checked against the bytes that remain before it is used, and
    an odd-sized chunk is followed by one padding byte. Bytes after the end
    that the RIFF header states, such as a tag that other software appended,
    are ignored; a chunk that runs past the end of the file means that the
    file was cut short.
    """
    if len(data) > WAV_MAX_BYTES:
        raise ValueError(
            f"The WAV file exceeds the {format_size(WAV_MAX_BYTES)} limit. "
            f"Save a shorter recording (at most {MAX_SAMPLES:,} samples)."
        )
    if len(data) < _RIFF_HEADER.size:
        raise ValueError("This file is too short to be a WAV file.")
    form, riff_size, riff_type = _RIFF_HEADER.unpack_from(data)
    if form == b"RIFX":
        raise ValueError(
            "This WAV file is big-endian (RIFX), which is not supported. Save it as a "
            "standard little-endian WAV file."
        )
    if form in (b"RF64", b"BW64"):
        raise ValueError(
            "This is an RF64 WAV file, a variant for recordings over 4 GB, which is not "
            f"supported. Save a shorter recording (at most {MAX_SAMPLES:,} samples) as a "
            "standard WAV file."
        )
    if form != b"RIFF":
        raise ValueError(
            "This is not a WAV file: it does not start with a RIFF header. If it holds raw "
            "samples without a header, give it an extension such as .cs16 or .cf32 instead."
        )
    if riff_type != b"WAVE":
        raise ValueError("This RIFF file does not hold WAVE audio, so it is not a WAV recording.")
    end = min(8 + riff_size, len(data))
    view = memoryview(data)
    position = _RIFF_HEADER.size
    while position + _CHUNK_HEADER.size <= end:
        chunk_id, size = _CHUNK_HEADER.unpack_from(data, position)
        start = position + _CHUNK_HEADER.size
        if size > len(data) - start:
            raise ValueError(_TRUNCATED_WAV)
        yield chunk_id, view[start : start + size]
        position = start + size + size % 2
    # A few bytes are left over when a file shorter than its header says ends
    # inside a chunk header.
    if position < end < 8 + riff_size:
        raise ValueError(_TRUNCATED_WAV)


def _wav_sample_type(fmt: memoryview) -> tuple[_SampleType, int]:
    """The sample type and sample rate that a "fmt " chunk states."""
    if len(fmt) < _WAV_FORMAT.size:
        raise ValueError("The WAV file's format chunk is too short. The file may be damaged.")
    code, channels, rate, _, block_size, bits = _WAV_FORMAT.unpack_from(fmt)
    if channels == 1:
        raise ValueError(
            "This WAV file has one channel, so it looks like ordinary audio, not I/Q. "
            "An I/Q recording has two channels: I (left) and Q (right)."
        )
    if channels != 2:
        raise ValueError(
            f"This WAV file has {channels:,} channels. An I/Q recording has exactly two: "
            "I (left) and Q (right)."
        )
    if code == _WAV_EXTENSIBLE:
        # The GUID follows the extension's size, the valid bits, and the channel mask.
        if len(fmt) >= 40 and bytes(fmt[26:40]) == _WAV_GUID_TAIL:
            code = int.from_bytes(fmt[24:26], "little")
    kind = _WAV_TYPES.get((code, bits))
    if kind is None:
        if code in (_WAV_PCM, _WAV_FLOAT):
            number = "integer" if code == _WAV_PCM else "float"
            raise ValueError(
                f"This WAV file holds {bits:,}-bit {number} samples, which are not "
                f"supported. {_WAV_TYPES_HINT}"
            )
        raise ValueError(
            "This WAV file stores its samples in a compressed or other unsupported format. "
            + _WAV_TYPES_HINT
        )
    if block_size != kind.bytes_per_sample:
        raise ValueError(
            f"The WAV file's format chunk is inconsistent: it gives {block_size:,} bytes per "
            f"I/Q sample, but two channels of {kind.name} need {kind.bytes_per_sample}. "
            "The file may be damaged."
        )
    return kind, rate


def _auxi_center_frequency(auxi: memoryview) -> float | None:
    """The center frequency in an "auxi" chunk, or None if it states none.

    The layout is SpectraVue's, which SDRuno, SDRangel, and other SDR programs
    also write. Other programs use the same chunk name for other contents (SDR
    Console stores XML text), so a chunk whose times are impossible gives no
    frequency, and neither does a frequency of 0.
    """
    if len(auxi) < _AUXI.size:
        return None
    values = _AUXI.unpack_from(auxi)
    if any(value > limit for value, limit in zip(values[:16], _SYSTEMTIME_LIMITS * 2)):
        return None
    return float(values[16]) or None


def _read_wav(data: bytes) -> Recording:
    """A two-channel WAV file: I on the left channel, Q on the right."""
    formats: list[memoryview] = []
    auxi: memoryview | None = None
    # kiwirecorder --kiwi-wav writes many data chunks with time stamps between them.
    data_chunks = 0
    sample_data = bytearray()
    for chunk_id, contents in _wav_chunks(data):
        if chunk_id == b"fmt ":
            formats.append(contents)
        elif chunk_id == b"data":
            data_chunks += 1
            sample_data += contents
        elif chunk_id == b"auxi" and auxi is None:
            auxi = contents
    if not formats:
        raise ValueError("The WAV file has no format chunk, so its sample type is unknown.")
    if len(formats) > 1:
        raise ValueError("The WAV file has more than one format chunk. The file may be damaged.")
    kind, rate = _wav_sample_type(formats[0])
    if not data_chunks:
        raise ValueError("The WAV file has no data chunk, so it holds no samples.")
    what = "The WAV file's sample data"
    # Check the size before the samples are converted to floating point.
    _check_size(len(sample_data), kind, what)
    samples = _decode(bytes(sample_data), kind, what)
    center_frequency = None if auxi is None else _auxi_center_frequency(auxi)
    # The header states the rate in whole hertz, so some recorders round it.
    sample_rate = float(rate) if rate else None
    return Recording(
        samples, sample_rate, center_frequency, f"WAV ({kind.description}, I and Q channels)"
    )


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
    if extension == "wav":
        return _read_wav(data)
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
