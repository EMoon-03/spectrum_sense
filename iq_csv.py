"""Small, explicit CSV interchange format for a single I/Q recording."""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np

from signal_config import SAMPLES_PER_WINDOW


MAX_UPLOAD_BYTES = 2_000_000
MAX_SAMPLES = 100_000


def to_csv(samples: np.ndarray) -> str:
    """Export I/Q components with enough digits to round-trip float64 values."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("I", "Q"))
    writer.writerows((f"{sample.real:.17g}", f"{sample.imag:.17g}") for sample in samples)
    return output.getvalue()


def from_csv(data: bytes) -> np.ndarray:
    """Read a recording, reporting invalid input as a user-facing ValueError."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV exceeds 2 MB. Export a shorter recording.")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the CSV as UTF-8 text.") from exc

    reader = csv.reader(StringIO(content, newline=""), strict=True)
    values: list[complex] = []
    try:
        header = next(reader, None)
        if header is None or [part.strip().lower() for part in header] != ["i", "q"]:
            raise ValueError("The CSV must start with two columns named I,Q.")

        for row in reader:
            line_number = reader.line_num
            if len(row) != 2:
                raise ValueError(f"Line {line_number}: expected exactly two numeric columns.")
            try:
                real, imaginary = (float(part) for part in row)
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: I and Q must be numbers.") from exc
            if not np.isfinite(real) or not np.isfinite(imaginary):
                raise ValueError(f"Line {line_number}: values must be finite.")
            values.append(complex(real, imaginary))
            if len(values) > MAX_SAMPLES:
                raise ValueError(f"The CSV has more than {MAX_SAMPLES:,} samples.")
    except csv.Error as exc:
        raise ValueError(f"Line {reader.line_num}: invalid CSV format ({exc}).") from exc

    if len(values) < SAMPLES_PER_WINDOW:
        raise ValueError(f"The CSV needs at least {SAMPLES_PER_WINDOW:,} I/Q rows.")
    return np.asarray(values, dtype=np.complex128)
