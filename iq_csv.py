"""Small, explicit CSV interchange format for a single I/Q recording."""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np

from spectrum_sense import SAMPLES_PER_WINDOW


MAX_UPLOAD_BYTES = 2_000_000
MAX_SAMPLES = 100_000


def to_csv(samples: np.ndarray) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(("I", "Q"))
    writer.writerows((f"{sample.real:.9g}", f"{sample.imag:.9g}") for sample in samples)
    return output.getvalue()


def from_csv(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV exceeds 2 MB. Export a shorter recording.")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the CSV as UTF-8 text.") from exc

    reader = csv.reader(StringIO(content))
    header = next(reader, None)
    if header is None or [part.strip().lower() for part in header] != ["i", "q"]:
        raise ValueError("The CSV must start with two columns named I,Q.")

    values: list[complex] = []
    for line_number, row in enumerate(reader, start=2):
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
            raise ValueError("The CSV has more than 100,000 samples.")

    if len(values) < SAMPLES_PER_WINDOW:
        raise ValueError(f"The CSV needs at least {SAMPLES_PER_WINDOW:,} I/Q rows.")
    return np.asarray(values, dtype=np.complex128)
