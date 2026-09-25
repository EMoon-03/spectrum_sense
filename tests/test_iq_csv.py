"""Regression tests for the public I/Q CSV interchange contract."""

import csv
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from iq_csv import MAX_SAMPLES, MAX_UPLOAD_BYTES, SAMPLES_PER_WINDOW, from_csv, to_csv


def recording(rows: int = SAMPLES_PER_WINDOW, newline: bytes = b"\n") -> bytes:
    return b"I,Q" + newline + (b"1,2" + newline) * rows


class IQCSVTests(unittest.TestCase):
    def test_common_newline_formats(self):
        for newline in (b"\n", b"\r\n", b"\r"):
            with self.subTest(newline=newline):
                samples = from_csv(recording(newline=newline))
                self.assertEqual(samples.dtype, np.dtype(np.complex128))
                np.testing.assert_array_equal(samples, np.full(SAMPLES_PER_WINDOW, 1 + 2j))

    def test_utf8_bom_and_normalized_header(self):
        data = b"\xef\xbb\xbf i , q \r\n" + b"1,2\r\n" * SAMPLES_PER_WINDOW
        np.testing.assert_array_equal(from_csv(data), np.full(SAMPLES_PER_WINDOW, 1 + 2j))

    def test_invalid_encoding(self):
        with self.assertRaisesRegex(ValueError, "UTF-8") as caught:
            from_csv(b"I,Q\n\xff,0\n")
        self.assertIsInstance(caught.exception.__cause__, UnicodeDecodeError)

    def test_invalid_headers(self):
        for header in (b"", b"Q,I\n", b"I\n", b"I,Q,other\n", b"real,imaginary\n"):
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, "columns named I,Q"):
                    from_csv(header)

    def test_invalid_column_counts(self):
        for row in (b"\n", b"1\n", b"1,2,3\n"):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "Line 2:.*two numeric columns"):
                    from_csv(b"I,Q\n" + row)

    def test_nonnumeric_values(self):
        for row in (b"wrong,2\n", b"1,wrong\n", b",2\n"):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "Line 2:.*numbers"):
                    from_csv(b"I,Q\n" + row)

    def test_nonfinite_values_in_either_component(self):
        for value in (b"nan", b"inf", b"-inf", b"1e309"):
            for row in (value + b",0\n", b"0," + value + b"\n"):
                with self.subTest(row=row):
                    with self.assertRaisesRegex(ValueError, "Line 2:.*finite"):
                        from_csv(b"I,Q\n" + row)

    def test_malformed_quoting_is_reported_as_value_error(self):
        for data in (
            b'"I"oops,Q\n',
            b'"I,Q\n',
            b'I,Q\n"1"0,2\n' + b"1,2\n" * SAMPLES_PER_WINDOW,
            b'I,Q\n"1,2\n',
        ):
            with self.subTest(data=data[:30]):
                with self.assertRaises(ValueError) as caught:
                    from_csv(data)
                self.assertIsInstance(caught.exception.__cause__, csv.Error)

    def test_oversized_fields_in_header_and_rows(self):
        previous_limit = csv.field_size_limit(128)
        self.addCleanup(csv.field_size_limit, previous_limit)
        oversized = b"1" * 129
        for data in (oversized + b",Q\n", b"I,Q\n" + oversized + b",0\n"):
            with self.subTest(header=not data.startswith(b"I,Q")):
                with self.assertRaises(ValueError) as caught:
                    from_csv(data)
                self.assertIsInstance(caught.exception.__cause__, csv.Error)

    def test_errors_use_physical_line_numbers_after_multiline_fields(self):
        prefix = b'I,Q\n"1\n",2\n'
        for row in (b"bad,0\n", b"1\n", b"nan,0\n", b'"1"0,2\n'):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "Line 4:"):
                    from_csv(prefix + row)

    def test_minimum_sample_boundary(self):
        self.assertEqual(len(from_csv(recording())), SAMPLES_PER_WINDOW)
        for rows in (0, SAMPLES_PER_WINDOW - 1):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "at least"):
                    from_csv(recording(rows))

    def test_maximum_sample_boundary(self):
        self.assertEqual(len(from_csv(recording(MAX_SAMPLES))), MAX_SAMPLES)
        with self.assertRaisesRegex(ValueError, "more than"):
            from_csv(recording(MAX_SAMPLES + 1))

    def test_upload_byte_boundary(self):
        # Spread padding across fields to stay below the CSV parser's field limit.
        padding, remainder = divmod(MAX_UPLOAD_BYTES - len(recording()), SAMPLES_PER_WINDOW)
        data = b"I,Q\n" + b"".join(
            b" " * (padding + (index < remainder)) + b"1,2\n"
            for index in range(SAMPLES_PER_WINDOW)
        )
        self.assertEqual(len(data), MAX_UPLOAD_BYTES)
        self.assertEqual(len(from_csv(data)), SAMPLES_PER_WINDOW)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            from_csv(data + b" ")

    def test_complex128_round_trip_preserves_every_bit(self):
        maximum = np.finfo(np.float64).max
        smallest = np.nextafter(0.0, 1.0)
        examples = np.array(
            [
                complex(1.2345678901234567, -9.876543210987654),
                complex(maximum, -maximum),
                complex(smallest, -smallest),
                complex(1e-250, -1e250),
                complex(-0.0, 0.0),
                complex(0.0, -0.0),
            ],
            dtype=np.complex128,
        )
        samples = np.resize(examples, SAMPLES_PER_WINDOW)
        restored = from_csv(to_csv(samples).encode("utf-8"))
        np.testing.assert_array_equal(restored.view(np.uint64), samples.view(np.uint64))

    def test_standalone_import_does_not_load_training_or_plotting_modules(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import iq_csv; "
                "loaded = {'spectrum_sense', 'sklearn', 'matplotlib'} & sys.modules.keys(); "
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
