"""Regression tests for the public I/Q CSV interchange contract."""

import csv
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from iq_csv import (
    MAX_SAMPLES,
    MAX_UPLOAD_BYTES,
    SAMPLES_PER_WINDOW,
    format_size,
    from_csv,
    to_csv,
)


# The longest 17-significant-digit float64 text: a sign, 17 digits, a decimal
# point, and a three-digit exponent such as "e-308".
WORST_CASE_COMPONENTS = (
    -1.7976931348623157e308,
    -2.2250738585072014e-308,
    -4.9406564584124654e-324,
    -1.2345678901234567e-100,
)
WORST_COMPONENT_LENGTH = 24


def recording(rows: int = SAMPLES_PER_WINDOW, newline: bytes = b"\n") -> bytes:
    return b"I,Q" + newline + (b"1,2" + newline) * rows


def worst_case_samples(count: int) -> np.ndarray:
    pairs = [complex(i, q) for i in WORST_CASE_COMPONENTS for q in WORST_CASE_COMPONENTS]
    return np.resize(np.array(pairs, dtype=np.complex128), count)


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
        for row in (b"1\n", b"1,2,3\n", b"1,2,\n"):
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

    def test_errors_use_physical_line_numbers_after_blank_lines(self):
        for newline in (b"\n", b"\r\n", b"\r"):
            # Lines 1-7: blank, header, three blank variants, a sample, blank.
            prefix = newline.join((b"", b"I,Q", b"", b"  ", b",", b"1,2", b"", b""))
            for row in (b"bad,0", b"1", b"nan,0", b'"1"0,2'):
                with self.subTest(newline=newline, row=row):
                    with self.assertRaisesRegex(ValueError, "Line 8:"):
                        from_csv(prefix + row + newline)

    def test_blank_lines_between_and_after_rows_are_skipped(self):
        for newline in (b"\n", b"\r\n", b"\r"):
            for blank in (b"", b"   ", b",", b" ,\t"):
                with self.subTest(newline=newline, blank=blank):
                    blank_line = blank + newline
                    data = (
                        b"I,Q" + newline
                        + blank_line
                        + (b"1,2" + newline + blank_line) * SAMPLES_PER_WINDOW
                        + blank_line * 3
                    )
                    np.testing.assert_array_equal(
                        from_csv(data), np.full(SAMPLES_PER_WINDOW, 1 + 2j)
                    )

    def test_blank_lines_before_the_header_are_skipped(self):
        data = b"\xef\xbb\xbf\n \r\n,\n" + recording()
        np.testing.assert_array_equal(from_csv(data), np.full(SAMPLES_PER_WINDOW, 1 + 2j))

    def test_blank_lines_do_not_count_as_samples(self):
        def sparse(rows: int) -> bytes:
            return b"I,Q\n" + b"1,2\n\n" * rows + b"\n" * SAMPLES_PER_WINDOW

        self.assertEqual(len(from_csv(sparse(MAX_SAMPLES))), MAX_SAMPLES)
        with self.assertRaisesRegex(ValueError, "more than"):
            from_csv(sparse(MAX_SAMPLES + 1))
        for rows in (0, SAMPLES_PER_WINDOW - 1):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "at least"):
                    from_csv(sparse(rows))
        for data in (b"\n", b" \n,\n\n"):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "columns named I,Q"):
                    from_csv(data)

    def test_minimum_sample_boundary(self):
        self.assertEqual(len(from_csv(recording())), SAMPLES_PER_WINDOW)
        for rows in (0, SAMPLES_PER_WINDOW - 1):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "at least"):
                    from_csv(recording(rows))

    def test_maximum_sample_boundary(self):
        self.assertEqual(len(from_csv(recording(MAX_SAMPLES))), MAX_SAMPLES)
        with self.assertRaisesRegex(ValueError, f"more than {MAX_SAMPLES:,} samples"):
            from_csv(recording(MAX_SAMPLES + 1))

    def test_no_finite_float64_exports_longer_than_the_worst_case(self):
        for value in WORST_CASE_COMPONENTS:
            with self.subTest(value=value):
                self.assertEqual(len(f"{value:.17g}"), WORST_COMPONENT_LENGTH)
        rng = np.random.default_rng(3)
        values = rng.integers(0, 2**64, size=50_000, dtype=np.uint64).view(np.float64)
        edges = np.array(
            [np.finfo(np.float64).max, np.finfo(np.float64).tiny, np.nextafter(0.0, 1.0),
             1.2345678901234567e-5, 1.2345678901234567e16, 1.2345678901234567e17, 0.0]
        )
        values = np.concatenate([values[np.isfinite(values)], edges])
        longest = max(len(f"{sign * value:.17g}") for value in values for sign in (1, -1))
        self.assertEqual(longest, WORST_COMPONENT_LENGTH)

    def test_worst_case_export_of_maximum_length_fits_and_round_trips(self):
        samples = worst_case_samples(MAX_SAMPLES)
        data = to_csv(samples).encode("utf-8")
        longest_row = max(len(row) for row in data.split(b"\r\n"))
        self.assertEqual(longest_row, 2 * WORST_COMPONENT_LENGTH + 1)
        self.assertLessEqual(len(data), MAX_UPLOAD_BYTES)
        # A byte-order mark added by another program must not push it over.
        restored = from_csv(b"\xef\xbb\xbf" + data)
        np.testing.assert_array_equal(restored.view(np.uint64), samples.view(np.uint64))

    def test_worst_case_export_one_sample_too_long_is_rejected_by_count(self):
        data = to_csv(worst_case_samples(MAX_SAMPLES + 1)).encode("utf-8")
        self.assertLessEqual(len(data), MAX_UPLOAD_BYTES)
        with self.assertRaisesRegex(ValueError, f"more than {MAX_SAMPLES:,} samples"):
            from_csv(data)

    def test_upload_byte_boundary(self):
        # Spread padding across fields to stay below the CSV parser's field limit.
        padding, remainder = divmod(MAX_UPLOAD_BYTES - len(recording()), SAMPLES_PER_WINDOW)
        data = b"I,Q\n" + b"".join(
            b" " * (padding + (index < remainder)) + b"1,2\n"
            for index in range(SAMPLES_PER_WINDOW)
        )
        self.assertEqual(len(data), MAX_UPLOAD_BYTES)
        self.assertEqual(len(from_csv(data)), SAMPLES_PER_WINDOW)
        with self.assertRaisesRegex(ValueError, "exceeds") as caught:
            from_csv(data + b" ")
        message = str(caught.exception)
        self.assertIn(format_size(MAX_UPLOAD_BYTES), message)
        self.assertIn(f"{MAX_SAMPLES:,} samples", message)

    def test_sizes_are_readable_and_never_overstated(self):
        for size, expected in (
            (6_000_000, "6 MB"),
            (5_100_005, "5.1 MB"),
            (5_999_999, "5.9 MB"),
            (1_234_000_000, "1,234 MB"),
            (1_500, "1.5 kB"),
            (999, "999 bytes"),
        ):
            with self.subTest(size=size):
                self.assertEqual(format_size(size), expected)

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
