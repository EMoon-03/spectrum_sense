"""Tests for reading and writing the Explore tab's settings in the URL."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from explore_url import (
    DEFAULT_SIGNAL,
    MAX_VALUE_LENGTH,
    OFFSET,
    SEED,
    SNR,
    URL_KEYS,
    ExploreSettings,
    NumberControl,
    explore_params,
    explore_settings,
    format_number,
    parse_explore_params,
    snap,
)
from signal_config import CLASSES
from spectrum_sense import generate_capture


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ExploreSettings()
CONTROLS = (SNR, OFFSET, SEED)
FIELDS = {SNR.key: "snr_db", OFFSET.key: "offset", SEED.key: "seed"}


def allowed_values(control: NumberControl) -> list[float]:
    """Every value the control can take."""
    count = round((control.maximum - control.minimum) / control.step)
    return [snap(control.minimum + index * control.step, control) for index in range(count + 1)]


def with_value(control: NumberControl, value: float) -> ExploreSettings:
    return replace(DEFAULTS, **{FIELDS[control.key]: value})


class ParseTests(unittest.TestCase):
    def parse_one(self, params, expected: ExploreSettings) -> str:
        """Parse params that should produce exactly one note, and return it."""
        settings, notes = parse_explore_params(params)
        self.assertEqual(settings, expected, params)
        self.assertEqual(len(notes), 1, notes)
        return notes[0]

    def test_no_parameters_give_the_defaults_without_notes(self):
        self.assertEqual(parse_explore_params({}), (DEFAULTS, []))
        self.assertEqual(DEFAULTS, ExploreSettings("BPSK", 6.0, 0.0, 42))
        for control in CONTROLS:
            with self.subTest(control=control.key):
                self.assertLessEqual(control.minimum, control.default)
                self.assertLessEqual(control.default, control.maximum)
                self.assertEqual(snap(control.default, control), control.default)

    def test_every_allowed_setting_round_trips_without_notes(self):
        for signal in CLASSES:
            for snr in allowed_values(SNR):
                for offset in allowed_values(OFFSET):
                    for seed in (SEED.minimum, SEED.default, SEED.maximum):
                        settings = ExploreSettings(signal, snr, offset, seed)
                        self.assertEqual(
                            parse_explore_params(explore_params(settings)), (settings, [])
                        )

    def test_values_have_the_types_the_controls_need(self):
        settings, _ = parse_explore_params({"snr": "3", "offset": "0", "seed": "42.0"})
        self.assertIs(type(settings.snr_db), float)
        self.assertIs(type(settings.offset), float)
        self.assertIs(type(settings.seed), int)
        self.assertEqual(settings.seed, 42)

    def test_signal_type_ignores_case_and_surrounding_spaces(self):
        for text, expected in (("qpsk", "QPSK"), (" Noise ", "noise"), ("bPsK", "BPSK")):
            with self.subTest(text=text):
                settings, notes = parse_explore_params({"signal": text})
                self.assertEqual((settings.signal, notes), (expected, []))

    def test_unknown_or_ambiguous_signal_type_uses_the_default(self):
        for value in ("8PSK", "", "   ", "BPSK,QPSK", ["BPSK", "QPSK"], "x" * 1000):
            with self.subTest(value=value[:20]):
                note = self.parse_one({"signal": value}, DEFAULTS)
                self.assertTrue(note.startswith("The link's signal type was ignored because"))
                self.assertTrue(note.endswith(f"so the default, {DEFAULT_SIGNAL}, is used."))
        note = self.parse_one({"signal": "8PSK"}, DEFAULTS)
        self.assertIn("it is not noise, BPSK, or QPSK", note)

    def test_text_that_is_not_a_plain_finite_number_uses_the_default(self):
        for text in (
            "abc", "nan", "NaN", "inf", "-Infinity", "1_000", "0x10", "٤٢", "1e",
            "--1", "1.2.3", "6 dB", "", " ", "1e999", "-1e999",
        ):
            for control in CONTROLS:
                with self.subTest(text=text, control=control.key):
                    note = self.parse_one({control.key: text}, DEFAULTS)
                    default = format_number(control.default, control.decimals) + control.unit
                    self.assertTrue(note.startswith(f"The link's {control.name} was ignored"))
                    self.assertTrue(note.endswith(f"so the default, {default}, is used."))

    def test_out_of_range_values_move_to_the_nearest_end(self):
        for control, text, expected, shown in (
            (SNR, "50", 20.0, "20 dB"),
            (SNR, "-100", -12.0, "-12 dB"),
            (OFFSET, "0.5", 0.03, "0.03 cycles/sample"),
            (OFFSET, "-1e300", -0.03, "-0.03 cycles/sample"),
            (SEED, "-3", 0, "0"),
            (SEED, "2000000", 1_000_000, "1000000"),
            (SEED, "1e300", 1_000_000, "1000000"),
        ):
            with self.subTest(control=control.key, text=text):
                note = self.parse_one({control.key: text}, with_value(control, expected))
                self.assertIn(f"was changed to {shown}, the nearest value the control allows", note)

    def test_values_between_steps_move_to_the_nearest_step(self):
        for control, text, expected in (
            (SNR, "6.4", 6.0),
            (SNR, "6.5", 7.0),
            (SNR, "-3.5", -3.0),
            (OFFSET, "0.0014", 0.001),
            (OFFSET, "-0.0126", -0.013),
            (SEED, "42.5", 43),
            (SEED, "41.4", 41),
        ):
            with self.subTest(control=control.key, text=text):
                note = self.parse_one({control.key: text}, with_value(control, expected))
                self.assertIn(f"The link's {control.name} of {float(text):g}", note)

    def test_a_note_shows_enough_digits_to_differ_from_the_value_used(self):
        note = self.parse_one({"offset": "0.0120000012"}, with_value(OFFSET, 0.012))
        self.assertIn("of 0.0120000012 cycles/sample was changed to 0.012 cycles/sample", note)

    def test_other_spellings_of_allowed_values_need_no_note(self):
        for control, text, expected in (
            (SNR, "6.0", 6.0),
            (SNR, "+6", 6.0),
            (SNR, " -3e0 ", -3.0),
            (OFFSET, "0.0050", 0.005),
            (OFFSET, ".005", 0.005),
            (OFFSET, "-5E-3", -0.005),
            (OFFSET, "-0", 0.0),
            (SEED, "4.2e1", 42),
            (SEED, "0007", 7),
        ):
            with self.subTest(control=control.key, text=text):
                self.assertEqual(
                    parse_explore_params({control.key: text}), (with_value(control, expected), [])
                )

    def test_repeated_keys_are_ignored_and_single_item_lists_are_accepted(self):
        for control in CONTROLS:
            with self.subTest(control=control.key):
                note = self.parse_one({control.key: ["3", "3"]}, DEFAULTS)
                self.assertIn("because it appears more than once", note)
                self.assertEqual(
                    parse_explore_params({control.key: ["3"]}),
                    parse_explore_params({control.key: "3"}),
                )
                self.assertEqual(parse_explore_params({control.key: []}), (DEFAULTS, []))

    def test_overlong_values_are_rejected_before_parsing(self):
        self.assertEqual(
            parse_explore_params({"seed": "0" * (MAX_VALUE_LENGTH - 1) + "7"})[0].seed, 7
        )
        for key in URL_KEYS:
            with self.subTest(key=key):
                note = self.parse_one({key: "1" * (MAX_VALUE_LENGTH + 1)}, DEFAULTS)
                self.assertIn("because it is too long", note)

    def test_notes_never_repeat_the_links_text(self):
        hostile = "[here](https://example.com)<b>"
        for value in (hostile, "1" + hostile, "1e1" + hostile[:10]):
            params = {key: value for key in URL_KEYS}
            settings, notes = parse_explore_params(params)
            self.assertEqual(settings, DEFAULTS)
            self.assertEqual(len(notes), len(URL_KEYS))
            for note in notes:
                with self.subTest(value=value, note=note):
                    for fragment in ("example.com", "[", "<", "here"):
                        self.assertNotIn(fragment, note)

    def test_other_keys_are_left_alone(self):
        params = {"utm_source": "mail", "embed": "true", "SNR": "3", "Signal": "noise"}
        self.assertEqual(parse_explore_params(params), (DEFAULTS, []))

    def test_notes_follow_the_order_of_the_controls(self):
        _, notes = parse_explore_params({"seed": "x", "offset": "x", "snr": "x", "signal": "x"})
        names = ["signal type", *(control.name for control in CONTROLS)]
        self.assertEqual(
            [note.split(" was ")[0] for note in notes], [f"The link's {name}" for name in names]
        )


class SettingsTests(unittest.TestCase):
    def test_control_values_snap_to_the_steps_in_the_url(self):
        settings = explore_settings("QPSK", 3.0000000000000004, 0.005000000000000001, 7)
        self.assertEqual(settings, ExploreSettings("QPSK", 3.0, 0.005, 7))
        self.assertEqual(
            explore_params(settings),
            {"signal": "QPSK", "snr": "3", "offset": "0.005", "seed": "7"},
        )
        self.assertEqual(tuple(explore_params(settings)), URL_KEYS)

    def test_url_numbers_are_short_and_never_negative_zero(self):
        for settings, snr, offset in (
            (ExploreSettings(snr_db=-12.0, offset=-0.0), "-12", "0"),
            (ExploreSettings(snr_db=20.0, offset=-0.03), "20", "-0.03"),
            (ExploreSettings(offset=0.01), "6", "0.01"),
        ):
            with self.subTest(settings=settings):
                params = explore_params(settings)
                self.assertEqual((params["snr"], params["offset"]), (snr, offset))
        self.assertEqual(format_number(-0.0001, 3), "0")
        self.assertEqual(format_number(0.03, 3), "0.03")
        self.assertEqual(format_number(-7.0, 0), "-7")
        self.assertEqual(str(snap(-0.0001, OFFSET)), "0.0")

    def test_unknown_signal_or_non_finite_control_value_is_an_error(self):
        for arguments in (
            ("8PSK", 6.0, 0.0, 42),
            ("BPSK", float("nan"), 0.0, 42),
            ("BPSK", 6.0, float("inf"), 42),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    explore_settings(*arguments)

    def test_a_link_reproduces_the_same_recording(self):
        shown = explore_settings("QPSK", 3.0, 0.012000000000000002, 7)
        opened, notes = parse_explore_params(explore_params(shown))
        self.assertEqual(notes, [])
        recordings = [
            generate_capture(
                settings.signal, np.random.default_rng(settings.seed), windows=1,
                snr_db_override=settings.snr_db, freq_offset_override=settings.offset,
            )[0]
            for settings in (shown, opened)
        ]
        np.testing.assert_array_equal(*recordings)


class ImportTests(unittest.TestCase):
    def test_parser_loads_neither_streamlit_nor_numpy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, explore_url; "
                "assert not {'streamlit', 'numpy'} & set(sys.modules), 'heavy import'",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
