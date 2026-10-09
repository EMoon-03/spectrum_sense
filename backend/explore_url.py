"""Settings of the "Explore a signal" tab as URL query parameters.

A shared link is untrusted input. parse_explore_params checks every value,
moves numbers to the nearest value the controls allow, falls back to the
default for anything else, and describes each change in plain language
without repeating the link's text. Only the standard library and
signal_config are used, so this runs and is tested without Streamlit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

from signal_config import CLASSES


@dataclass(frozen=True)
class NumberControl:
    """A numeric control: its URL key, its name in notes, and its range."""

    key: str
    name: str
    minimum: float
    maximum: float
    step: float
    default: float
    decimals: int  # digits after the decimal point that a value can need
    unit: str = ""


SIGNAL_KEY = "signal"
DEFAULT_SIGNAL = "BPSK"
SNR = NumberControl("snr", "SNR", -12.0, 20.0, 1.0, 6.0, 0, " dB")
OFFSET = NumberControl("offset", "carrier offset", -0.03, 0.03, 0.001, 0.0, 3, " cycles/sample")
SEED = NumberControl("seed", "example seed", 0, 1_000_000, 1, 42, 0)
URL_KEYS = (SIGNAL_KEY, SNR.key, OFFSET.key, SEED.key)
# Longer values are rejected without being parsed; no valid value comes close.
MAX_VALUE_LENGTH = 32
# Plain ASCII decimal numbers only: no "nan", "inf", "0x10", "1_000", or
# non-ASCII digits, all of which float() would accept.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


@dataclass(frozen=True)
class ExploreSettings:
    """What the Explore tab simulates; equal settings give the same recording."""

    signal: str = DEFAULT_SIGNAL
    snr_db: float = SNR.default
    offset: float = OFFSET.default
    seed: int = SEED.default


def format_number(value: float, decimals: int) -> str:
    """Fixed-point text without trailing zeros, such as 6, -0.005, or 0."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def snap(value: float, control: NumberControl) -> float:
    """The allowed value nearest to a finite number: clamped to the control's
    range, then rounded to the nearest step (halves round up)."""
    clamped = min(max(value, control.minimum), control.maximum)
    steps = math.floor((clamped - control.minimum) / control.step + 0.5)
    return round(control.minimum + steps * control.step, control.decimals) + 0  # Never -0.0.


def explore_settings(signal: str, snr_db: float, offset: float, seed: int) -> ExploreSettings:
    """Settings from the controls' values, snapped to their steps so that a link
    reproduces exactly the recording on screen."""
    if signal not in CLASSES:
        raise ValueError(f"Unknown signal type: {signal!r}")
    if not all(math.isfinite(value) for value in (snr_db, offset, seed)):
        raise ValueError("The SNR, carrier offset, and seed must be finite numbers.")
    return ExploreSettings(
        signal, float(snap(snr_db, SNR)), float(snap(offset, OFFSET)), int(snap(seed, SEED))
    )


def explore_params(settings: ExploreSettings) -> dict[str, str]:
    """URL query parameters that parse_explore_params reads back unchanged."""
    return {
        SIGNAL_KEY: settings.signal,
        SNR.key: format_number(settings.snr_db, SNR.decimals),
        OFFSET.key: format_number(settings.offset, OFFSET.decimals),
        SEED.key: str(settings.seed),
    }


def _one_value(params: Mapping[str, str | Sequence[str]], key: str) -> tuple[str | None, str]:
    """The key's single value, stripped, and no problem; or None and why the
    value cannot be used, which is "" when the key is absent."""
    if key not in params:
        return None, ""
    values = params[key]
    if not isinstance(values, str):
        values = list(values)
        if len(values) != 1:
            return None, "it appears more than once" if values else ""
        values = values[0]
    text = str(values).strip()
    if not text:
        return None, "it is empty"
    if len(text) > MAX_VALUE_LENGTH:
        return None, "it is too long"
    return text, ""


def _ignored(name: str, problem: str, default: str) -> str:
    return f"The link's {name} was ignored because {problem}, so the default, {default}, is used."


def _number(
    params: Mapping[str, str | Sequence[str]], control: NumberControl, notes: list[str]
) -> float:
    text, problem = _one_value(params, control.key)
    if text is not None and not _NUMBER.fullmatch(text):
        problem = "it is not a number"
    elif text is not None:
        value = float(text)
        if not math.isfinite(value):
            problem = "it is too large"  # Only overflow makes a matched number infinite.
        else:
            used = snap(value, control)
            if not math.isclose(used, value, rel_tol=0, abs_tol=control.step * 1e-6):
                notes.append(
                    f"The link's {control.name} of {value:.12g}{control.unit} was changed to "
                    f"{format_number(used, control.decimals)}{control.unit}, the nearest value "
                    "the control allows."
                )
            return used
    if problem:
        default = format_number(control.default, control.decimals) + control.unit
        notes.append(_ignored(control.name, problem, default))
    return control.default


def parse_explore_params(
    params: Mapping[str, str | Sequence[str]],
) -> tuple[ExploreSettings, list[str]]:
    """Settings from URL query parameters (each a string or a list of strings),
    and one note for each value that was ignored or changed.

    Missing keys take their defaults without a note, and other keys are left
    alone. A signal type matches regardless of case.
    """
    notes: list[str] = []
    signal = DEFAULT_SIGNAL
    text, problem = _one_value(params, SIGNAL_KEY)
    if text is not None:
        known = {name.lower(): name for name in CLASSES}
        if text.lower() in known:
            signal = known[text.lower()]
        else:
            problem = f"it is not {', '.join(CLASSES[:-1])}, or {CLASSES[-1]}"
    if problem:
        notes.append(_ignored("signal type", problem, DEFAULT_SIGNAL))
    snr_db = float(_number(params, SNR, notes))
    offset = float(_number(params, OFFSET, notes))
    seed = int(_number(params, SEED, notes))
    return ExploreSettings(signal, snr_db, offset, seed), notes
