"""Possible causes of an "uncertain" answer, measured from the window itself.

An "uncertain" answer says a window is unlike the training data, not how. A
few properties that no training window has can be measured directly, and
each check here names one, with the number it measured and the margin it was
chosen with. A check never changes a decision, and none names a modulation:
"possible cause" means "something about this window the model never saw".

The limits were set on 1,500 simulated noise, BPSK, and QPSK windows from -2
to 14 dB SNR; tests/test_causes.py repeats those measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from signal_config import UNCERTAIN

# The power check compares this many equal parts of a window.
POWER_PARTS = 8
# Parts of a training window differ in power by at most about 2.2 dB; a
# window that holds part of a burst and part of a gap 6 dB weaker differs by
# 6 dB or more.
POWER_STEP_DB = 3.0
# Samples one symbol apart (extract_features column 5) correlate by at most
# about 0.23 in training windows and by 0.74 or more for a bare carrier at
# 5 dB SNR and above.
CARRIER_CORRELATION = 0.5
# The mean sample of a training window is at most about 0.22 of its RMS
# amplitude; for a carrier at 0 Hz it is about 0.9.
CENTER_COMPONENT = 0.5
_CARRIER_COLUMN = 5


@dataclass(frozen=True)
class Cause:
    key: str  # stable identifier: "power-step", "bare-carrier", or "center-component"
    title: str  # a short phrase
    detail: str  # what was measured, against what training windows show


TITLES = {
    "power-step": "The power changes within the window",
    "bare-carrier": "Like an unmodulated carrier",
    "center-component": "A steady component at 0 Hz",
}


def _scaled(samples: np.ndarray) -> np.ndarray:
    """The samples divided by their largest component, so squaring them can
    neither overflow nor, for the largest, underflow (as extract_features
    does). Each check below is a ratio, so the scale changes no result."""
    samples = np.asarray(samples, dtype=np.complex128)
    if not np.isfinite(samples).all():
        raise ValueError("The I/Q window must contain only finite values.")
    scale = max(float(np.max(np.abs(samples.real), initial=0.0)),
                float(np.max(np.abs(samples.imag), initial=0.0)))
    if scale == 0:
        return samples
    scaled = np.empty_like(samples)
    scaled.real = samples.real / scale
    scaled.imag = samples.imag / scale
    return scaled


def power_step_db(samples: np.ndarray, parts: int = POWER_PARTS) -> float:
    """How far the loudest of `parts` equal parts of a window lies above the
    quietest, in dB; infinite when a part is silent and another is not, and 0
    for an all-zero window. Samples after the last whole part are ignored."""
    samples = _scaled(samples)
    size = len(samples) // parts
    if size == 0:
        raise ValueError(f"Provide at least {parts} samples.")
    powers = np.mean(np.abs(samples[: size * parts].reshape(parts, size)) ** 2, axis=1)
    loudest, quietest = float(powers.max()), float(powers.min())
    if loudest == 0:
        return 0.0
    return float("inf") if quietest == 0 else 10 * np.log10(loudest / quietest)


def center_component(samples: np.ndarray) -> float:
    """The magnitude of the mean sample over the RMS amplitude: 1 for a
    constant, near 0 for anything that averages out over the window."""
    samples = _scaled(samples)
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    return 0.0 if rms == 0 else float(np.abs(np.mean(samples))) / rms


def possible_causes(samples: np.ndarray, features: np.ndarray | None) -> tuple[Cause, ...]:
    """The checks a window fails, in a fixed order. `features` are the
    window's extract_features values, or None when they cannot be computed.
    A window with a NaN or infinite sample has no possible cause: it cannot
    be classified at all."""
    if not np.isfinite(np.asarray(samples, dtype=np.complex128)).all():
        return ()
    found = []
    step = power_step_db(samples)
    if step > POWER_STEP_DB:
        amount = "a silent part" if np.isinf(step) else f"{step:.1f} dB"
        found.append(Cause(
            "power-step", TITLES["power-step"],
            f"The loudest eighth of the window is {amount} above the quietest; in training "
            f"windows they differ by at most about 2 dB. A window that holds part of a burst "
            f"and part of a gap looks like this, so a window inside one burst is a fairer test.",
        ))
    if features is not None and features[_CARRIER_COLUMN] > CARRIER_CORRELATION:
        found.append(Cause(
            "bare-carrier", TITLES["bare-carrier"],
            f"Samples one symbol apart are alike (correlation {features[_CARRIER_COLUMN]:.2f}; "
            f"training windows stay below about 0.25), as for a carrier with no symbols on "
            f"it. None of the trained classes is a bare carrier.",
        ))
    center = center_component(samples)
    if center > CENTER_COMPONENT:
        found.append(Cause(
            "center-component", TITLES["center-component"],
            f"The window's mean sample is {center:.2f} of its RMS amplitude (training windows: "
            f"at most about 0.2): a DC offset from the receiver, or a carrier at the center "
            f"frequency.",
        ))
    return tuple(found)


# How causes_summary counts windows with each cause: (one window, several).
COUNTED = {
    "power-step": ("changes power within the window", "change power within the window"),
    "bare-carrier": ("looks like an unmodulated carrier", "look like an unmodulated carrier"),
    "center-component": ("has a steady component at 0 Hz", "have a steady component at 0 Hz"),
}


def _counted(count: int, key: str) -> str:
    one, several = COUNTED[key]
    return f"{count:,} {one if count == 1 else several}"


def causes_summary(labels: Sequence[str], causes: Sequence[Sequence[str]]) -> str:
    """Describe the causes found among a recording's uncertain windows:
    `causes` holds each window's cause keys, in the same order as `labels`."""
    if len(labels) != len(causes):
        raise ValueError("Provide the causes of each window.")
    uncertain = [keys for label, keys in zip(labels, causes) if label == UNCERTAIN]
    if not uncertain:
        return "No window is uncertain."
    total = len(uncertain)
    windows = f"{total:,} uncertain window" + ("" if total == 1 else "s")
    found = [
        _counted(count, key)
        for key in TITLES
        if (count := sum(key in keys for keys in uncertain))
    ]
    unexplained = sum(not keys for keys in uncertain)
    if not found:
        return (
            f"None of the {windows} shows a measured cause: their features lie outside "
            "what the model learned in ways these checks do not name."
        )
    listed = found[0] if len(found) == 1 else ", ".join(found[:-1]) + f" and {found[-1]}"
    overlap = " (a window can show more than one)" if len(found) > 1 else ""
    rest = (
        "every one shows at least one cause"
        if not unexplained
        else f"{unexplained:,} show{'s' if unexplained == 1 else ''} no measured cause"
    )
    return f"Of the {windows}, {listed}{overlap}; {rest}."
