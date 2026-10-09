"""The one demo model a process serves, and its fingerprint.

The model is trained once per process from a fixed seed (a fraction of a
second) and never from user data. MODEL_VERSION hashes everything that
determines its answers: the training settings, the shared signal settings,
and the source of the feature, training, and decision code. A fingerprint
that changes when any of those change is what lets a client, or a saved
per-window CSV, say which model produced a result.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from pathlib import Path

import spectrum_sense
import signal_config
from spectrum_sense import DemoModel, build_demo_model

from . import config

_ROOT = Path(__file__).resolve().parents[1]


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for part in (
        f"seed={config.MODEL_SEED}",
        f"captures={config.CAPTURES_PER_CLASS}",
        f"windows={config.WINDOWS_PER_CAPTURE}",
        (_ROOT / "signal_config.py").read_bytes(),
        inspect.getsource(spectrum_sense.extract_features).encode(),
        inspect.getsource(spectrum_sense.train_classifier).encode(),
        inspect.getsource(spectrum_sense.decide).encode(),
        inspect.getsource(spectrum_sense.fit_novelty).encode(),
        inspect.getsource(spectrum_sense.generate_capture).encode(),
        inspect.getsource(spectrum_sense._transmit).encode(),
    ):
        data = part if isinstance(part, bytes) else part.encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()[:16]


MODEL_VERSION = _fingerprint()


@lru_cache(maxsize=1)
def get_demo_model() -> DemoModel:
    """Train, calibrate, and measure the demo model once per process."""
    return build_demo_model(
        seed=config.MODEL_SEED,
        captures_per_class=config.CAPTURES_PER_CLASS,
        windows=config.WINDOWS_PER_CAPTURE,
    )


def is_trained() -> bool:
    """Whether get_demo_model has completed in this process."""
    return get_demo_model.cache_info().currsize > 0


def training_settings() -> dict:
    return {
        "seed": config.MODEL_SEED,
        "captures_per_class": config.CAPTURES_PER_CLASS,
        "windows_per_capture": config.WINDOWS_PER_CAPTURE,
        "classes": list(signal_config.CLASSES),
        "samples_per_symbol": signal_config.SAMPLES_PER_SYMBOL,
        "samples_per_window": signal_config.SAMPLES_PER_WINDOW,
        "snr_db_range": list(signal_config.SNR_DB_RANGE),
        "freq_offset_range": list(signal_config.FREQ_OFFSET_RANGE),
    }
