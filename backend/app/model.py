"""The one demo model a process serves, and its fingerprint.

The model is trained once per process from a fixed seed (a fraction of a
second) and never from user data. MODEL_VERSION hashes everything that
determines its answers: the training settings, the shared signal settings,
the source of every spectrum_sense function and class that generates,
measures, trains, or decides (FINGERPRINTED), and the versions of Python and
of the numerical libraries. A fingerprint that changes when any of those
change is what lets a client, or a saved per-window CSV, say which model
produced a result.

tests/test_model_version.py fails when spectrum_sense gains a function that
is in neither FINGERPRINTED nor NOT_FINGERPRINTED, so a new piece of decision
code cannot silently escape the fingerprint.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
from functools import lru_cache
from pathlib import Path

import numpy
import scipy
import sklearn

import spectrum_sense
import signal_config
from spectrum_sense import DemoModel, build_demo_model

from . import config

_ROOT = Path(__file__).resolve().parents[1]

# Everything in spectrum_sense that changes an answer, or a number /model
# reports about the model.
FINGERPRINTED = (
    spectrum_sense.complex_gaussian,
    spectrum_sense.generate_capture,
    spectrum_sense._transmit,
    spectrum_sense.generate_probe_capture,
    spectrum_sense.line_strength,
    spectrum_sense.extract_features,
    spectrum_sense.make_dataset,
    spectrum_sense.NoveltyModel,
    spectrum_sense.fit_novelty,
    spectrum_sense.novelty_distance,
    spectrum_sense.Decision,
    spectrum_sense.decide,
    spectrum_sense.SignalClassifier,
    spectrum_sense.classify_features,
    spectrum_sense.FeatureEvidence,
    spectrum_sense.explain_features,
    spectrum_sense.train_classifier,
    spectrum_sense.uncertain_rates,
    spectrum_sense.probe_flag_rates,
    spectrum_sense.GroupRates,
    spectrum_sense.rates_by_snr,
    spectrum_sense.DemoModel,
    spectrum_sense.build_demo_model,
    spectrum_sense.FileClassification,
    spectrum_sense.classify_recording,
)

# The rest of spectrum_sense, and why leaving it out cannot change an answer.
NOT_FINGERPRINTED = {
    "_display_classes": "orders classes for display only",
    "decision_order": "orders decisions for display only",
    "decision_counts": "counts decisions that are already made",
    "results_csv": "formats decisions that are already made",
    "plot_examples": "command-line figures only",
    "main": "the command-line baseline, not the app's model",
}


def _library_versions() -> str:
    python = ".".join(platform.python_version_tuple()[:2])
    return (
        f"python={python} numpy={numpy.__version__} "
        f"scipy={scipy.__version__} scikit-learn={sklearn.__version__}"
    )


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for part in (
        f"seed={config.MODEL_SEED}",
        f"captures={config.CAPTURES_PER_CLASS}",
        f"windows={config.WINDOWS_PER_CAPTURE}",
        _library_versions(),
        (_ROOT / "signal_config.py").read_bytes(),
        *(inspect.getsource(item).encode() for item in FINGERPRINTED),
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
