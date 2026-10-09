"""What the demo model measured about itself: the "How the model works" tab."""

from __future__ import annotations

import math

from fastapi import APIRouter

import explore_url
import signal_config

from .. import schemas
from ..model import MODEL_VERSION, get_demo_model, training_settings

router = APIRouter(prefix="/model", tags=["model"])


def _control(control: explore_url.NumberControl) -> schemas.NumberControlOut:
    return schemas.NumberControlOut(
        key=control.key,
        name=control.name,
        minimum=control.minimum,
        maximum=control.maximum,
        step=control.step,
        default=control.default,
        decimals=control.decimals,
        unit=getattr(control, "unit", "") or "",
    )


def explore_controls() -> schemas.ExploreControls:
    return schemas.ExploreControls(
        signals=list(signal_config.CLASSES),
        default_signal=explore_url.DEFAULT_SIGNAL,
        snr=_control(explore_url.SNR),
        offset=_control(explore_url.OFFSET),
        seed=_control(explore_url.SEED),
    )


def _optional(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


@router.get("", response_model=schemas.ModelReport)
def model_report():
    """Training settings, the limits behind "uncertain", held-out accuracy,
    false-reject rates by SNR band, and how often each probe signal (never
    trained on) is flagged. Nothing here was tuned on the held-out data."""
    demo = get_demo_model()
    return schemas.ModelReport(
        version=MODEL_VERSION,
        training=schemas.TrainingSettings(**training_settings()),
        limits=schemas.UncertainLimits(
            distance_threshold=demo.model.distance_threshold,
            min_top_score=demo.model.min_top_score,
            distance_percentile=signal_config.DISTANCE_PERCENTILE,
            calibration_fraction=signal_config.CALIBRATION_FRACTION,
            covariance_shrinkage=signal_config.COVARIANCE_SHRINKAGE,
        ),
        controls=explore_controls(),
        fit_captures=demo.fit_captures,
        calibration_captures=demo.calibration_captures,
        test_captures=demo.test_captures,
        test_windows=demo.test_windows,
        test_accuracy=demo.test_accuracy,
        false_reject_rate=demo.false_reject_rate,
        accepted_accuracy=_optional(demo.accepted_accuracy),
        rates_by_snr=[
            schemas.GroupRatesOut(
                group=rates.group,
                windows=rates.windows,
                false_reject_rate=rates.false_reject_rate,
                accepted_accuracy=_optional(rates.accepted_accuracy),
            )
            for rates in demo.rates_by_snr
        ],
        probe_kinds=list(signal_config.PROBE_KINDS),
        probe_windows=demo.probe_windows,
        probe_flag_rates=[
            schemas.ProbeRates(snr_db=snr_db, rates=dict(rates))
            for snr_db, rates in sorted(demo.probe_flag_rates.items())
        ],
    )
