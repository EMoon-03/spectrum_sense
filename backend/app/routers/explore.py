"""Simulate one window of noise, BPSK, or QPSK and classify it.

The query string is the same one the Streamlit lab's shareable links use
(?signal=QPSK&snr=3&offset=0.012&seed=7), parsed by explore_url with the
same rules: numbers outside a control's range or between its steps move to
the nearest allowed value, anything else falls back to the default, and
`notes` lists what changed. A client that reads `settings` back and puts
them in its own address bar gets a link that reproduces the example exactly.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request, Response

from explore_url import ExploreSettings, explore_params, parse_explore_params
from iq_csv import to_csv
from signal_config import FREQ_OFFSET_RANGE, SNR_DB_RANGE
from spectrum_sense import generate_capture

from .. import schemas
from ..model import MODEL_VERSION, get_demo_model
from ..views import decision_for, signal_view

router = APIRouter(prefix="/explore", tags=["explore"])


def _settings(request: Request) -> tuple[ExploreSettings, list[str]]:
    params = {key: request.query_params.getlist(key) for key in request.query_params.keys()}
    return parse_explore_params(params)


def _window(settings: ExploreSettings) -> np.ndarray:
    recording, _, _ = generate_capture(
        settings.signal,
        np.random.default_rng(settings.seed),
        windows=1,
        snr_db_override=settings.snr_db,
        freq_offset_override=settings.offset,
    )
    return recording[0]


@router.get("", response_model=schemas.ExploreResponse)
def explore(request: Request):
    """One simulated 1,024-sample window, the model's decision with its
    reasons and limits, and the time, constellation, and spectrum views
    with their text summaries."""
    settings, notes = _settings(request)
    demo = get_demo_model()
    window = _window(settings)
    low, high = SNR_DB_RANGE
    return schemas.ExploreResponse(
        settings=schemas.ExploreSettingsOut(
            signal=settings.signal, snr_db=settings.snr_db, offset=settings.offset, seed=settings.seed
        ),
        notes=notes,
        in_training_range=schemas.TrainingRangeCheck(
            snr=None if settings.signal == "noise" else bool(low <= settings.snr_db <= high),
            offset=bool(FREQ_OFFSET_RANGE[0] <= settings.offset <= FREQ_OFFSET_RANGE[1]),
        ),
        model_version=MODEL_VERSION,
        decision=decision_for(demo.model, window),
        view=signal_view(window),
    )


@router.get("/csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def explore_csv(request: Request):
    """The same window as `I,Q` text, lossless, in the format /analyze reads."""
    settings, _ = _settings(request)
    params = explore_params(settings)
    name = "spectrum_sense_" + "_".join(f"{key}-{value}" for key, value in params.items()) + ".csv"
    return Response(
        to_csv(_window(settings)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
