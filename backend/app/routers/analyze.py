"""Analyze a recording, then read parts of the result from its run.

    POST /analyze              multipart: one or more files + optional settings
    POST /analyze/example      JSON settings; analyzes the bundled recording
    GET  /runs/{id}            the analysis again (after a reload, say)
    POST /runs/{id}/prepare    other settings for the same recording
    GET  /runs/{id}/windows/n  one window's decision and plots
    GET  /runs/{id}/results.csv

Errors: 413 when a body is larger than the limit (app/uploads.py refuses one
whose Content-Length is too large before reading it, and 411 one without a
Content-Length), 422 for a file the readers
reject or settings prepare.py rejects (the detail is the reader's own
plain-language message, which never repeats text from the file), and 404 for
a run this process no longer holds, which the client answers by analyzing
again.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from prepare import PrepareSettings

from .. import analysis, config, schemas
from ..model import get_demo_model
from ..runs import store

router = APIRouter(tags=["analyze"])

# Each error response states its own description. Left out, FastAPI uses
# Python's status phrase, which Python 3.13 renamed for 413 and 422, so the
# OpenAPI document, and the frontend types generated from it, would depend on
# the Python version that wrote it.
ERRORS = {
    404: {"model": schemas.ErrorOut, "description": "The run, or that window of it, is not available"},
    411: {"model": schemas.ErrorOut, "description": "The upload has no Content-Length header"},
    413: {"model": schemas.ErrorOut, "description": "The upload is larger than the limit"},
    422: {"model": schemas.ErrorOut, "description": "A file or setting the readers reject, in their own words"},
}


def _errors(*codes: int) -> dict:
    return {code: ERRORS[code] for code in codes}


def _settings(body: schemas.PrepareSettingsIn) -> PrepareSettings:
    return PrepareSettings(
        offset=body.offset, bandwidth=body.bandwidth, symbol_rate=body.symbol_rate, factor=body.factor
    )


def _run_or_404(run_id: str):
    run = store.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="This analysis is no longer available. Analyze the recording again.",
        )
    return run


def _analyze(files, settings: PrepareSettings, source: str) -> schemas.AnalyzeResponse:
    demo = get_demo_model()
    try:
        run = analysis.analyze(demo, files, settings, source=source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return analysis.to_response(demo, run)


@router.post("/analyze", response_model=schemas.AnalyzeResponse, responses=_errors(411, 413, 422))
async def analyze_upload(
    files: Annotated[list[UploadFile], File(description="One recording, or a .sigmf-meta and .sigmf-data pair")],
    offset: Annotated[float, Form()] = 0.0,
    bandwidth: Annotated[float, Form()] = 0.0,
    symbol_rate: Annotated[float, Form()] = 0.0,
    factor: Annotated[float, Form()] = 1.0,
):
    """Read the most recently added file (with its SigMF partner if any),
    prepare it with the optional settings, classify every complete
    1,024-sample window, and compute the spectrogram."""
    loaded: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        data = await upload.read()
        total += len(data)
        if total > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploads are limited to {config.MAX_UPLOAD_BYTES:,} bytes in total.",
            )
        loaded.append((upload.filename or "", data))
    settings = _settings(
        schemas.PrepareSettingsIn(offset=offset, bandwidth=bandwidth, symbol_rate=symbol_rate, factor=factor)
    )
    # Reading the uploads needs the event loop; the analysis is CPU work, so it
    # runs in a worker thread and other requests are answered meanwhile.
    return await run_in_threadpool(_analyze, tuple(loaded), settings, "upload")


@router.post("/analyze/example", response_model=schemas.AnalyzeResponse, responses=_errors(422))
def analyze_example(body: schemas.PrepareSettingsIn | None = None):
    """Analyze the bundled real recording (see /example) exactly as an upload."""
    return _analyze(analysis.load_example(), _settings(body or schemas.PrepareSettingsIn()), "example")


@router.get("/runs/{run_id}", response_model=schemas.AnalyzeResponse, responses=_errors(404))
def get_run(run_id: str):
    return analysis.to_response(get_demo_model(), _run_or_404(run_id))


@router.post("/runs/{run_id}/prepare", response_model=schemas.AnalyzeResponse, responses=_errors(404, 422))
def prepare_run(run_id: str, body: schemas.PrepareSettingsIn):
    """Prepare the same recording with other settings, without sending it again."""
    run = _run_or_404(run_id)
    demo = get_demo_model()
    try:
        run = analysis.reprepare(demo, run, _settings(body))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return analysis.to_response(demo, run)


@router.get("/runs/{run_id}/windows/{number}", response_model=schemas.WindowView, responses=_errors(404))
def get_window(run_id: str, number: int):
    """One window of the prepared recording: its level, decision, and plots."""
    run = _run_or_404(run_id)
    try:
        return analysis.window_view(get_demo_model(), run, number)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/results.csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}, **_errors(404)})
def get_csv(run_id: str):
    """Per-window results, with the preparation and the model fingerprint."""
    run = _run_or_404(run_id)
    return Response(
        analysis.csv_for(run),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="spectrum_sense_windows.csv"'},
    )
