"""SpectrumSense API: FastAPI application, CORS, health check.

Run locally with `uvicorn app.main:app --reload` from backend/; the OpenAPI
document is at /docs. The model trains during startup (lifespan), so the
first request never pays for it and /health reports "starting" only while
that is in progress.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, schemas
from .model import MODEL_VERSION, get_demo_model, is_trained
from .routers import analyze, example, explore, model


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_demo_model()
    yield


app = FastAPI(
    title="SpectrumSense API",
    description=(
        "Simulate noise, BPSK, and QPSK windows, classify I/Q recordings with a "
        "synthetic logistic-regression baseline that answers \"uncertain\" when it "
        "should, and read what the model measured about itself."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    # No cookies or auth headers: the API is public and stateless apart from
    # the in-memory run cache, which a run id addresses.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],
)

app.include_router(model.router)
app.include_router(explore.router)
app.include_router(analyze.router)
app.include_router(example.router)


@app.get("/", response_model=schemas.HealthOut, tags=["health"])
@app.get("/health", response_model=schemas.HealthOut, tags=["health"])
def health():
    """Liveness plus whether the model has finished training.

    A health check that returns a constant keeps an instance in rotation
    while it cannot answer; this one reports "starting" until the model is
    ready, and always carries the fingerprint of the model it serves.
    """
    trained = is_trained()
    return schemas.HealthOut(
        status="ok" if trained else "starting",
        service=config.SERVICE_NAME,
        model=schemas.ModelStatus(trained=trained, version=MODEL_VERSION),
    )
