"""Settings from the environment, with defaults that need nothing set.

Local and deployed differ by environment variables only; see .env.example
and render.yaml.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from iq_csv import MAX_SAMPLES as _MAX_SAMPLES

load_dotenv()


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, not {value!r}") from exc


# Browsers reject a wildcard origin combined with credentials, and the API
# is public and read-only, so the list is explicit and credentials are off.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# Training settings of the demo model. Changing any of them changes every
# answer, which is why they are part of the model fingerprint.
MODEL_SEED = _int("MODEL_SEED", 42)
CAPTURES_PER_CLASS = _int("CAPTURES_PER_CLASS", 100)
WINDOWS_PER_CAPTURE = _int("WINDOWS_PER_CAPTURE", 4)

# How many analyzed recordings one process keeps in memory for follow-up
# reads (a window's plots, the CSV). Samples are never written to disk.
RUN_CACHE_SIZE = _int("RUN_CACHE_SIZE", 32)
RUN_TTL_SECONDS = _int("RUN_TTL_SECONDS", 30 * 60)

# Upload limits. The CSV reader has its own byte limit (iq_csv.MAX_UPLOAD_BYTES);
# this is the largest request body the API accepts for any format.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 8_000_000)
MAX_SAMPLES = _MAX_SAMPLES

SERVICE_NAME = "spectrum-sense-api"
