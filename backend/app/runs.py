"""Analyzed recordings kept in memory for follow-up reads.

An analysis returns every window's decision and the spectrogram at once, but
a client then asks for one window's plots at a time, or for the CSV, or
re-prepares the same recording with other settings. Sending the recording
again for each of those would cost more than keeping it: a run holds the
loaded recording and the results of the last preparation, under a random id,
for RUN_TTL_SECONDS or until RUN_CACHE_SIZE newer runs push it out.

Nothing is written to disk, and a run id is the only handle. Restarting the
process forgets every run, which the client handles by analyzing again.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from iq_formats import Recording
from prepare import Prepared, PrepareSettings
from spectrogram import Spectrogram
from spectrum_sense import FileClassification

from . import config


@dataclass(eq=False)
class Run:
    run_id: str
    source: str  # "upload" or "example"
    recording: Recording  # as loaded, before preparation
    ignored_uploads: int
    settings: PrepareSettings
    prepared: Prepared
    truncated_to: int | None
    result: FileClassification
    spectrogram: Spectrogram
    loudest: list[tuple[int, float]]
    expires_at: float


class RunStore:
    def __init__(self, size: int, ttl_seconds: int) -> None:
        self._size = size
        self._ttl = ttl_seconds
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def _evict(self, now: float) -> None:
        expired = [key for key, run in self._runs.items() if run.expires_at <= now]
        for key in expired:
            del self._runs[key]
        while len(self._runs) > self._size:
            self._runs.popitem(last=False)

    def put(self, run: Run) -> Run:
        now = time.monotonic()
        run.expires_at = now + self._ttl
        with self._lock:
            self._runs[run.run_id] = run
            self._runs.move_to_end(run.run_id)
            self._evict(now)
        return run

    def get(self, run_id: str) -> Run | None:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            run = self._runs.get(run_id)
            if run is not None:
                run.expires_at = now + self._ttl  # reading keeps a run alive
                self._runs.move_to_end(run_id)
            return run

    def remaining_seconds(self, run: Run) -> int:
        return max(0, int(run.expires_at - time.monotonic()))

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()

    def __len__(self) -> int:
        return len(self._runs)


def new_run_id() -> str:
    return secrets.token_urlsafe(12)


store = RunStore(config.RUN_CACHE_SIZE, config.RUN_TTL_SECONDS)
