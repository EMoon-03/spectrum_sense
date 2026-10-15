"""Keep uploads within the limit, and in memory.

Two guarantees the README and the pages make depend on this module:

- An upload over the limit is refused from its Content-Length header,
  before any of it is read. A request without one (chunked) is refused too,
  since its size is unknown until it has been read; browsers always send
  the header with a form.
- Uploaded files are never written to disk. Starlette's form parser keeps a
  file part in memory up to MultiPartParser.spool_max_size (1 MB by
  default) and moves anything larger to a temporary file; raising that
  size above the largest request this middleware lets through keeps every
  accepted upload in memory.
"""

from __future__ import annotations

import json

from starlette.formparsers import MultiPartParser
from starlette.types import ASGIApp, Receive, Scope, Send

from . import config

# Room for the multipart boundaries, part headers, and the settings fields
# around the files; the route itself counts only the files' bytes.
FORM_OVERHEAD_BYTES = 64 * 1024
UPLOAD_PATHS = frozenset({"/analyze"})


def largest_request() -> int:
    return config.MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES


def keep_uploads_in_memory() -> None:
    MultiPartParser.spool_max_size = largest_request()


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in UPLOAD_PATHS:
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        length = headers.get(b"content-length")
        if length is None:
            await _refuse(send, 411, "Send the upload with a Content-Length header.")
            return
        try:
            size = int(length)
        except ValueError:
            await _refuse(send, 400, "The Content-Length header is not a number.")
            return
        if size > largest_request():
            await _refuse(
                send, 413, f"Uploads are limited to {config.MAX_UPLOAD_BYTES:,} bytes in total."
            )
            return
        await self.app(scope, receive, send)


async def _refuse(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ],
    })
    await send({"type": "http.response.body", "body": body})
