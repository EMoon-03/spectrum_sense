"""The bundled real recording: credit, license, and what it holds."""

from __future__ import annotations

from fastapi import APIRouter, Response

import real_example

from .. import analysis, schemas

router = APIRouter(prefix="/example", tags=["example"])


@router.get("", response_model=schemas.ExampleInfo)
def example():
    """Credit and description of examples/intelsat37e_qpsk_excerpt.sigmf
    (Daniel Estévez, Zenodo 13371136, CC BY 4.0)."""
    return analysis.example_info()


@router.get("/file", response_class=Response, responses={200: {"content": {"application/x-tar": {}}}})
def example_file():
    """The SigMF archive itself, to download and upload elsewhere. Keep the
    credit (see /example) with it: the recording stays under CC BY 4.0."""
    return Response(
        real_example.EXAMPLE_PATH.read_bytes(),
        media_type="application/x-tar",
        headers={"Content-Disposition": f'attachment; filename="{real_example.EXAMPLE_PATH.name}"'},
    )
