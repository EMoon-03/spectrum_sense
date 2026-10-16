"""Write the API's OpenAPI document without starting a server.

    python tools/dump_openapi.py openapi.json

The frontend generates its types from this document (`npm run types` reads
it from a running server instead). CI writes it, regenerates the types, and
fails if the committed src/lib/api.generated.ts differs, so a schema change
without regenerated types cannot be merged. Importing the app does not train
the model; training happens at startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="where to write the JSON document")
    args = parser.parse_args()
    args.output.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
