#!/usr/bin/env bash
# Run the API and the frontend together for development. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
(cd backend && exec "${PYTHON:-.venv/bin/python}" -m uvicorn app.main:app --reload --port 8000) &
(cd frontend && exec npm run dev) &
trap 'kill 0' INT TERM
wait
