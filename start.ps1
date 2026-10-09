# start.ps1
#
# Launches the SpectrumSense backend (FastAPI) and frontend (Next.js) each
# in their own PowerShell window.
#
# Usage, from the repository root:
#   .\start.ps1
#
# If PowerShell blocks it with an execution-policy error, run this once
# first (only affects the current window):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

$root = $PSScriptRoot

Write-Host "Starting SpectrumSense API (http://localhost:8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root\backend'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"
)

Start-Sleep -Seconds 2

Write-Host "Starting SpectrumSense frontend (http://localhost:3000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root\frontend'; npm run dev"
)

Write-Host ""
Write-Host "  API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  Site:     http://localhost:3000" -ForegroundColor Green
