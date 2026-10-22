# One-time setup on Windows: Python venv, frontend packages, demo clips.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$root\backend\.venv")) {
    python -m venv "$root\backend\.venv"
}
& "$root\backend\.venv\Scripts\python.exe" -m pip install -q --upgrade pip
& "$root\backend\.venv\Scripts\python.exe" -m pip install -q -r "$root\backend\requirements-dev.txt"

Push-Location "$root\frontend"
npm install --no-audit --no-fund
Pop-Location

if (-not (Test-Path "$root\demo_assets\clips\street_crossing.mp4")) {
    & "$root\backend\.venv\Scripts\python.exe" "$root\demo_assets\generate_demo_clips.py"
}
if (-not (Test-Path "$root\.env")) {
    Copy-Item "$root\.env.example" "$root\.env"
}
Write-Host "Setup done. Run scripts\dev.ps1 to start Veil."
