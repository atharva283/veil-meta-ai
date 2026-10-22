# Starts the API on :8000 (auto-reload) and the web UI on :5173. Ctrl+C stops both.
$root = Split-Path -Parent $PSScriptRoot
$api = Start-Process -PassThru -NoNewWindow `
    -FilePath "$root\backend\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8000", "--reload", "--reload-dir", "app" `
    -WorkingDirectory "$root\backend"
try {
    Push-Location "$root\frontend"
    Write-Host "Veil: open http://localhost:5173"
    npm run dev
}
finally {
    Pop-Location
    # /T also stops uvicorn's reload worker.
    taskkill /PID $api.Id /T /F | Out-Null
}
