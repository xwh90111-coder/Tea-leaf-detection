# Start the tea disease detection system on Windows PowerShell.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Starting TeaGuard at http://127.0.0.1:8050 ..." -ForegroundColor Green
python src/app.py
