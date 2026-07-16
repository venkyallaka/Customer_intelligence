$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
$codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $python311) {
    $python = $python311
} elseif (Test-Path $codexPython) {
    $python = $codexPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = (Get-Command python -ErrorAction Stop).Source
} else {
    throw "Python was not found. Install Python 3.11+ or run this project from Codex/VS Code where the bundled Python runtime is available."
}

New-Item -ItemType Directory -Force -Path (Join-Path $project "outputs") | Out-Null

Write-Host "Starting chatbot at http://127.0.0.1:8000"
Write-Host "Keep this PowerShell window open while using the site. Press Ctrl+C to stop it."
& $python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
