$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== Jarvis Windows setup ===" -ForegroundColor Cyan
$PythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($PythonCommand) {
    & py -3.11 --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = "py"
        $PythonArgs = @("-3.11")
    } else {
        $PythonExe = "python"
        $PythonArgs = @()
    }
} else {
    $PythonExe = "python"
    $PythonArgs = @()
}
& $PythonExe @PythonArgs --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Install it from https://www.python.org/downloads/ and enable the Python launcher."
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $PythonExe @PythonArgs -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the error above." }

$ModelsRoot = Join-Path $Root "models"
$ModelDir = Join-Path $ModelsRoot "vosk-model-small-en-us-0.15"
$ModelMarker = Join-Path $ModelDir "am"
if (-not (Test-Path $ModelMarker)) {
    New-Item -ItemType Directory -Force -Path $ModelsRoot | Out-Null
    $ZipPath = Join-Path $env:TEMP "vosk-model-small-en-us-0.15.zip"
    $ModelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    Write-Host "Downloading the offline English speech model (~40 MB)..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ZipPath
    Expand-Archive -Path $ZipPath -DestinationPath $ModelsRoot -Force
    Remove-Item $ZipPath -Force
    if (-not (Test-Path $ModelMarker)) {
        $Nested = Join-Path $ModelDir "vosk-model-small-en-us-0.15"
        if (Test-Path (Join-Path $Nested "am")) {
            Get-ChildItem $Nested -Force | Move-Item -Destination $ModelDir -Force
            Remove-Item $Nested -Force
        }
    }
    if (-not (Test-Path $ModelMarker)) {
        throw "Vosk download/extraction completed, but the model folder is invalid. Extract it manually under models\."
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "data"), (Join-Path $Root "logs") | Out-Null
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start Jarvis with: .\.venv\Scripts\python.exe main.py"
Write-Host "For local LLM support, install Ollama, then run: ollama pull qwen2.5:3b"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama was not detected. Jarvis can still start with its PyTorch intent model; local LLM features need Ollama." -ForegroundColor Yellow
}
