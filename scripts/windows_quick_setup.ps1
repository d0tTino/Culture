# Quick setup for Culture.ai on Windows
param(
    [string]$VenvDir = '.venv'
)

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at $VenvDir"
    py -3.10 -m venv $VenvDir
}

& "$VenvDir/Scripts/Activate.ps1"

pip install -r requirements.txt -r requirements-dev.txt

ollama pull mistral:latest

