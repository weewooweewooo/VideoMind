$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "[VideoMind] Starting Redis Stack with docker-compose..."
docker-compose up -d redis

Write-Host "[VideoMind] Starting Ollama server in a new PowerShell window..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"

Write-Host "[VideoMind] Waiting 3 seconds for services to be ready..."
Start-Sleep -Seconds 3

Write-Host "[VideoMind] Starting FastAPI with uvicorn..."
uvicorn src.main:app --reload
