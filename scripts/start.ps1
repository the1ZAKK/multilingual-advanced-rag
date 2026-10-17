# Start the existing local installation without reinstalling or overwriting .env.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$python = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Missing .venv. Follow README installation steps first." }
if (-not (Test-Path ".env")) { throw "Missing .env. Copy .env.example once and configure it first." }

function Get-LocalJson($url) {
    try { return Invoke-RestMethod -Uri $url -TimeoutSec 4 } catch { return $null }
}
function Start-ServiceTerminal($moduleArguments, $label) {
    $escapedProject = $project.Replace("'", "''")
    $escapedPython = $python.Replace("'", "''")
    $command = "Set-Location '$escapedProject'; & '$escapedPython' $moduleArguments"
    Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $command) | Out-Null
    Write-Host "Started $label in its own terminal; use Ctrl+C there to stop it."
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $schema = Get-LocalJson "http://127.0.0.1:8000/openapi.json"
    if ($schema.info.title -ne "Multilingual Advanced RAG") { throw "Port 8000 belongs to another service. Nothing was stopped." }
    Write-Host "Reusing the running RAG API. Restart its terminal if you changed code or .env."
} else {
    Start-ServiceTerminal "-m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1" "RAG API"
}

$ollamaStatus = Get-LocalJson "http://127.0.0.1:11434/api/tags"
if (-not $ollamaStatus) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    $localOllama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if ($ollama) { Start-Process $ollama.Source -ArgumentList "serve" | Out-Null }
    elseif (Test-Path $localOllama) { Start-Process $localOllama -ArgumentList "serve" | Out-Null }
    else { Write-Warning "Ollama is not installed; retrieval and source excerpts remain available." }
}

$uiListener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($uiListener) {
    try {
        $health = Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" -UseBasicParsing -TimeoutSec 4
        if ($health.Content.Trim() -ne "ok") { throw "Unexpected health response" }
        Write-Host "Reusing the running Streamlit dashboard."
    } catch { throw "Port 8501 is occupied by an unverified service. Nothing was stopped." }
} else {
    Start-ServiceTerminal "-m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false" "dashboard"
}
Write-Host "Open http://127.0.0.1:8501 once startup completes. No models were downloaded by this launcher."
