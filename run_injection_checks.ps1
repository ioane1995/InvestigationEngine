<#
 Applies injection-hardening.zip to the project, ACTIVATES THE .venv, then runs the checks.
 Put this file in the project root (next to the .venv folder) and run from PowerShell:

   .\run_injection_checks.ps1                          # unzip + venv + tests + free worst-case eval
   .\run_injection_checks.ps1 -Real                    # ...plus the real-model re-check (uses Groq quota)
   .\run_injection_checks.ps1 -Real -Model "openai/gpt-oss-20b"
   .\run_injection_checks.ps1 -Zip "D:\path\injection-hardening.zip"

 If PowerShell blocks scripts:  Set-ExecutionPolicy -Scope Process Bypass
 The real-model run needs your LLM_BASE_URL / LLM_API_KEY in .env as usual.
#>
param(
    [string]$Zip = "$env:USERPROFILE\Downloads\injection-hardening.zip",
    [switch]$Real,
    [string]$Model = "openai/gpt-oss-20b"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run($what) {
    Write-Host "`n==> $what" -ForegroundColor Cyan
    & ([scriptblock]::Create($what))
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit code $LASTEXITCODE)" }
}

# 1. apply the update
if (Test-Path $Zip) {
    Write-Host "Applying $Zip" -ForegroundColor Cyan
    Expand-Archive -Path $Zip -DestinationPath $PSScriptRoot -Force
} else {
    Write-Host "Zip not found at $Zip - skipping the unzip step (use -Zip <path> if needed)." -ForegroundColor Yellow
}

# 2. open (activate) the virtual environment
$activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw ".venv not found. Create it first:  python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements-dev.txt"
}
. $activate
Write-Host "venv active: $env:VIRTUAL_ENV" -ForegroundColor Green

# 3. checks
Run "python -m pytest -q"
Run "python -m eval.injection_eval --llm obedient --embedder fastembed --tag inj_obedient"

if ($Real) {
    $env:LLM_MODEL = $Model
    Run "python -m eval.injection_eval --llm openai --embedder fastembed --tag inj_20b_v3 --category false_fact,override,role_forgery --pause 20"
}

Write-Host "`nDone. Results: eval\results\inj_obedient.md" -ForegroundColor Green
if ($Real) { Write-Host "              eval\results\inj_20b_v3.md" -ForegroundColor Green }
