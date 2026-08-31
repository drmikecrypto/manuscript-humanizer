# Humanize any manuscript file — one command, no flags required.
# Usage:
#   .\humanize.ps1 manuscript.md
#   .\humanize.ps1 manuscript.md output.md
param(
    [Parameter(Position = 0)]
    [string]$InputFile,

    [Parameter(Position = 1)]
    [string]$OutputFile
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$mh = Join-Path $Root ".venv\Scripts\mh.exe"

function Ensure-Environment {
    if (-not (Test-Path $venvPython)) {
        Write-Host "First run: creating virtual environment..." -ForegroundColor Cyan
        python -m venv (Join-Path $Root ".venv")
        & $venvPython -m pip install -q --upgrade pip
        & $venvPython -m pip install -q -e ".[full]"
        Write-Host "Ready." -ForegroundColor Green
    }
    elseif (-not (Test-Path $mh)) {
        & $venvPython -m pip install -q -e ".[full]"
    }
}

if (-not $InputFile) {
    Write-Host ""
    Write-Host "Manuscript Humanizer" -ForegroundColor Cyan
    Write-Host "--------------------"
    Write-Host "Usage:  .\humanize.ps1 <input-file> [output-file]"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\humanize.ps1 manuscript.md"
    Write-Host "  .\humanize.ps1 examples\demo_manuscript.md"
    Write-Host ""
    exit 1
}

if (-not (Test-Path -LiteralPath $InputFile)) {
    Write-Error "Input file not found: $InputFile"
}

Ensure-Environment

$resolvedInput = (Resolve-Path -LiteralPath $InputFile).Path
$mhArgs = @($resolvedInput)
if ($OutputFile) {
    $mhArgs += @("-o", $OutputFile)
}

& $mh @mhArgs
exit $LASTEXITCODE
