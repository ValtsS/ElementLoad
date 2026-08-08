# Wrapper for elementload that uses the project venv python without activating it.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
    $py = Join-Path $root ".venv\bin\python.exe"
}
if (-not (Test-Path -LiteralPath $py)) {
    Write-Error "elementload: venv python not found. Create it with: python -m venv .venv; .venv\Scripts\python -m pip install -e ."
    exit 1
}
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root;$env:PYTHONPATH" } else { $root }
& $py -m elementload @args
exit $LASTEXITCODE
