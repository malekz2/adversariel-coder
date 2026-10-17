# Start (or continue) training. Extra arguments go to python -m adv.train, e.g.
#   run.bat --resume        run.bat --dry-run --steps 15        run.bat --max-hours 6
$Root  = $PSScriptRoot
$Tools = Join-Path $Root "tools"
$py    = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run setup.bat first." -ForegroundColor Red; exit 1 }
$env:PATH       = "$(Join-Path $Tools 'node');$(Join-Path $Root '.venv\Scripts');$env:PATH"
$env:PYTHONPATH = Join-Path $Root "src"
$env:HF_HOME    = Join-Path $Tools "hf-home"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location $Root
Write-Host "Training. Watch log.txt and train.png in $Root (they refresh every step). Ctrl+C stops safely." -ForegroundColor Cyan
& $py -m adv.train @args
exit $LASTEXITCODE
