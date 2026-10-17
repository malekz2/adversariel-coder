@echo off
REM Whole pipeline with a fake model (no weights): checks setup, Pi, sandbox, log.txt, train.png.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" --dry-run --steps 15 %*
pause
