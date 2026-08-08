@echo off
rem Wrapper for elementload that uses the project venv python without activating it.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%.venv\bin\python.exe"
if not exist "%PY%" (
    echo elementload: venv python not found. Create it with:
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -e .
    exit /b 1
)
if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)
"%PY%" -m elementload %*
exit /b %ERRORLEVEL%
