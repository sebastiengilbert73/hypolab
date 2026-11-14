@echo off
setlocal
set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment not found in %VENV_DIR%.
    exit /b 1
)

if "%1"=="ui" (
    "%VENV_DIR%\Scripts\streamlit.exe" run "%~dp0app\main.py"
) else (
    "%VENV_DIR%\Scripts\hypolab.exe" %*
)
