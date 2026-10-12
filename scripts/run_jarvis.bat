@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Jarvis environment not found.
  echo Run scripts\setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
  echo.
  echo Jarvis exited with an error. Check logs\jarvis.log for details.
  pause
)
