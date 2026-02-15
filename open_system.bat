@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo   CloudyShiny System Launcher
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not available in PATH.
  echo Install Python and try again.
  pause
  exit /b 1
)

echo [1/2] Fetching latest sentiment snapshot...
python sentiment_tracker.py
if errorlevel 1 (
  echo [WARN] Data fetch failed. Dashboard will still attempt to start.
)

echo.
echo [2/2] Starting Streamlit dashboard...
start "CloudyShiny Dashboard" cmd /k "cd /d ""%~dp0"" && python -m streamlit run app.py"

timeout /t 3 >nul
start "" http://localhost:8501

echo.
echo System launch command has been sent.
echo Close this window or press any key to exit launcher.
pause >nul

