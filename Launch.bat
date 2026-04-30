@echo off
setlocal EnableExtensions
title AdaptivePortfolio

REM --- Switch working dir to the script folder (may contain Cyrillic) --------
cd /d "%~dp0"
set "PROJECT_DIR=%CD%"

REM --- Put venv + log in LOCALAPPDATA (ASCII-safe, not synced to cloud) ------
set "APP_HOME=%LOCALAPPDATA%\AdaptivePortfolio"
if not exist "%APP_HOME%" mkdir "%APP_HOME%"
set "VENV_DIR=%APP_HOME%\venv"
set "LOG=%APP_HOME%\install.log"

> "%LOG%" echo ============================================================
>> "%LOG%" echo AdaptivePortfolio launcher log
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Project: %PROJECT_DIR%
>> "%LOG%" echo Venv:    %VENV_DIR%
>> "%LOG%" echo ============================================================

echo ============================================================
echo   AdaptivePortfolio
echo   Project : %PROJECT_DIR%
echo   Venv    : %VENV_DIR%
echo   Log     : %LOG%
echo ============================================================
echo.

REM --- 1. Find Python --------------------------------------------------------
set "PY="
where py >nul 2>nul && for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
if not defined PY where python >nul 2>nul && for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"

if not defined PY (
    echo [ERROR] Python 3.11+ not found. Opening download page.
    start "" "https://www.python.org/downloads/"
    goto :fail
)

echo [1/5] Python: %PY%
>> "%LOG%" echo [1/5] Python: %PY%

REM --- 2. Create virtual environment in ASCII path ---------------------------
if exist "%VENV_DIR%\Scripts\python.exe" goto :have_venv

echo [2/5] Creating virtual environment (one-time, ~30 sec)...
"%PY%" -m venv "%VENV_DIR%" 1>> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] venv creation failed. See log:
    echo         %LOG%
    goto :show_log
)

:have_venv
set "VPY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VPY%" (
    echo [ERROR] %VPY% missing after venv creation.
    goto :show_log
)
>> "%LOG%" echo [2/5] venv OK

REM --- 3. Install dependencies -----------------------------------------------
if exist "%VENV_DIR%\.installed" goto :skip_install

echo [3/5] Upgrading pip...
"%VPY%" -m pip install --upgrade pip wheel setuptools 1>> "%LOG%" 2>&1
if errorlevel 1 goto :show_log

echo [3/5] Installing dependencies (5-15 min on first run)...
echo       Live progress is in the log file.
"%VPY%" -m pip install --prefer-binary --only-binary=:all: -r "%PROJECT_DIR%\requirements.txt" 1>> "%LOG%" 2>&1
if not errorlevel 1 goto :install_done

echo       Binary-only install failed; retrying with source builds allowed...
>> "%LOG%" echo --- retry without only-binary ---
"%VPY%" -m pip install --prefer-binary -r "%PROJECT_DIR%\requirements.txt" 1>> "%LOG%" 2>&1
if errorlevel 1 goto :show_log

:install_done
> "%VENV_DIR%\.installed" echo done
goto :verify

:skip_install
echo [3/5] Dependencies already installed.
>> "%LOG%" echo [3/5] install skipped

:verify
REM --- 4. Verify imports -----------------------------------------------------
echo [4/5] Verifying imports...
"%VPY%" -c "import streamlit, cvxpy, statsmodels, hmmlearn, pandas, numpy; print('OK')" 1>> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] Core library import failed.
    goto :show_log
)
echo       OK
>> "%LOG%" echo [4/5] imports OK

REM --- 5. Launch Streamlit ---------------------------------------------------
echo [5/5] Starting dashboard at http://localhost:8501
echo       Close this window to stop the app.
echo ============================================================
echo.

cd /d "%PROJECT_DIR%"
"%VPY%" -m streamlit run "%PROJECT_DIR%\app\main.py" --server.port 8501 --browser.gatherUsageStats false
>> "%LOG%" echo Streamlit exit: %errorlevel%

echo.
echo Dashboard stopped. Press any key to close.
pause >nul
exit /b 0

:show_log
echo.
echo ---------------- last 40 lines of install.log ----------------
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40"
echo --------------------------------------------------------------
echo Full log: %LOG%

:fail
echo.
echo Press any key to close.
pause >nul
exit /b 1
