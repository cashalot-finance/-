@echo off
setlocal

REM Run from the folder where this .bat is located
cd /d "%~dp0"

REM Force UTF-8 in console and Python to avoid UnicodeEncodeError on emojis
chcp 65001 >nul
set PYTHONUTF8=1

set "SCRIPT=parser.py"
set "LOG=run.log"

> "%LOG%" echo ===== Run started %DATE% %TIME% =====
>>"%LOG%" echo Folder: %CD%
>>"%LOG%" echo Script: %SCRIPT%
>>"%LOG%" echo.

echo [INFO] Folder: %CD%
echo [INFO] Log: %LOG%

if not exist "%SCRIPT%" (
  echo [ERROR] %SCRIPT% not found in %CD%
  >>"%LOG%" echo [ERROR] %SCRIPT% not found in %CD%
  pause
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher "py" not found. Install Python from python.org
  >>"%LOG%" echo [ERROR] Python Launcher "py" not found.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating venv...
  >>"%LOG%" echo [INFO] Creating venv...
  py -m venv .venv >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] venv creation failed. See %LOG%
    >>"%LOG%" echo [ERROR] venv creation failed.
    pause
    exit /b 1
  )
)

echo [INFO] Installing requirements...
>>"%LOG%" echo [INFO] Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip >>"%LOG%" 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt >>"%LOG%" 2>&1

echo [INFO] Running parser...
>>"%LOG%" echo [INFO] Running parser...
".venv\Scripts\python.exe" -u "%SCRIPT%" >>"%LOG%" 2>&1

echo.
echo [INFO] Done. Open %LOG% if something went wrong.
>>"%LOG%" echo.
>>"%LOG%" echo ===== Run finished %DATE% %TIME% =====
pause

endlocal
