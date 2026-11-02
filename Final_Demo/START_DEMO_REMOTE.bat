@echo off
REM Remote presentation mode: password-protected demo on port 5001 (eero forwards 5100 -^> 5001).
REM Close this window to stop the demo.
cd /d "%~dp0"
set /p "FRAUDSHIELD_PASSWORD=Choose the demo password for this session: "
if "%FRAUDSHIELD_PASSWORD%"=="" (echo A password is required. & pause & exit /b 1)
echo.
echo Demo running. From outside: http://b2518067.eero.online:5100   At home: http://127.0.0.1:5001
echo Any user name works; use the password you just typed. Close this window to stop.
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 5001
