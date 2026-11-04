@echo off
REM Remote presentation: starts the demo, a Cloudflare tunnel, prints the link and opens a QR code.
REM Press Enter in the window to close the tunnel and stop the demo.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Demo-Remote.ps1" %*
pause
