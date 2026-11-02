@echo off
cd /d "%~dp0"
echo FraudShield local demo: http://127.0.0.1:8766
python -m uvicorn app:app --host 127.0.0.1 --port 8766
pause
