@echo off
cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File run_fixed_pipeline.ps1
