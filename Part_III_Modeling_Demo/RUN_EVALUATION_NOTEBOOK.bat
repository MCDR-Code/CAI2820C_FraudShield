@echo off
cd /d "%~dp0"
python run_notebook.py Team_A_Final_Evaluation_2026.ipynb > results\notebook_run_log.txt 2>&1
echo Finished. See results\notebook_run_log.txt
