@echo off
cd /d "%~dp0\..\.."
python Part_III/Modeling/run_notebook.py Part_III/Modeling/Team_A_Final_Evaluation_2026.ipynb > Part_III\Results\notebook_run_log.txt 2>&1
echo See Part_III\Results\notebook_run_log.txt
pause
