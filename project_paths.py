"""Locate the single project root independently of the working directory.

Legacy module names remain importable so trusted saved joblib pipelines load
without retraining. No dataset is read and no model is loaded by this module.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
for relative in ('Shared', 'Part_III/Modeling', 'Final_Demo'):
    directory = str(PROJECT_ROOT / relative)
    if directory not in sys.path:
        sys.path.insert(0, directory)
