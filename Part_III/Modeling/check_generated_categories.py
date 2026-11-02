"""Fail if generated legitimate transactions do not cover every category each month.

Guards against the upstream Sparkov date/category coupling corrected by sparkov_fixes.py.
Usage: python check_generated_categories.py <generated folder> [...]
"""

# Discover the project from any supported entry point; retain saved-model imports.
import sys as _sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / 'project_paths.py').is_file())
if str(_project) not in _sys.path:
    _sys.path.insert(0, str(_project))
from project_paths import PROJECT_ROOT

import json
import sys
from pathlib import Path
import pandas as pd

problems, report = [], {}
for folder in map(Path, sys.argv[1:]):
    for path in sorted(folder.glob('transactions_*.csv')):
        frame = pd.read_csv(path, usecols=['trans_date_trans_time', 'category', 'is_fraud'])
        legit = frame[frame.is_fraud == 0]
        months = legit.groupby(legit.trans_date_trans_time.str[:7]).category.nunique()
        categories = frame.category.nunique()
        report[path.name] = {'categories': int(categories), 'legit_categories': int(legit.category.nunique()),
                             'min_legit_categories_per_month': int(months.min())}
        if months.min() < categories:
            problems.append(f'{path.name}: a month has only {months.min()} of {categories} legitimate categories')
print(json.dumps(report, indent=2))
if problems:
    sys.exit('Category coverage check FAILED:\n' + '\n'.join(problems))
print('Category coverage check passed: every month has legitimate transactions in every category.')
