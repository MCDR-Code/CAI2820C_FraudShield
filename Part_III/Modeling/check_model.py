"""Verify a trusted trained-model folder before sharing or loading it on another computer."""

# Discover the project from any supported entry point; retain saved-model imports.
import sys as _sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / 'project_paths.py').is_file())
if str(_project) not in _sys.path:
    _sys.path.insert(0, str(_project))
from project_paths import PROJECT_ROOT

import argparse
import json
from pathlib import Path

from model_sharing import check_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'Models/expanded_fixed_run1/history_v3')
    args = parser.parse_args()
    try:
        print(json.dumps(check_model(args.artifacts), indent=2))
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f'Model verification failed: {exc}\n')
