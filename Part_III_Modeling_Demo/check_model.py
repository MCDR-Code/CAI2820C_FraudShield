"""Verify a trusted trained-model folder before sharing or loading it on another computer."""

import argparse
import json
from pathlib import Path

from model_sharing import check_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/history_v3'))
    args = parser.parse_args()
    try:
        print(json.dumps(check_model(args.artifacts), indent=2))
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f'Model verification failed: {exc}\n')
