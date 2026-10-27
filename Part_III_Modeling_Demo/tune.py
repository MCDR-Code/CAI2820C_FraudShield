"""Random-search the gradient boosting settings without touching validation or test.

Fits on the earliest part of the development-training period and scores average
precision on the latest part of that same period. The chronological validation
window used by train.py and the official test file are never read for scoring.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from core import REQUIRED, ROOT, validate_input
from history import HISTORY_FEATURES, add_history
from shared.project_data import TARGET
from train import SEED, make_pipeline

SPACE = {
    'learning_rate': [.03, .05, .1, .2],
    'max_iter': [200, 400, 600],
    'max_leaf_nodes': [15, 31, 63, 127],
    'min_samples_leaf': [20, 50, 100, 200],
    'l2_regularization': [0., 1., 10.],
    'max_features': [.5, .8, 1.],
    'class_weight': ['balanced', None],
}


def development_split(data_dir, history=True):
    """Sorted development rows (with history features) and the train/validation boundary train.py uses."""
    raw = pd.read_csv(Path(data_dir) / 'fraudTrain.csv', dtype={'cc_num': 'string'})
    raw = validate_input(raw, history=True)
    stamps = pd.to_datetime(raw.trans_date_trans_time)
    raw = raw.iloc[np.argsort(stamps.to_numpy(), kind='stable')].reset_index(drop=True)
    stamps = pd.to_datetime(raw.trans_date_trans_time)
    if history:
        raw = raw.join(add_history(raw))
    boundary = int(stamps.searchsorted(stamps.iloc[int(len(raw) * .8)], side='left'))
    return raw, stamps, boundary


def tune(data_dir, output, trials, history=True):
    raw, stamps, boundary = development_split(data_dir, history)
    # Tuning stays left of the validation boundary.
    training = raw.iloc[:boundary]
    inner = int(stamps.searchsorted(stamps.iloc[int(boundary * .75)], side='left'))
    columns = REQUIRED + (HISTORY_FEATURES if history else [])
    fit, hold = training.iloc[:inner], training.iloc[inner:]
    print(f'Tuning fit rows {len(fit):,} ({int(fit[TARGET].sum())} fraud); '
          f'holdout rows {len(hold):,} ({int(hold[TARGET].sum())} fraud)', flush=True)
    prep = Pipeline(make_pipeline(None, history, dense=True).steps[:-1]).fit(fit[columns])
    Xf, Xh = prep.transform(fit[columns]), prep.transform(hold[columns])
    rng = np.random.RandomState(SEED)
    results = []
    for trial in range(trials):
        params = {k: v[rng.randint(len(v))] for k, v in SPACE.items()}
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params)
        model.fit(Xf, fit[TARGET])
        ap = float(average_precision_score(hold[TARGET], model.predict_proba(Xh)[:, 1]))
        results.append({'trial': trial, 'holdout_average_precision': ap,
                        'seconds': time.perf_counter() - start, 'params': params})
        print(f'{trial:2d} AP {ap:.4f} {params}', flush=True)
    results.sort(key=lambda r: -r['holdout_average_precision'])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({'history': history, 'results': results}, indent=2), encoding='utf-8')
    print('Best:', json.dumps(results[0], indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts_tuning/tuning.json')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--no-history', action='store_true')
    args = parser.parse_args()
    tune(args.data_dir, args.output, args.trials, not args.no_history)
