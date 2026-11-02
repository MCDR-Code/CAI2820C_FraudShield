"""Score a saved model on its own fitting rows (training score) for the overfitting check.

Recreates train.py's exact chronological split from the locked experiment, scores
only the fitting period with the saved pipeline and threshold, and writes
training_score.json beside the model. It never retrains and never reads the test file.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from core import REQUIRED, load_bundle, required_columns, validate_input
from history import HISTORY_FEATURES, add_history
from train import metrics
from shared.project_data import TARGET, sha256


def training_score(artifacts, data_dir):
    artifacts, data_dir = Path(artifacts).resolve(), Path(data_dir).resolve()
    output = artifacts / 'training_score.json'
    if output.exists():
        raise ValueError(f'{output} already exists; read it instead of rerunning.')
    bundle = load_bundle(artifacts / 'selected_model.joblib')
    locked = bundle['metadata']
    train_path = data_dir / 'fraudTrain.csv'
    if sha256(train_path) != locked['data']['train_sha256']:
        raise ValueError('fraudTrain.csv differs from the file this model was trained on.')
    history = bool(locked.get('history_features'))
    columns = list(dict.fromkeys(REQUIRED + required_columns(True) + [TARGET, 'trans_num']))
    print('Loading training file...', flush=True)
    raw = pd.read_csv(train_path, dtype={'cc_num': 'string'}, usecols=columns)
    raw = validate_input(raw, history=history)
    # Same ordering and boundary rule as train.py.
    timestamps = pd.to_datetime(raw.trans_date_trans_time)
    order = np.argsort(timestamps.to_numpy(), kind='stable')
    raw = raw.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    boundary = int(len(raw) * .8)
    boundary = int(timestamps.searchsorted(timestamps.iloc[boundary], side='left'))
    development = raw.iloc[:boundary]
    if len(development) != locked['data']['training_rows'] or str(timestamps.iloc[boundary - 1]) != locked['data']['training_end']:
        raise ValueError('Recreated training split does not match the locked experiment.')
    pipeline = bundle['pipeline']
    if history:
        print('Building history features for training rows...', flush=True)
        development = development.join(add_history(development))
        pipeline.set_params(features__history='precomputed')
    print(f'Scoring {len(development):,} training rows...', flush=True)
    scores = pipeline.predict_proba(development[REQUIRED + (HISTORY_FEATURES if history else [])])[:, 1]
    result = {'scored_utc': datetime.now(timezone.utc).isoformat(), 'model': locked['selected_model'],
              'history_features': history, 'split': 'training (fitting) period',
              'training_start': locked['data']['training_start'], 'training_end': locked['data']['training_end'],
              'model_sha256': sha256(artifacts / 'selected_model.joblib'),
              **metrics(development[TARGET], scores, bundle['threshold'])}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    training_score(args.artifacts, args.data_dir)
