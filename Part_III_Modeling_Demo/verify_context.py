"""Verify validation/inference parity without opening official test labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import REQUIRED, load_bundle, required_columns, score_transactions, validate_input
from history import HISTORY_VERSION, history_with_context
from shared.project_data import sha256
from train import metrics


def verify(data_dir, artifacts):
    artifacts = Path(artifacts)
    saved = load_bundle(artifacts / 'selected_model.joblib')
    locked = saved['metadata']
    if locked.get('history_version') != HISTORY_VERSION:
        raise ValueError('Retrain with the corrected history source first.')
    path = Path(data_dir) / 'fraudTrain.csv'
    if sha256(path) != locked['data']['train_sha256']:
        raise ValueError('Original training CSV fingerprint does not match this experiment.')
    raw = pd.read_csv(path, usecols=required_columns(True) + ['is_fraud'], dtype={'cc_num': 'string'})
    raw = validate_input(raw, history=True).sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)
    end = locked['data']['training_end']
    context = raw.loc[raw.trans_date_trans_time <= end, required_columns(True)]
    review = raw.loc[raw.trans_date_trans_time > end]
    precomputed = review[REQUIRED].join(history_with_context(context, review))
    expected = saved['pipeline'].predict_proba(precomputed)[:, 1]
    deployed = score_transactions(review, saved, context).sort_values('row_number')
    actual = deployed.model_score.to_numpy()
    np.testing.assert_array_equal(expected, actual)
    measured = metrics(review.is_fraud, actual, saved['threshold'])
    recorded = pd.read_csv(artifacts / 'validation_comparison.csv').set_index('model').loc[locked['selected_model']]
    for name in ['precision', 'recall', 'f1', 'average_precision']:
        np.testing.assert_allclose(measured[name], recorded[name], rtol=0, atol=1e-12)
    report = {'history_version': HISTORY_VERSION, 'context_rows': len(context), 'review_rows': len(review),
              'model_sha256': sha256(artifacts / 'selected_model.joblib'),
              'training_sha256': locked['data']['train_sha256'],
              'validation_inference_max_absolute_difference': float(np.max(np.abs(expected - actual))),
              'official_test_evaluated': False, **measured}
    (artifacts / 'context_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/history'))
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir, args.artifacts), indent=2))
