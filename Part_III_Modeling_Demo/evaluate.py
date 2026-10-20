"""Evaluate a locked model once; never select models or thresholds here."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from core import REQUIRED, ROOT, load_bundle, validate_input
from train import metrics
from shared.project_data import TARGET, sha256


def evaluate(artifacts=ROOT / 'artifacts', data_dir=None):
    artifacts = Path(artifacts).resolve()
    output = artifacts / 'final_test.json'
    if output.exists():
        raise ValueError('Final test has already been evaluated. Read the saved results instead of rerunning.')
    bundle = load_bundle(artifacts / 'selected_model.joblib')
    locked = bundle['metadata']
    if locked != json.loads((artifacts / 'experiment.json').read_text()):
        raise ValueError('Saved model and experiment metadata do not match.')
    path = Path(data_dir) / 'fraudTest.csv' if data_dir else Path(locked['data']['test_file'])
    if sha256(path) != locked['data']['test_sha256']:
        raise ValueError('The final test file changed since model selection.')
    train_path = Path(data_dir) / 'fraudTrain.csv' if data_dir else Path(locked['data']['train_file'])
    if sha256(train_path) != locked['data']['train_sha256']:
        raise ValueError('The original training file changed; subgroup coverage cannot be verified.')
    # Labels are first opened only after the pipeline and threshold have been saved.
    test = pd.read_csv(path, usecols=REQUIRED + [TARGET, 'cc_num'], dtype={'cc_num': 'string'})
    validate_input(test)
    if test[TARGET].isna().any() or set(test[TARGET].unique()) != {0, 1}:
        raise ValueError('Final test labels must contain both 0 and 1.')
    start = time.perf_counter()
    scores = bundle['pipeline'].predict_proba(test[REQUIRED])[:, 1]
    seconds = time.perf_counter() - start
    result = {'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'model': locked['selected_model'],
              'model_sha256': sha256(artifacts / 'selected_model.joblib'),
              'test_sha256': locked['data']['test_sha256'],
              'scoring_seconds': seconds, 'rows_per_second': len(test) / seconds,
              **metrics(test[TARGET], scores, bundle['threshold'])}
    # Reserve the evaluation record before writing presentation outputs.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    prediction = scores >= bundle['threshold']
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(test[TARGET], prediction, labels=[0, 1]),
                           display_labels=['Legitimate', 'Fraud']).plot(ax=ax, colorbar=False, values_format=',d', cmap='Greens')
    ax.set_title('Reserved final test'); fig.tight_layout(); fig.savefig(artifacts / 'test_confusion.png', dpi=160); plt.close(fig)
    groups = []
    # Card coverage is calculated from the same training-period boundary, not validation.
    original = pd.read_csv(train_path, usecols=['cc_num', 'trans_date_trans_time'], dtype={'cc_num': 'string'})
    train_cards = set(original.loc[pd.to_datetime(original.trans_date_trans_time) <= pd.Timestamp(locked['data']['training_end']), 'cc_num'])
    test['customer_coverage'] = np.where(test.cc_num.isin(train_cards), 'Seen in training', 'Unseen in training')
    for column in ['gender', 'category', 'customer_coverage']:
        for group, indices in test.groupby(column).groups.items():
            indices = np.asarray(indices)
            y, pred = test.loc[indices, TARGET], prediction[indices]
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            groups.append({'field': column, 'group': group, 'rows': len(indices), 'fraud': int(y.sum()),
                           'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                           'recall': float(tp / (tp+fn)) if tp+fn else None,
                           'precision': float(tp / (tp+fp)) if tp+fp else None,
                           'false_positive_rate': float(fp/(fp+tn)) if fp+tn else None,
                           'support_warning': 'Small fraud support (<30)' if y.sum()<30 else ''})
    pd.DataFrame(groups).to_csv(artifacts / 'test_subgroups.csv', index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--data-dir', type=Path, help='Override original file locations after moving the project')
    args = parser.parse_args()
    evaluate(args.artifacts, args.data_dir)
