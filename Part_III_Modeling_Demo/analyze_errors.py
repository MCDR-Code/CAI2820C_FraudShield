"""Describe what a saved model gets wrong on the chronological validation window (no test data)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import REQUIRED, ROOT, bundle_uses_history, load_bundle
from history import HISTORY_FEATURES, HISTORY_VERSION
from shared.project_data import TARGET, sha256
from tune import development_split


def breakdown(frame, by, bins=None):
    key = pd.cut(frame[by], bins) if bins is not None else frame[by]
    grouped = frame.groupby(key, observed=True)
    out = pd.DataFrame({'rows': grouped.size(), 'fraud': grouped[TARGET].sum(),
                        'caught': grouped.apply(lambda g: int(((g[TARGET] == 1) & g.flag).sum()), include_groups=False),
                        'false_alerts': grouped.apply(lambda g: int(((g[TARGET] == 0) & g.flag).sum()), include_groups=False)})
    out['missed'] = out.fraud - out.caught
    out['recall'] = (out.caught / out.fraud.where(out.fraud > 0)).round(3)
    return out


def validation_errors(artifacts, data_dir):
    """Check provenance before calling another file's predictions validation results."""
    artifacts = Path(artifacts)
    bundle = load_bundle(artifacts / 'selected_model.joblib')
    locked = bundle['metadata']
    history = bundle_uses_history(bundle)
    if history and locked.get('history_version') != HISTORY_VERSION:
        raise ValueError('History model version does not match this source. Train a new v3 model first.')
    if locked != json.loads((artifacts / 'experiment.json').read_text(encoding='utf-8')):
        raise ValueError('Saved model and experiment metadata do not match.')
    if sha256(Path(data_dir) / 'fraudTrain.csv') != locked['data']['train_sha256']:
        raise ValueError('The source training CSV fingerprint differs from the saved experiment.')
    pipeline = bundle['pipeline']
    if history and pipeline['features'].history != 'precomputed':
        raise ValueError('History analysis requires the saved explicit-context pipeline.')
    raw, _, boundary = development_split(data_dir, history=history)
    if boundary != locked['data']['training_rows'] or len(raw) - boundary != locked['data']['validation_rows']:
        raise ValueError('Validation split does not match the saved experiment.')
    validation = raw.iloc[boundary:].copy()
    columns = REQUIRED + (HISTORY_FEATURES if history else [])
    validation['score'] = pipeline.predict_proba(validation[columns])[:, 1]
    validation['flag'] = validation.score >= bundle['threshold']
    validation['hour'] = pd.to_datetime(validation.trans_date_trans_time).dt.hour
    return validation


def main(artifacts, data_dir):
    validation = validation_errors(artifacts, data_dir)
    fraud = validation[validation[TARGET] == 1]
    print(f'Validation rows {len(validation):,}; fraud {len(fraud)}; flagged {int(validation.flag.sum())}; '
          f'missed {int((~fraud.flag).sum())}; false alerts {int(((validation[TARGET] == 0) & validation.flag).sum())}')
    print('\nBy category (sorted by missed fraud):')
    print(breakdown(validation, 'category').sort_values('missed', ascending=False).to_string())
    print('\nBy amount band:')
    print(breakdown(validation, 'amt', [0, 10, 50, 100, 250, 500, 1000, 1e6]).to_string())
    print('\nBy hour of day:')
    print(breakdown(validation, 'hour').to_string())
    if 'hist_prior_count' in validation:
        print('\nBy number of prior transactions on the card:')
        print(breakdown(validation, 'hist_prior_count', [-1, 0, 10, 100, 500, 1e6]).to_string())
        print('\nBy new merchant for the card (1 = never seen before):')
        print(breakdown(validation, 'hist_new_merchant').to_string())
    print('\nScore distribution of missed fraud vs caught fraud:')
    print(fraud.groupby('flag').score.describe().round(3).to_string())
    print(f'\nMissed fraud: median amount ${fraud[~fraud.flag].amt.median():.2f} vs caught ${fraud[fraud.flag].amt.median():.2f}')
    print(f'Fraud spread over {fraud.cc_num.nunique()} cards; missed fraud on {fraud[~fraud.flag].cc_num.nunique()} cards')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/history_v3')
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    main(args.artifacts, args.data_dir)
