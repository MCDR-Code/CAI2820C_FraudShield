"""Checksums and a reproducible smoke test for trusted, team-generated models."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from core import ROOT, load_bundle, score_transactions
from history import HISTORY_VERSION
from shared.project_data import sha256

SOURCES = ['core.py', 'history.py', 'shared/project_data.py', 'train.py', 'model_sharing.py']


def source_hash(path):
    # Git can check out CRLF on Windows and LF on macOS; code identity ignores that difference.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def record_model_files(directory, locked, pipeline, demo, context=None):
    directory = Path(directory)
    bundle = {'pipeline': pipeline, 'threshold': locked['threshold'], 'metadata': locked}
    result = score_transactions(demo, bundle, context).sort_values('row_number')
    expected = result[['row_number', 'model_score', 'review_flag']].to_dict('records')
    (directory / 'demo_expected_scores.json').write_text(json.dumps(expected), encoding='utf-8')
    files = ['selected_model.joblib', 'experiment.json', 'validation_comparison.csv',
             'validation_pr.png', 'demo_transactions.csv', 'demo_expected_scores.json']
    if locked['history_features']:
        files.append('demo_history.csv')
    manifest = {
        'selected_model': locked['selected_model'], 'history_version': locked['history_version'],
        'environment': locked['environment'],
        'artifact_sha256': {name: sha256(directory / name) for name in files},
        'source_sha256_lf': {name: source_hash(ROOT / name) for name in SOURCES},
    }
    (directory / 'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def check_model(directory):
    """Verify a trusted model's copies, code, environment and example predictions.

    A manifest is not a digital signature. Only accept artifacts from a trusted
    teammate: loading joblib can execute code even when a supplied checksum matches.
    """
    directory = Path(directory)
    manifest = json.loads((directory / 'model_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifact_sha256'].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f'Artifact checksum differs: {name}')
    for name, expected in manifest['source_sha256_lf'].items():
        if name not in SOURCES or source_hash(ROOT / name) != expected:
            raise ValueError(f'Model needs matching training source: {name}. Check out the shared code version.')
    recorded = manifest['environment']
    if platform.python_version_tuple()[:2] != tuple(recorded['python'].split('.')[:2]):
        raise ValueError(f"Use Python {recorded['python']} (matching major/minor), not {platform.python_version()}.")
    for key, package in [('pandas', 'pandas'), ('numpy', 'numpy'), ('scipy', 'scipy'),
                         ('scikit_learn', 'scikit-learn'), ('joblib', 'joblib')]:
        if version(package) != recorded[key]:
            raise ValueError(f'Install {package}=={recorded[key]} to match training.')
    saved = load_bundle(directory / 'selected_model.joblib')
    locked = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
    if saved['metadata'] != locked:
        raise ValueError('Model and experiment metadata differ.')
    if locked['history_features'] and locked['history_version'] != HISTORY_VERSION:
        raise ValueError('The history model needs retraining for this feature version.')
    demo = pd.read_csv(directory / 'demo_transactions.csv', dtype={'cc_num': 'string'})
    context = pd.read_csv(directory / 'demo_history.csv', dtype={'cc_num': 'string'}) if locked['history_features'] else None
    result = score_transactions(demo, saved, context).sort_values('row_number')
    expected = pd.DataFrame(json.loads((directory / 'demo_expected_scores.json').read_text(encoding='utf-8')))
    np.testing.assert_array_equal(result.row_number.to_numpy(), expected.row_number.to_numpy())
    np.testing.assert_allclose(result.model_score.to_numpy(), expected.model_score.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(result.review_flag.to_numpy(), expected.review_flag.to_numpy())
    return {'model': locked['selected_model'], 'history_version': locked['history_version'],
            'features': locked['feature_count'], 'verified_demo_rows': len(result),
            'verified_demo_flags': int(result.review_flag.sum()),
            'model_sha256': manifest['artifact_sha256']['selected_model.joblib'],
            'status': 'Copy, source, environment and prediction checks passed'}
