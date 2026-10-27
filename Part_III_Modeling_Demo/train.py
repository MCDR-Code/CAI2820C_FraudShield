"""Full-data chronological development; test outcomes are not read here."""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import CATEGORICAL, NUMERIC, REQUIRED, ROOT, FeatureBuilder, required_columns, validate_input
from history import HISTORY_FEATURES, HISTORY_VERSION, add_history, history_with_context
from model_sharing import record_model_files
from shared.project_data import TARGET, quality_checks, sha256

SEED = 42
MIN_RECALL = .80
MIN_PRECISION = .10


def metrics(y, scores, threshold):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'rows': len(y), 'fraud': int(np.sum(y)),
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'average_precision': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'review_rate': float(np.mean(pred)),
            'targets_met': bool(precision_score(y, pred, zero_division=0) >= MIN_PRECISION
                                and recall_score(y, pred, zero_division=0) >= MIN_RECALL)}


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    eligible = np.flatnonzero((recall >= MIN_RECALL) & (precision >= MIN_PRECISION))
    pool = eligible if len(eligible) else np.arange(len(thresholds))
    chosen = pool[np.argmax(f1[pool])]
    return float(thresholds[chosen])


def make_pipeline(classifier, history=False, dense=False):
    return Pipeline([
        ('features', FeatureBuilder(history='precomputed' if history else 'none')),
        ('preprocess', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC + (HISTORY_FEATURES if history else [])),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=not dense), CATEGORICAL)
        ], sparse_threshold=1.0)),
        ('classifier', classifier),
    ])


def load_gb_params(value):
    """JSON text, a JSON file, or a tune.py results file (best trial's settings)."""
    text = Path(value).read_text(encoding='utf-8') if Path(value).is_file() else value
    data = json.loads(text)
    return data['results'][0]['params'] if 'results' in data else data


def demo_by_card(frame, limit=1000):
    """Whole validation histories of a few random cards, so batch history features are meaningful."""
    rng = np.random.RandomState(SEED)
    chosen, total = [], 0
    for card in rng.permutation(frame.cc_num.unique()):
        size = int((frame.cc_num == card).sum())
        if total + size <= limit:
            chosen.append(card)
            total += size
    demo = frame[frame.cc_num.isin(chosen)].copy()
    demo['cc_num'] = demo.cc_num.map({card: f'demo_card_{i + 1}' for i, card in enumerate(chosen)})
    return demo


def train(data_dir, output_dir=ROOT / 'artifacts', trees=80, history=False,
          candidates_to_fit=('Logistic Regression', 'Random Forest'), gb_params=None, expanded_data=False):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty. Preserve this run and choose a NEW --output-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Loading full development data; test labels remain unread.', flush=True)
    raw = pd.read_csv(data_dir / 'fraudTrain.csv', dtype={'cc_num': 'string'})
    if not expanded_data and len(raw) != 1296675:
        raise ValueError('This phase requires the full original fraudTrain.csv (1,296,675 rows), not a reduced subset.')
    quality, _ = quality_checks(raw)
    blockers = ['missing_cells', 'duplicate_rows', 'duplicate_transaction_ids', 'invalid_labels',
                'invalid_timestamps', 'invalid_birth_dates', 'invalid_numeric_cells',
                'nonpositive_amounts', 'negative_city_population', 'birth_after_transaction',
                'invalid_coordinates']
    if any(quality[k] for k in blockers):
        raise ValueError(f'Development quality gate failed: {quality}')
    raw = validate_input(raw, history=history)
    timestamps = pd.to_datetime(raw.trans_date_trans_time)
    order = np.argsort(timestamps.to_numpy(), kind='stable')
    raw = raw.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    boundary = int(len(raw) * .8)
    boundary = int(timestamps.searchsorted(timestamps.iloc[boundary], side='left'))
    development, validation = raw.iloc[:boundary], raw.iloc[boundary:]
    if history:
        development = development.join(add_history(development))
        validation = validation.join(history_with_context(
            development[required_columns(True)], validation[required_columns(True)]))
    if not len(development) or not len(validation) or development[TARGET].nunique() != 2 or validation[TARGET].nunique() != 2:
        raise ValueError('Both chronological splits need both classes.')
    # Inspect test inputs/chronology, not outcomes, before locking the experiment.
    test_path = data_dir / 'fraudTest.csv'
    test_inputs = (pd.read_csv(test_path, usecols=list(dict.fromkeys(required_columns(history) + ['trans_num', 'cc_num'])), dtype={'cc_num': 'string'})
                   if test_path.exists() else None)
    if not expanded_data and (test_inputs is None or len(test_inputs) != 555719):
        raise ValueError('This phase requires the full original fraudTest.csv (555,719 rows), not a reduced subset.')
    test_time = None
    if test_inputs is not None:
        validate_input(test_inputs, history=history)
        test_time = pd.to_datetime(test_inputs.trans_date_trans_time)
        if timestamps.max() >= test_time.min() or test_inputs.trans_num.duplicated().any() or set(raw.trans_num) & set(test_inputs.trans_num):
            raise ValueError('Test chronology or transaction ID isolation failed.')
    provenance = {
        'dataset_mode': 'expanded' if expanded_data else 'original',
        'train_file': str(data_dir / 'fraudTrain.csv'), 'test_file': str(test_path) if test_inputs is not None else None,
        'train_sha256': sha256(data_dir / 'fraudTrain.csv'), 'test_sha256': sha256(test_path) if test_inputs is not None else None,
        'training_rows': len(development), 'validation_rows': len(validation), 'test_rows': len(test_inputs) if test_inputs is not None else None,
        'training_fraud': int(development[TARGET].sum()), 'validation_fraud': int(validation[TARGET].sum()),
        'training_cards': int(development.cc_num.nunique()), 'test_cards': int(test_inputs.cc_num.nunique()) if test_inputs is not None else None,
        'test_cards_unseen_in_training': len(set(test_inputs.cc_num) - set(development.cc_num)) if test_inputs is not None else None,
        'training_start': str(timestamps.iloc[0]), 'training_end': str(timestamps.iloc[boundary-1]),
        'validation_start': str(timestamps.iloc[boundary]), 'validation_end': str(timestamps.iloc[-1]),
        'test_start': str(test_time.min()) if test_time is not None else None, 'test_end': str(test_time.max()) if test_time is not None else None,
        'seed': SEED, 'test_labels_loaded': False, 'quality': quality,
    }
    del test_inputs
    columns = REQUIRED + (HISTORY_FEATURES if history else [])
    X, y = development[columns], development[TARGET]
    V, v = validation[columns], validation[TARGET]
    records, candidates, curves = [], {}, {}
    majority = metrics(v, np.zeros(len(v)), .5)
    records.append({'model': 'Majority class', 'fit_seconds': 0., **majority})
    rule_cutoff = float(development.amt.quantile(.99))
    rule = metrics(v, V.amt.to_numpy(), rule_cutoff)
    records.append({'model': 'Amount rule', 'fit_seconds': 0., **rule})
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=500, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=trees, max_depth=18, min_samples_leaf=3,
                                                max_samples=.6, class_weight='balanced_subsample',
                                                n_jobs=4, random_state=SEED),
        'Gradient Boosting': HistGradientBoostingClassifier(
            **{'max_iter': 200, 'learning_rate': .1, 'max_leaf_nodes': 31, 'class_weight': 'balanced',
               **(gb_params or {}), 'early_stopping': False, 'random_state': SEED}),
    }
    models = {k: v for k, v in models.items() if k in candidates_to_fit}
    for name, classifier in models.items():
        print(f'Fitting {name} on {len(X):,} training rows...', flush=True)
        pipeline = make_pipeline(classifier, history, dense=name == 'Gradient Boosting')
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            pipeline.fit(X, y)
        seconds = time.perf_counter() - start
        scores = pipeline.predict_proba(V)[:, 1]
        threshold = choose_threshold(v, scores)
        row = {'model': name, 'fit_seconds': seconds, **metrics(v, scores, threshold)}
        records.append(row)
        candidates[name] = (pipeline, threshold)
        curves[name] = precision_recall_curve(v, scores)[:2]
        print(json.dumps(row, indent=2), flush=True)
    table = pd.DataFrame(records)
    eligible = table[table.model.isin(models)].sort_values(['targets_met', 'f1', 'average_precision'], ascending=False, kind='stable')
    selected = eligible.iloc[0]
    pipeline, threshold = candidates[selected['model']]
    locked = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_model': selected['model'],
              'threshold': threshold, 'min_recall': MIN_RECALL, 'min_precision': MIN_PRECISION,
              'selection_rule': 'Prefer candidates meeting both targets; then maximum validation F1, then average precision.',
              'threshold_rule': 'Maximum validation F1 subject to recall >= .80 and precision >= .10, if feasible; otherwise maximum F1.',
              'refit_on_validation': False, 'history_features': history,
              'history_version': HISTORY_VERSION if history else None,
              'history_contract': 'earlier-context-plus-review-batch' if history else 'transaction-only',
              'data': provenance,
              'feature_count': len(pipeline['preprocess'].get_feature_names_out()),
              'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                              'scipy': version('scipy'), 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
              'parameters': {k: c.get_params() for k, c in models.items()},
              'limitations': ['Synthetic historical U.S. data; no claim of banking performance.',
                              'Known customers cross time periods by design; new-customer results may differ.',
                              'Gender, age and geography can act as proxies; subgroup evidence is descriptive, not fairness certification.',
                              'Class-weighted model scores are not calibrated fraud probabilities.',
                              'Earlier EDA exposed development patterns; validation is a development estimate.']
                   + (['History scoring requires explicitly loaded earlier context or an explicit cold start. '
                       'Validation uses training-period context plus strictly earlier validation transactions. '
                       'Simultaneous transactions cannot influence each other. Context is not automatically updated by uploads.'] if history else [])}
    table.to_csv(output_dir / 'validation_comparison.csv', index=False)
    (output_dir / 'experiment.json').write_text(json.dumps(locked, indent=2), encoding='utf-8')
    joblib.dump({'pipeline': pipeline, 'threshold': threshold, 'metadata': locked}, output_dir / 'selected_model.joblib', compress=3)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (precision, recall) in curves.items():
        ax.plot(recall, precision, label=name)
    ax.axhline(v.mean(), color='#777777', linestyle='--', label='Prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Chronological validation precision-recall curves', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / 'validation_pr.png', dpi=160); plt.close(fig)
    # A deterministic validation-only sample, without labels or identity fields.
    if history:
        demo = demo_by_card(validation[required_columns(True)])
        identities = development[required_columns(True)].copy()
        mapping = dict(zip(validation.loc[demo.index, 'cc_num'], demo.cc_num))
        identities = identities[identities.cc_num.isin(mapping)].copy()
        identities['cc_num'] = identities.cc_num.map(mapping)
        identities.to_csv(output_dir / 'demo_history.csv', index=False)
    else:
        demo = V.sample(n=min(1000, len(V)), random_state=SEED)
    demo.to_csv(output_dir / 'demo_transactions.csv', index=False)
    record_model_files(output_dir, locked, pipeline, demo, identities if history else None)
    print(f'Locked {selected["model"]}; artifacts: {output_dir}', flush=True)
    return locked, table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1] / 'fraud')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--trees', type=int, default=80)
    parser.add_argument('--models', default='Logistic Regression,Random Forest',
                        help='comma-separated candidates from: Logistic Regression, Random Forest, Gradient Boosting')
    parser.add_argument('--history', action='store_true', help='add per-card history features')
    parser.add_argument('--expanded-data', action='store_true', help='Explicit new experiment: allow new dataset sizes and a pending future test file.')
    parser.add_argument('--gb-params', help='Gradient Boosting settings: JSON text, a JSON file, or tune.py output')
    args = parser.parse_args()
    if args.trees < 1:
        parser.error('--trees must be positive')
    candidates = tuple(name.strip() for name in args.models.split(','))
    if not candidates or set(candidates) - {'Logistic Regression', 'Random Forest', 'Gradient Boosting'}:
        parser.error('Choose valid --models names: Logistic Regression, Random Forest, Gradient Boosting.')
    train(args.data_dir, args.output_dir, args.trees, args.history, candidates,
          load_gb_params(args.gb_params) if args.gb_params else None, expanded_data=args.expanded_data)
