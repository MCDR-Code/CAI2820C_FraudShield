"""Full-data chronological development; test outcomes are not read here."""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
from history import HISTORY_FEATURES, add_history
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
          candidates_to_fit=('Logistic Regression', 'Random Forest'), gb_params=None):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty. Preserve this run and choose a NEW --output-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Loading full development data; test labels remain unread.', flush=True)
    raw = pd.read_csv(data_dir / 'fraudTrain.csv', dtype={'cc_num': 'string'})
    if len(raw) != 1296675:
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
    if history:
        # Causal: every row sees only earlier transactions, so validation never leaks into training.
        raw = raw.join(add_history(raw))
    boundary = int(len(raw) * .8)
    boundary = int(timestamps.searchsorted(timestamps.iloc[boundary], side='left'))
    development, validation = raw.iloc[:boundary], raw.iloc[boundary:]
    if not len(development) or not len(validation) or development[TARGET].nunique() != 2 or validation[TARGET].nunique() != 2:
        raise ValueError('Both chronological splits need both classes.')
    # Inspect test inputs/chronology, not outcomes, before locking the experiment.
    test_inputs = pd.read_csv(data_dir / 'fraudTest.csv', usecols=required_columns(history) + ['trans_num', 'cc_num'], dtype={'cc_num': 'string'})
    if len(test_inputs) != 555719:
        raise ValueError('This phase requires the full original fraudTest.csv (555,719 rows), not a reduced subset.')
    validate_input(test_inputs, history=history)
    test_time = pd.to_datetime(test_inputs.trans_date_trans_time)
    if timestamps.max() >= test_time.min() or test_inputs.trans_num.duplicated().any() or set(raw.trans_num) & set(test_inputs.trans_num):
        raise ValueError('Test chronology or transaction ID isolation failed.')
    provenance = {
        'train_file': str(data_dir / 'fraudTrain.csv'), 'test_file': str(data_dir / 'fraudTest.csv'),
        'train_sha256': sha256(data_dir / 'fraudTrain.csv'), 'test_sha256': sha256(data_dir / 'fraudTest.csv'),
        'training_rows': len(development), 'validation_rows': len(validation), 'test_rows': len(test_inputs),
        'training_fraud': int(development[TARGET].sum()), 'validation_fraud': int(validation[TARGET].sum()),
        'training_cards': int(development.cc_num.nunique()), 'test_cards': int(test_inputs.cc_num.nunique()),
        'test_cards_unseen_in_training': len(set(test_inputs.cc_num) - set(development.cc_num)),
        'training_start': str(timestamps.iloc[0]), 'training_end': str(timestamps.iloc[boundary-1]),
        'validation_start': str(timestamps.iloc[boundary]), 'validation_end': str(timestamps.iloc[-1]),
        'test_start': str(test_time.min()), 'test_end': str(test_time.max()),
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
    if history:
        pipeline.set_params(features__history='batch')  # saved model derives history from the scored batch
    locked = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_model': selected['model'],
              'threshold': threshold, 'min_recall': MIN_RECALL, 'min_precision': MIN_PRECISION,
              'selection_rule': 'Prefer candidates meeting both targets; then maximum validation F1, then average precision.',
              'threshold_rule': 'Maximum validation F1 subject to recall >= .80 and precision >= .10, if feasible; otherwise maximum F1.',
              'refit_on_validation': False, 'history_features': history, 'data': provenance,
              'feature_count': len(pipeline['preprocess'].get_feature_names_out()),
              'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                              'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
              'parameters': {k: c.get_params() for k, c in models.items()},
              'limitations': ['Synthetic historical U.S. data; no claim of banking performance.',
                              'Known customers cross time periods by design; new-customer results may differ.',
                              'Gender, age and geography can act as proxies; subgroup evidence is descriptive, not fairness certification.',
                              'Class-weighted model scores are not calibrated fraud probabilities.',
                              'Earlier EDA exposed development patterns; validation is a development estimate.']
                   + (['History features use only earlier transactions on the same card. Uploads are scored with the history inside the uploaded batch only, so short batches see less history than training did.'] if history else [])}
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
    else:
        demo = V.sample(n=min(1000, len(V)), random_state=SEED)
    demo.to_csv(output_dir / 'demo_transactions.csv', index=False)
    print(f'Locked {selected["model"]}; artifacts: {output_dir}', flush=True)
    return locked, table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1] / 'fraud')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--trees', type=int, default=80)
    parser.add_argument('--models', default='Logistic Regression,Random Forest',
                        help='comma-separated candidates from: Logistic Regression, Random Forest, Gradient Boosting')
    parser.add_argument('--gb-params', help='Gradient Boosting settings: JSON text, a JSON file, or tune.py output')
    parser.add_argument('--history', action='store_true', help='add per-card history features')
    args = parser.parse_args()
    if args.trees < 1:
        parser.error('--trees must be positive')
    train(args.data_dir, args.output_dir, args.trees, args.history, tuple(args.models.split(',')),
          load_gb_params(args.gb_params) if args.gb_params else None)
