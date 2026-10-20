"""Shared input contract and saved inference pipeline for FraudShield."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from shared.project_data import CATEGORICAL, FEATURES, NUMERIC, derive_features

ROOT = Path(__file__).resolve().parent
REQUIRED = ['trans_date_trans_time', 'dob', 'amt', 'city_pop', 'lat', 'long',
            'merch_lat', 'merch_long', 'category', 'gender', 'state']
MAX_ROWS = 50000
MAX_BYTES = 20 * 1024 * 1024


def validate_input(raw):
    """Return a validated copy; never silently drop or impute uploaded rows."""
    if raw.empty:
        raise ValueError('The CSV contains no transactions.')
    if raw.columns.duplicated().any():
        raise ValueError('CSV column names must be unique.')
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df = raw.copy()
    for column in REQUIRED:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{column} contains missing or blank values.')
    for column in NUMERIC[:6]:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if not np.isfinite(df[column]).all():
            raise ValueError(f'{column} must contain finite numbers.')
    timestamp = pd.to_datetime(df.trans_date_trans_time, errors='coerce', utc=True, format='mixed')
    birth = pd.to_datetime(df.dob, errors='coerce', utc=True, format='mixed')
    if timestamp.isna().any() or birth.isna().any():
        raise ValueError('Invalid transaction timestamp or date of birth.')
    age = (timestamp - birth).dt.days / 365.2425
    if not age.between(0, 120).all():
        raise ValueError('Age at transaction must be between 0 and 120 years.')
    df['trans_date_trans_time'] = timestamp.dt.tz_convert(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['dob'] = birth.dt.tz_convert(None).dt.strftime('%Y-%m-%d')
    if (df.amt <= 0).any() or (df.city_pop < 0).any():
        raise ValueError('Amounts must be positive; city population cannot be negative.')
    for column, limit in [('lat', 90), ('merch_lat', 90), ('long', 180), ('merch_long', 180)]:
        if not df[column].between(-limit, limit).all():
            raise ValueError(f'{column} is outside the valid coordinate range.')
    for column in CATEGORICAL:
        df[column] = df[column].astype(str).str.strip()
    return df


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return derive_features(validate_input(X))[FEATURES]


def load_bundle(path=None):
    # Only load model files produced locally by this project, never uploaded joblib files.
    path = Path(path or ROOT / 'artifacts/selected_model.joblib')
    if not path.exists():
        raise FileNotFoundError('No saved model. Run train.py before starting the demo.')
    return joblib.load(path)


def score_transactions(raw, bundle):
    validated = validate_input(raw)
    scores = bundle['pipeline'].predict_proba(validated)[:, 1]
    result = pd.DataFrame({
        'row_number': np.arange(1, len(raw) + 1),
        'transaction_time': validated.trans_date_trans_time.to_numpy(),
        'category': validated.category.to_numpy(),
        'amount_usd': validated.amt.to_numpy(),
        'model_score': scores,
        'review_flag': scores >= bundle['threshold'],
    })
    # Do not echo customer names, addresses, DOB or card numbers in results.
    return result.sort_values('model_score', ascending=False, kind='stable')
