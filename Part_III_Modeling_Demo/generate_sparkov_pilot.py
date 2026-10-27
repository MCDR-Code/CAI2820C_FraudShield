"""Generate a small, labeled Sparkov pilot across ALL 12 configured profiles.

Uses upstream Sparkov transaction/fraud generation unchanged at a pinned commit.
Creates new customers, with five per profile by default. Saves raw pipe-delimited
outputs, reusable customer records, a compatible CSV, and earlier-history/review
CSVs. Never trains or changes a model, or changes the original datasets.

Customer DOB sampling is anchored to the simulation start (not today's date),
and restricted to adults aged 18-95. Python, NumPy and Faker are seeded. Run the
generator worker in its isolated environment; preparation uses project packages.
See Synthetic_Pilot.md for setup, limitations and the two commands.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timedelta, date

ROOT = Path(__file__).resolve().parent
UPSTREAM = 'b5eb45c89d36f2aa4ef16044a42945bed8b96d93'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    source, output = args.generator_dir.resolve(), args.output.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError(f'Use the documented Sparkov commit {UPSTREAM}, found {revision}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Sparkov checkout must be unchanged.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty. Preserve it and choose a new output directory.')
    if args.customers_per_profile < 1 or (args.end - args.start).days < 8:
        raise ValueError('Use at least one customer/profile and at least nine calendar days.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw').mkdir()
    os.chdir(source)  # Upstream resolves its configuration/merchant files here.
    sys.path.insert(0, str(source))
    import numpy as np
    import faker
    from faker import Faker
    import datagen_customer as dc
    import datagen_transaction as dt
    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    config = source / 'profiles/main_config.json'
    names = sorted(json.loads(config.read_text()))
    wanted = {name: args.customers_per_profile for name in names}

    class SimulationCustomer(dc.Customer):
        def generate_age_gender(self):
            while True:
                number = random.random()
                gender, value = dc.age_gender[min(key for key in dc.age_gender if key > number)]
                age = int(value)
                if 18 <= age <= 95:
                    break
            asof = args.start.date()
            # Birthdate interval for this exact age on the simulation start date.
            def year_back(years):
                try:
                    return asof.replace(year=asof.year - years)
                except ValueError:
                    return date(asof.year - years, 2, 28)
            first = year_back(age + 1) + timedelta(days=1)
            last = year_back(age)
            born = first + timedelta(days=random.randrange((last - first).days + 1))
            return gender[0], born.isoformat(), age

    generator = SimulationCustomer(config, args.seed)
    customers, cards = [], set()
    for attempt in range(200000):
        row = generator.generate_customer()
        record = dict(zip(dc.headers, row))
        name, card = record['profile'], record['cc_num']
        if name in wanted and wanted[name] and card not in cards:
            customers.append(row)
            cards.add(card)
            wanted[name] -= 1
        if not any(wanted.values()):
            break
    else:
        raise ValueError(f'Could not fill every profile: {wanted}')
    customer_file = output / 'customers.psv'
    with customer_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='|')
        writer.writerow(dc.headers)
        writer.writerows(customers)
    for index, name in enumerate(names):
        # Independent deterministic streams per profile, no multiprocessing races.
        random.seed(args.seed + 1000 + index)
        np.random.seed(args.seed + 1000 + index)
        Faker.seed(args.seed + 1000 + index)
        print(f'Generating {name}', flush=True)
        dt.main(customer_file, source / 'profiles' / name, args.start, args.end,
                output / 'raw' / name.replace('.json', '.psv'))
    source_files = ['datagen_customer.py', 'datagen_transaction.py', 'profile_weights.py',
                    'main_config.py', 'data/merchants.csv']
    source_files += [str(p.relative_to(source)).replace('\\', '/') for p in sorted((source / 'profiles').glob('*.json'))]
    metadata = {
        'upstream_url': 'https://github.com/namebrandon/Sparkov_Data_Generation',
        'upstream_commit': revision, 'seed': args.seed,
        'start': args.start.date().isoformat(), 'end': args.end.date().isoformat(),
        'customers_per_profile': args.customers_per_profile, 'customer_count': len(customers),
        'profiles': names, 'environment': {'python': platform.python_version(),
                                        'faker': faker.VERSION, 'numpy': np.__version__},
        'wrapper_sha256': digest(__file__),
        'upstream_sha256': {name: digest(source / name) for name in source_files},
        'customer_generation': 'Sparkov customer sampler with adult DOB sampling anchored to start date; equal customers per profile.',
        'notes': ['New customers, not a continuation of the original cardholders.',
                  'Upstream transaction and fraud rules are unchanged; prevalence may differ greatly from Kaggle.',
                  'Uniform profile coverage is a scenario design, not population weighting.'],
    }
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def prepare(args):
    import pandas as pd
    from core import validate_input, load_bundle, score_transactions
    from shared.project_data import quality_checks
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    output = args.output.resolve()
    if (output / 'pilot_transactions.csv').exists():
        raise ValueError('Prepared outputs already exist; choose a new generation directory.')
    metadata = json.loads((output / 'generation.json').read_text())
    parts = [pd.read_csv(path, sep='|', dtype={'cc_num': 'string', 'zip': 'string', 'acct_num': 'string'})
             for path in sorted((output / 'raw').glob('*.psv'))]
    combined = pd.concat(parts, ignore_index=True)
    combined['trans_date_trans_time'] = combined.trans_date + ' ' + combined.trans_time
    # Sparkov uses local-machine epoch seconds. Normalize this redundant field
    # to UTC for portable files; model/history calculations use the ISO timestamp.
    combined['unix_time'] = pd.to_datetime(combined.trans_date_trans_time, utc=True).map(lambda value: int(value.timestamp()))
    combined['merchant'] = 'fraud_' + combined.merchant.astype(str)
    columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last',
               'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob',
               'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
    frame = combined[columns].sort_values(['trans_date_trans_time', 'trans_num'], kind='stable').reset_index(drop=True)
    validate_input(frame, history=True)
    quality, _ = quality_checks(frame)
    blockers = [key for key in quality if key not in ('rows', 'columns', 'fraud_rows', 'nonfraud_rows', 'fraud_rate') and
                (key.startswith('invalid_') or key in ('missing_cells', 'duplicate_rows', 'duplicate_transaction_ids',
                 'nonpositive_amounts', 'negative_city_population', 'birth_after_transaction'))]
    if any(quality[key] for key in blockers):
        raise ValueError(f'Generated data quality gate failed: {quality}')
    stamps = pd.to_datetime(frame.trans_date_trans_time)
    if stamps.min() < pd.Timestamp(metadata['start']) or stamps.max() >= pd.Timestamp(metadata['end']) + pd.Timedelta(days=1):
        raise ValueError('Generated timestamps outside the requested range.')
    if combined.groupby('profile').cc_num.nunique().to_dict() != {name: metadata['customers_per_profile'] for name in metadata['profiles']}:
        raise ValueError('Some profiles/customers have no transactions.')
    overlap = set()
    for filename in ['fraudTrain.csv', 'fraudTest.csv']:
        for chunk in pd.read_csv(args.original_dir / filename, usecols=['cc_num'], dtype='string', chunksize=250000):
            overlap.update(set(chunk.cc_num) & set(frame.cc_num))
    if overlap:
        raise ValueError('Generated card IDs collide with original cards; use another seed.')
    cutoff = pd.Timestamp(metadata['start']) + (pd.Timestamp(metadata['end']) + pd.Timedelta(days=1) - pd.Timestamp(metadata['start'])) * .75
    cutoff = cutoff.normalize()
    history, review = frame[stamps < cutoff].copy(), frame[stamps >= cutoff].copy()
    if history.empty or review.empty:
        raise ValueError('Pilot must have both earlier history and review transactions.')
    for name, data in [('pilot_transactions', frame), ('pilot_history', history), ('pilot_review', review)]:
        data.to_csv(output / f'{name}.csv', index=False)
    report = {'rows': len(frame), 'cards': int(frame.cc_num.nunique()), 'profiles': len(metadata['profiles']),
              'first_transaction': frame.trans_date_trans_time.min(), 'last_transaction': frame.trans_date_trans_time.max(),
              'fraud': int(frame.is_fraud.sum()), 'fraud_percent': float(frame.is_fraud.mean() * 100),
              'history_rows': len(history), 'review_rows': len(review), 'review_starts': str(cutoff),
              'original_card_overlap': len(overlap), 'quality_checks': quality,
              'profile_counts': combined.groupby('profile').agg(customers=('cc_num', 'nunique'), rows=('cc_num', 'size'), fraud=('is_fraud', 'sum')).to_dict('index'),
              'evaluations': [], 'not_evaluated': []}
    for model in ['baseline', 'boosting', 'history_v3']:
        path = ROOT / 'artifacts' / model / 'selected_model.joblib'
        if not path.exists():
            report['not_evaluated'].append(f'{model}: trained artifact not available')
            continue
        bundle = load_bundle(path)
        results = score_transactions(review, bundle, history).sort_values('row_number')
        truth = review.is_fraud.to_numpy()
        predictions = results.review_flag.to_numpy()
        tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
        report['evaluations'].append({'model': model, 'saved_model': bundle['metadata']['selected_model'],
             'model_sha256': digest(path), 'threshold': float(bundle['threshold']), 'review_rows': len(review),
             'true_positives': int(tp), 'false_positives': int(fp), 'false_negatives': int(fn), 'true_negatives': int(tn),
             'precision': precision_score(truth, predictions, zero_division=0),
             'recall': recall_score(truth, predictions, zero_division=0), 'f1': f1_score(truth, predictions, zero_division=0)})
        results.to_csv(output / f'{model}_review_results.csv', index=False)
    inputs = [output / 'customers.psv', *sorted((output / 'raw').glob('*.psv')), *sorted(output.glob('*.csv'))]
    report['output_sha256'] = {str(path.relative_to(output)).replace('\\', '/'): digest(path) for path in inputs}
    (output / 'pilot_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help='Run only in isolated Sparkov environment.')
    parser.add_argument('--generator-dir', type=Path, default=ROOT / 'data/Sparkov_Data_Generation')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/synthetic_pilot_2021_01')
    parser.add_argument('--start', type=datetime.fromisoformat, default=datetime(2021, 1, 1))
    parser.add_argument('--end', type=datetime.fromisoformat, default=datetime(2021, 1, 31))
    parser.add_argument('--customers-per-profile', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--original-dir', type=Path, default=ROOT.parents[1] / '01_Assignments/fraud')
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else prepare(arguments)
