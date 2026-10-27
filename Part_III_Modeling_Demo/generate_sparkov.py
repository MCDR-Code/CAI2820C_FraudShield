"""Generate NEW labeled customers/transactions for 2022-2026, without training.

Generation uses the pinned Sparkov engine shared with the earlier pilot. This
entry point defaults to 999 new customers across all 12 profiles, a distinct
seed, and four independent generation processes in the isolated environment.
Preparation runs in the project environment, validates all rows, checks card
overlap with originals/pilot, and writes five chronologically sorted yearly CSVs.
Year files are the primary data; no redundant merged copy is saved.

Usage: isolated-python generate_sparkov.py --worker [--customers N]
       project-python generate_sparkov.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
from pathlib import Path

from generate_sparkov_pilot import ROOT, digest, worker, generate_profile


def complete_customers(args):
    """Retry ONLY customers with zero rows, retaining each original raw output."""
    output, source = args.output.resolve(), args.generator_dir.resolve()
    if (output / 'dataset_report.json').exists():
        raise ValueError('Do not alter a prepared dataset.')
    metadata = json.loads((output / 'generation.json').read_text())
    with (output / 'customers.psv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='|')
        fields, customers = reader.fieldnames, list(reader)
    coverage = output / 'coverage'
    retries = metadata.get('zero_transaction_retries', [])
    for attempt in range(1, 6):
        seen = set()
        for path in sorted((output / 'raw').glob('*.psv')):
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.reader(stream, delimiter='|')
                card_column = next(reader).index('cc_num')
                seen.update(row[card_column] for row in reader)
        missing = [row for row in customers if row['cc_num'] not in seen]
        if not missing:
            metadata['zero_transaction_retries'] = retries
            metadata['completion_code_sha256'] = digest(__file__)
            (output / 'generation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            return
        coverage.mkdir(exist_ok=True)
        tasks = []
        for index, name in enumerate(metadata['profiles']):
            selected = [row for row in missing if row['profile'] == name]
            if not selected:
                continue
            seed = metadata['seed'] + 1000000 + attempt * 1000 + index
            customer_file = coverage / f'{attempt}_{index}_customers.psv'
            destination = output / 'raw' / f'coverage_{attempt}_{index}.psv'
            if destination.exists():
                raise ValueError('Coverage retry output exists; inspect the incomplete run before retrying.')
            with customer_file.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter='|')
                writer.writeheader()
                writer.writerows(selected)
            retries.append({'profile': name, 'customer_count': len(selected), 'seed': seed,
                            'customer_file': str(customer_file.relative_to(output)),
                            'transaction_file': str(destination.relative_to(output))})
            tasks.append((source, customer_file, name, datetime.fromisoformat(metadata['start']),
                          datetime.fromisoformat(metadata['end']), destination, seed))
        print(f'Retrying {len(missing)} customers with no transactions', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(generate_profile, tasks))
    raise ValueError('Some customers still have zero transactions after five retry passes.')


def prepare(args):
    import pandas as pd
    from core import validate_input
    from shared.project_data import quality_checks

    output = args.output.resolve()
    if (output / 'dataset_report.json').exists() or list(output.glob('transactions_*.csv')):
        raise ValueError('Prepared dataset already exists; preserve it and select a new folder.')
    metadata = json.loads((output / 'generation.json').read_text())
    staging = output / 'staging'
    staging.mkdir(exist_ok=False)
    names = ['trans_date', 'trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last',
             'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob',
             'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud', 'profile']
    final_columns = ['trans_date_trans_time'] + [c for c in names if c not in ['trans_date', 'trans_time', 'profile']]
    cards, per_profile, years = set(), {}, set()
    earliest, latest, total, fraud = None, None, 0, 0
    for path in sorted((output / 'raw').glob('*.psv')):
        print(f'Validating and partitioning {path.name}', flush=True)
        for chunk in pd.read_csv(path, sep='|', usecols=names, dtype={'cc_num': 'string', 'zip': 'string'}, chunksize=100000):
            chunk['trans_date_trans_time'] = chunk.trans_date + ' ' + chunk.trans_time
            stamps = pd.to_datetime(chunk.trans_date_trans_time)
            # pandas 3 may use microsecond datetimes. Convert explicitly to
            # nanoseconds before dividing, so pandas 2 and 3 export epoch seconds.
            chunk['unix_time'] = stamps.dt.tz_localize('UTC').astype('datetime64[ns, UTC]').astype('int64') // 10**9
            chunk['merchant'] = 'fraud_' + chunk.merchant.astype(str)
            validate_input(chunk, history=True)
            quality, _ = quality_checks(chunk)
            bad = {key: value for key, value in quality.items() if value and
                   (key.startswith('invalid_') or key in ['missing_cells', 'duplicate_rows', 'duplicate_transaction_ids',
                    'nonpositive_amounts', 'negative_city_population', 'birth_after_transaction'])}
            if bad:
                raise ValueError(f'Generated quality failure: {bad}')
            if stamps.min() < pd.Timestamp(metadata['start']) or stamps.max() >= pd.Timestamp(metadata['end']) + pd.Timedelta(days=1):
                raise ValueError('Transaction outside requested dates.')
            earliest = min(earliest or chunk.trans_date_trans_time.min(), chunk.trans_date_trans_time.min())
            latest = max(latest or chunk.trans_date_trans_time.max(), chunk.trans_date_trans_time.max())
            cards.update(chunk.cc_num)
            total += len(chunk)
            fraud += int(chunk.is_fraud.sum())
            for profile, group in chunk.groupby('profile'):
                counts = per_profile.setdefault(profile, {'cards': set(), 'rows': 0, 'fraud': 0})
                counts['cards'].update(group.cc_num)
                counts['rows'] += len(group)
                counts['fraud'] += int(group.is_fraud.sum())
            for year, group in chunk.groupby(stamps.dt.year):
                years.add(int(year))
                target = staging / f'{year}.csv'
                group[final_columns].to_csv(target, index=False, mode='a', header=not target.exists())
    if {name: len(values['cards']) for name, values in per_profile.items()} != metadata['profile_customer_counts']:
        raise ValueError('Customer/profile coverage does not match generation settings.')
    overlaps = {}
    checks = [args.original_dir / 'fraudTrain.csv', args.original_dir / 'fraudTest.csv']
    pilot = ROOT / 'data/synthetic_pilot_2021_01/pilot_transactions.csv'
    if pilot.exists():
        checks.append(pilot)
    for source in checks:
        found = set()
        for part in pd.read_csv(source, usecols=['cc_num'], dtype='string', chunksize=250000):
            found.update(cards & set(part.cc_num))
        overlaps[source.name] = len(found)
    if any(overlaps.values()):
        raise ValueError(f'Card numbers overlap previous datasets: {overlaps}. Use another seed.')
    summaries, seen_ids = {}, set()
    for year in sorted(years):
        print(f'Sorting and saving year {year}', flush=True)
        frame = pd.read_csv(staging / f'{year}.csv', dtype={'cc_num': 'string', 'zip': 'string'})
        identifiers = set(frame.trans_num)
        if len(identifiers) != len(frame) or seen_ids & identifiers:
            raise ValueError('Duplicate transaction IDs across generated chunks/years.')
        seen_ids.update(identifiers)
        frame = frame.sort_values(['trans_date_trans_time', 'trans_num'], kind='stable').reset_index(drop=True)
        path = output / f'transactions_{year}.csv'
        frame.to_csv(path, index=False)
        summaries[str(year)] = {'file': path.name, 'rows': len(frame), 'cards': int(frame.cc_num.nunique()),
            'fraud': int(frame.is_fraud.sum()), 'fraud_percent': float(frame.is_fraud.mean() * 100),
            'first_transaction': frame.trans_date_trans_time.min(), 'last_transaction': frame.trans_date_trans_time.max(),
            'bytes': path.stat().st_size, 'sha256': digest(path)}
        # The exact staged file has now been checked and exported. Remove only
        # this explicit file inside the known output staging directory.
        (staging / f'{year}.csv').unlink()
        del frame
    staging.rmdir()
    report = {'generation': metadata, 'rows': total, 'cards': len(cards), 'profiles': len(per_profile),
        'fraud': fraud, 'fraud_percent': fraud / total * 100,
        'first_transaction': earliest, 'last_transaction': latest, 'previous_card_overlap': overlaps,
        'quality': 'All rows passed input/quality validation; transaction IDs unique across all years.',
        'years': summaries, 'profile_counts': {name: {**values, 'cards': len(values['cards'])} for name, values in per_profile.items()},
        'customers_sha256': digest(output / 'customers.psv'), 'preparation_code_sha256': digest(__file__),
        'model_training': False, 'model_evaluation': False,
        'notes': ['All transactions are newly simulated, including dates later than today.',
                  'Year files share the same new customers; they are not train/test splits.',
                  'Earlier years can supply history; future transactions must never be history for earlier ones.']}
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['rows', 'cards', 'profiles', 'fraud', 'fraud_percent', 'years']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--complete-customers', action='store_true', help='Complete zero-row customer coverage after an interrupted generation/preparation.')
    parser.add_argument('--generator-dir', type=Path, default=ROOT / 'data/Sparkov_Data_Generation')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/synthetic_2022_2026')
    parser.add_argument('--start', type=datetime.fromisoformat, default=datetime(2022, 1, 1))
    parser.add_argument('--end', type=datetime.fromisoformat, default=datetime(2026, 12, 31))
    parser.add_argument('--customers', type=int, default=999)
    parser.add_argument('--customer-file', type=Path, help='Reuse saved pipe-delimited Sparkov customer records unchanged.')
    parser.add_argument('--customers-per-profile', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=4)
    parser.add_argument('--seed', type=int, default=20221027)
    parser.add_argument('--original-dir', type=Path, default=ROOT.parents[1] / '01_Assignments/fraud')
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    arguments.generator_dir = arguments.generator_dir.resolve()
    if arguments.customer_file:
        arguments.customer_file = arguments.customer_file.resolve()
    if arguments.worker:
        worker(arguments)
        complete_customers(arguments)
    elif arguments.complete_customers:
        complete_customers(arguments)
    else:
        prepare(arguments)
