"""Join original training plus 2021-2025; reserve 2026 for test.

Creates fraudTrain.csv and fraudTest.csv under data/expanded_training_2021_2026.
Uses the general join_csvs.py tool. Verifies generated source hashes.
--replace-existing archives previous prepared outputs before publishing.
No model is trained; original Kaggle data and original notebooks are unchanged.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from join_csvs import join

ROOT = Path(__file__).resolve().parent


def write_split(source, destination, report, years, filename):
    temporary = destination / (filename + '.partial')
    details_by_year = {year: report['years'][year] for year in years}
    header, previous_end, combined_hash = None, None, hashlib.sha256()
    with temporary.open('xb') as output:
        for year, details in details_by_year.items():
            if previous_end is not None and previous_end >= details['first_transaction']:
                raise ValueError('Year files overlap or are not chronological.')
            previous_end = details['last_transaction']
            checksum = hashlib.sha256()
            with (source / details['file']).open('rb') as stream:
                first = stream.readline()
                checksum.update(first)
                if header is None:
                    header = first
                    output.write(first)
                    combined_hash.update(first)
                elif first != header:
                    raise ValueError('Yearly CSV headers differ.')
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
                    combined_hash.update(chunk)
                    output.write(chunk)
            if checksum.hexdigest() != details['sha256']:
                raise ValueError(f'Year {year} source checksum changed.')
            print(f'{filename}: included {year}, {details["rows"]:,} rows', flush=True)
    return {'file': filename, 'rows': sum(item['rows'] for item in details_by_year.values()),
            'fraud': sum(item['fraud'] for item in details_by_year.values()),
            'years': details_by_year, 'bytes': temporary.stat().st_size,
            'sha256': combined_hash.hexdigest()}


def combine(replace_existing=False):
    source = ROOT / 'data/synthetic_2022_2026'
    source_2021 = ROOT / 'data/synthetic_2021_full'
    destination = ROOT / 'data/expanded_training_2021_2026'
    destination.mkdir(exist_ok=True)
    if (destination / 'fraudTrain.csv').exists() and not replace_existing:
        raise ValueError('Prepared files exist. Use --replace-existing to archive and rebuild them.')
    report = json.loads((source / 'dataset_report.json').read_text())
    year_2021 = json.loads((source_2021 / 'dataset_report.json').read_text())['years']['2021']
    inputs = [ROOT.parents[1] / '01_Assignments/fraud/fraudTrain.csv', source_2021 / year_2021['file']]
    inputs.extend(source / report['years'][str(year)]['file'] for year in range(2022, 2026))
    for path, details in [(source_2021 / year_2021['file'], year_2021),
                          *[(source / item['file'], item) for item in report['years'].values()]]:
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != details['sha256']:
            raise ValueError(f'Generated source checksum changed: {path}')
    if report['years']['2025']['last_transaction'] >= report['years']['2026']['first_transaction']:
        raise ValueError('Test transactions must be strictly later than training data.')
    training = join(inputs, destination / 'fraudTrain.csv', drop_export_index=True, replace_existing=replace_existing)
    testing = write_split(source, destination, report, ['2026'], 'fraudTest.csv')
    existing = [destination / name for name in ['fraudTest.csv', 'training_data_manifest.json']
                if (destination / name).exists()]
    if existing:
        archive = destination / 'archive' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for old in existing:
            new = archive / old.name
            if not old.resolve().is_relative_to(destination.resolve()) or not new.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive paths must stay inside the experiment folder.')
            old.rename(new)
        print(f'Previous prepared files archived at: {archive}', flush=True)
    (destination / 'fraudTest.csv.partial').rename(destination / 'fraudTest.csv')
    metadata = {**training, 'test_file': testing['file'], 'test_rows': testing['rows'],
        'test_sha256': testing['sha256'], 'test': testing, 'original_data_included': True,
        'original_files_modified': False, 'test_status': '2026 reserved for test; excluded from fitting and validation.',
        'exploratory_note': 'Year files have been available for exploratory scoring; this is not a pristine untouched holdout.',
        'training_years': [2019, 2020, 2021, 2022, 2023, 2024, 2025]}
    (destination / 'training_data_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'training_rows': training['rows'], 'training_years': metadata['training_years'],
                      'test_rows': testing['rows'], 'test_years': list(testing['years'])}, indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-existing', action='store_true', help='Archive previous prepared files and rebuild the agreed split.')
    combine(parser.parse_args().replace_existing)
