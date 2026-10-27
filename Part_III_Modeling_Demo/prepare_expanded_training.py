"""Combine verified new yearly files; preserve original data and submitted notebook.

Creates data/expanded_training_2022_2026/fraudTrain.csv using ONLY 2022-2026 new
transactions. Inputs are already chronological yearly partitions; hashes and
date boundaries are verified before publishing the combined file. No model is
trained. No old file is renamed or overwritten. The future test file is pending.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def combine():
    source = ROOT / 'data/synthetic_2022_2026'
    destination = ROOT / 'data/expanded_training_2022_2026'
    destination.mkdir(exist_ok=True)
    target = destination / 'fraudTrain.csv'
    if target.exists():
        raise ValueError('Combined training file exists; preserve it.')
    report = json.loads((source / 'dataset_report.json').read_text())
    temporary = destination / 'fraudTrain.csv.partial'
    header, previous_end, combined_hash = None, None, hashlib.sha256()
    with temporary.open('xb') as output:
        for year, details in sorted(report['years'].items()):
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
            print(f'Combined {year}: {details["rows"]:,} rows', flush=True)
    temporary.rename(target)
    metadata = {'file': target.name, 'rows': report['rows'], 'cards': report['cards'],
        'fraud': report['fraud'], 'years': report['years'], 'bytes': target.stat().st_size,
        'sha256': combined_hash.hexdigest(), 'original_data_included': False,
        'original_files_modified': False, 'test_file': None,
        'test_status': 'New future transactions for these same customers must be generated separately.',
        'exploratory_note': 'Year files have been explored/scored before fitting; do not call them an untouched test set.'}
    (destination / 'training_data_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({key: metadata[key] for key in ['file', 'rows', 'cards', 'bytes', 'sha256']}, indent=2))


if __name__ == '__main__':
    combine()
