"""Join selected CSVs with one header, streaming rows without changing values.

Example: python join_csvs.py --output combined.csv --inputs a.csv b.csv

Patterns are supported: --inputs "folder/*.csv" --exclude "*2026.csv"

Use --drop-export-index to combine original Sparkov (extra index column) with
generated yearly files. Source files are never changed; output cannot overwrite
an existing file unless --replace-existing archives it. Files retain the supplied
order; patterns expand alphabetically. Writes a small checksum/count manifest.

This only joins files. It does not train, sort rows, sample, or split train/test.
"""

import argparse
import csv
from datetime import datetime, timezone
import fnmatch
import glob
import hashlib
import json
from pathlib import Path
import sys


def join(inputs, output, exclude=(), drop_export_index=False, replace_existing=False):
    output = Path(output).resolve()
    if output.exists() and not replace_existing:
        raise ValueError(f'Output already exists: {output}. Choose a new filename.')
    files = []
    for pattern in inputs:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(f'No CSV files match: {pattern}')
        for match in matches:
            path = Path(match).resolve()
            if path == output or any(fnmatch.fnmatch(path.name, item) or fnmatch.fnmatch(str(path), item) for item in exclude):
                continue
            if not path.is_file() or path.suffix.lower() != '.csv':
                raise ValueError(f'Expected a CSV file: {path}')
            if path not in files:
                files.append(path)
    if not files:
        raise ValueError('No input files remain after exclusions.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    if temporary.exists():
        raise ValueError(f'Incomplete output already exists: {temporary}')
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    columns, total, counts = None, 0, []
    try:
        with temporary.open('x', newline='', encoding='utf-8') as target:
            writer = csv.writer(target)
            for path in files:
                count = 0
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    reader = csv.reader(stream)
                    original_header = next(reader, None)
                    if not original_header:
                        raise ValueError(f'CSV is empty: {path}')
                    skip = int(drop_export_index and original_header[0] in ['', 'Unnamed: 0', 'Unnamed:0'])
                    header = original_header[skip:]
                    if any(not name.strip() for name in header) or len(set(header)) != len(header):
                        raise ValueError(f'Blank/duplicate headings in {path}. For Sparkov export indexes use --drop-export-index.')
                    if columns is None:
                        columns = header
                        writer.writerow(columns)
                    if set(header) != set(columns):
                        raise ValueError(f'Columns differ in {path}: missing={sorted(set(columns)-set(header))}, extra={sorted(set(header)-set(columns))}')
                    positions = [original_header.index(name) for name in columns]
                    for row in reader:
                        if not row:
                            continue
                        if len(row) != len(original_header):
                            raise ValueError(f'Wrong column count in {path}, CSV line {reader.line_num}')
                        writer.writerow([row[index] for index in positions])
                        count += 1
                total += count
                counts.append((path.name, count))
                print(f'{path.name}: {count:,} rows', flush=True)
        if output.exists():
            if not replace_existing:
                raise ValueError('Output appeared while joining; refusing to overwrite it.')
            archive = output.parent / 'archive' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive.mkdir(parents=True)
            backup = archive / output.name
            if not backup.resolve().is_relative_to(output.parent.resolve()):
                raise ValueError('Archive must remain inside the output folder.')
            output.rename(backup)
            print(f'Previous output archived: {backup}', flush=True)
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(f'Saved {total:,} rows from {len(files)} files to {output}', flush=True)
    checksum = hashlib.sha256()
    with output.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    manifest = {'rows': total, 'files': counts, 'output': str(output), 'sha256': checksum.hexdigest(),
                'inputs': [str(path) for path in files], 'operation': 'CSV concatenation only; no model training.'}
    output.with_suffix('.join.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--drop-export-index', action='store_true')
    parser.add_argument('--replace-existing', action='store_true', help='Archive the previous output, then replace it with the newly joined file.')
    args = parser.parse_args()
    try:
        join(args.inputs, args.output, args.exclude, args.drop_export_index, args.replace_existing)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')
