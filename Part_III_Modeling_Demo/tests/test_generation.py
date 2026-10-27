"""Contract checks for full-data export: chronology, epoch, overlap and IDs."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import generate_sparkov as generator


class FullGenerationTests(unittest.TestCase):
    def fixture(self, directory):
        output, original = directory / 'generated', directory / 'original'
        (output / 'raw').mkdir(parents=True)
        original.mkdir()
        for name in ['fraudTrain.csv', 'fraudTest.csv']:
            pd.DataFrame({'cc_num': ['original-card']}).to_csv(original / name, index=False)
        metadata = {'start': '2022-01-01', 'end': '2026-12-31',
                    'profile_customer_counts': {'sample.json': 1}}
        (output / 'generation.json').write_text(json.dumps(metadata))
        (output / 'customers.psv').write_text('fixture customer file')
        rows = []
        for index, year in enumerate([2026, 2022]):
            rows.append({'trans_date': f'{year}-01-01', 'trans_time': '00:00:00',
                'cc_num': 'new-card', 'merchant': 'Example', 'category': 'shopping_net', 'amt': 10,
                'first': 'Example', 'last': 'Person', 'gender': 'F', 'street': 'Example street',
                'city': 'Example city', 'state': 'FL', 'zip': '00123', 'lat': 25.8, 'long': -80.2,
                'city_pop': 10000, 'job': 'Example job', 'dob': '1990-01-01',
                'trans_num': f'id-{index}', 'unix_time': 0, 'merch_lat': 25.9, 'merch_long': -80.1,
                'is_fraud': index, 'profile': 'sample.json'})
        pd.DataFrame(rows).to_csv(output / 'raw/sample.psv', sep='|', index=False)
        return SimpleNamespace(output=output, original_dir=original)

    def test_export_keeps_rows_and_correct_epoch(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            with patch.object(generator, 'ROOT', Path(folder)), contextlib.redirect_stdout(io.StringIO()):
                generator.prepare(args)
            report = json.loads((args.output / 'dataset_report.json').read_text())
            self.assertEqual(report['rows'], 2)
            self.assertEqual(report['cards'], 1)
            early = pd.read_csv(args.output / 'transactions_2022.csv', dtype={'zip': 'string'})
            self.assertEqual(int(early.unix_time.iloc[0]), 1640995200)
            self.assertEqual(early.zip.iloc[0], '00123')
            self.assertEqual(early.merchant.iloc[0], 'fraud_Example')
            self.assertFalse((args.output / 'staging').exists())
            self.assertFalse(report['model_training'])

    def test_rejects_existing_card_ids(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            pd.DataFrame({'cc_num': ['new-card']}).to_csv(args.original_dir / 'fraudTest.csv', index=False)
            with patch.object(generator, 'ROOT', Path(folder)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    generator.prepare(args)

    def test_rejects_duplicate_ids_across_years(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            raw = args.output / 'raw/sample.psv'
            frame = pd.read_csv(raw, sep='|')
            frame['trans_num'] = 'repeated-id'
            frame.iloc[:1].to_csv(raw, sep='|', index=False)
            frame.iloc[1:].to_csv(args.output / 'raw/second.psv', sep='|', index=False)
            with patch.object(generator, 'ROOT', Path(folder)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Duplicate transaction IDs'):
                    generator.prepare(args)
