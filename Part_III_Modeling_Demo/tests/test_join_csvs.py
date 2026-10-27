import contextlib
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from join_csvs import join


class JoinTests(unittest.TestCase):
    def test_original_index_reordered_headers_and_leading_zero_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'original.csv').write_text(',cc_num,amt\n0,00012,1.20\n')
            (root / 'year.csv').write_text('amt,cc_num\n2.30,00034\n')
            with contextlib.redirect_stdout(io.StringIO()):
                result = join([root / 'original.csv', root / 'year.csv'], root / 'joined.csv', drop_export_index=True)
            with (root / 'joined.csv').open(newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [['cc_num', 'amt'], ['00012', '1.20'], ['00034', '2.30']])
            self.assertEqual(result['rows'], 2)

    def test_glob_exclusion_and_safe_replacement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '2025.csv').write_text('cc_num\n001\n')
            (root / '2026.csv').write_text('cc_num\n002\n')
            output = root / 'joined.csv'
            output.write_text('previous output')
            with contextlib.redirect_stdout(io.StringIO()):
                result = join([str(root / '*.csv')], output, exclude=['2026.csv'], replace_existing=True)
            self.assertEqual(result['rows'], 1)
            self.assertEqual(list((root / 'archive').glob('*/joined.csv'))[0].read_text(), 'previous output')

    def test_mismatched_columns_publish_no_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.csv').write_text('a\n1\n')
            (root / 'b.csv').write_text('b\n2\n')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Columns differ'):
                join([root / 'a.csv', root / 'b.csv'], root / 'joined.csv')
            self.assertFalse((root / 'joined.csv').exists())
            self.assertFalse((root / 'joined.csv.partial').exists())
