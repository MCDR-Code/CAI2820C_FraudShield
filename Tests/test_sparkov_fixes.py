"""Regression checks for independent sampling and category-specific amounts."""

# Discover the project from any supported entry point; retain saved-model imports.
import sys as _sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / 'project_paths.py').is_file())
if str(_project) not in _sys.path:
    _sys.path.insert(0, str(_project))
from project_paths import PROJECT_ROOT
from datetime import datetime
from itertools import count
from types import SimpleNamespace
import unittest

import numpy as np
from sparkov_fixes import sample_from


class SamplerTests(unittest.TestCase):
    def generate(self):
        ids = count()
        profile = SimpleNamespace(
            start=datetime(2021, 1, 1), end=datetime(2021, 7, 20),
            profile={'avg_transactions_per_day': {'min': 20, 'max': 20}, 'travel_max_dist': 10},
            proportions={
                'date_prop': {.5: datetime(2021, 1, 1), 1.: datetime(2021, 6, 1)},
                'shopping_time': {.5: 'AM', 1.: 'PM'},
                'categories_wt': {.5: 'small', 1.: 'large'},
            },
            amt_specs={'small': {'shape': 100., 'scale': .1},
                       'large': {'shape': 100., 'scale': 10.}},
            fake=SimpleNamespace(md5=lambda **kwargs: str(next(ids))),
            sample_time=lambda daypart, fraud: (1 if daypart == 'AM' else 13, 0, 0),
        )
        np.random.seed(42)
        return sample_from(profile, 0)[0]

    def test_date_category_and_time_are_not_locked_together(self):
        rows = self.generate()
        for date in ('2021-01-01', '2021-06-01'):
            selected = [row for row in rows if row[1] == date]
            self.assertEqual({row[4] for row in selected}, {'small', 'large'})
            for category in ('small', 'large'):
                times = [row[2] for row in selected if row[4] == category]
                share = sum(time.startswith('01:') for time in times) / len(times)
                self.assertTrue(.4 < share < .6)

    def test_amounts_use_the_matching_category_and_repeat_with_seed(self):
        rows = self.generate()
        self.assertEqual(rows, self.generate())
        for category, expected in (('small', 10.), ('large', 1000.)):
            values = [float(row[5]) for row in rows if row[4] == category]
            self.assertLess(abs(np.mean(values) / expected - 1), .03)


if __name__ == '__main__':
    unittest.main()
