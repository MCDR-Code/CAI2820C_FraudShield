import io
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
from core import REQUIRED, score_transactions, validate_input
from train import choose_threshold, make_pipeline, metrics


def example(n=20):
    return pd.DataFrame({
        'trans_date_trans_time': ['2020-07-01 12:00:00'] * n, 'dob': ['1980-01-01'] * n,
        'amt': np.arange(1, n+1).astype(float), 'city_pop': [1000] * n,
        'lat': [30.] * n, 'long': [-80.] * n, 'merch_lat': [30.1] * n,
        'merch_long': [-80.1] * n, 'category': ['grocery_pos', 'misc_net'] * (n//2),
        'gender': ['F', 'M'] * (n//2), 'state': ['FL'] * n,
    })


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = example()
        cls.pipeline = make_pipeline(LogisticRegression(solver='liblinear'))
        cls.pipeline.fit(cls.frame, np.arange(20) % 2)
        cls.saved = {'pipeline': cls.pipeline, 'threshold': .5, 'metadata': {}}

    def test_invalid_input_rejected(self):
        for column, value in [('amt', 'bad'), ('amt', -1), ('lat', 91),
                              ('dob', '2030-01-01'), ('state', ''), ('city_pop', np.inf)]:
            with self.subTest(column=column, value=value):
                frame = self.frame.copy()
                frame[column] = frame[column].astype(object)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError): validate_input(frame)
        with self.assertRaises(ValueError): validate_input(self.frame.drop(columns='dob'))

    def test_label_and_identity_are_not_predictors(self):
        raw = self.frame.assign(is_fraud=0, first='PRIVATE', cc_num='0000000000000001')
        other = raw.assign(is_fraud=1, first='OTHER', cc_num='9999999999999999')
        a = score_transactions(raw, self.saved)
        b = score_transactions(other, self.saved)
        np.testing.assert_array_equal(a.model_score, b.model_score)
        self.assertTrue({'first','cc_num','is_fraud'}.isdisjoint(a.columns))
        self.assertEqual(len(a), len(raw))
        self.assertEqual(a.review_flag.sum(), (a.model_score >= .5).sum())

    def test_unknown_category_supported(self):
        frame = self.frame.copy(); frame['category'] = 'new_category'
        self.assertEqual(len(score_transactions(frame, self.saved)), 20)

    def test_threshold_is_reproducible_and_meets_feasible_targets(self):
        y = [0,0,0,0,0,1,1,1,1,1]; scores = np.arange(10)/10
        threshold = choose_threshold(y, scores)
        self.assertEqual(threshold, choose_threshold(y, scores))
        self.assertTrue(metrics(y, scores, threshold)['targets_met'])

    def test_upload_errors_never_return_sample_results(self):
        client = TestClient(web.app)
        for name, data in [('wrong.txt', b'anything'), ('empty.csv', b''),
                           ('missing.csv', b'amt\n10\n'), ('duplicate.csv', b'amt,amt\n1,2\n'),
                           ('broken.csv', b'"amt\n')]:
            response = client.post('/api/analyze', files={'file': (name, data)})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn('error', response.json())
            self.assertNotIn('rows', response.json())

    def test_valid_upload_and_numeric_strings(self):
        previous = web._bundle
        web._bundle = self.saved
        try:
            response = TestClient(web.app).post('/api/analyze', files={'file': ('transactions.csv', self.frame.to_csv(index=False).encode())})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['total'], 20)
        finally:
            web._bundle = previous


if __name__ == '__main__':
    unittest.main()
