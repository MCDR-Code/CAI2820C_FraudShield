import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import score_transactions, validate_input
from history import HISTORY_FEATURES, add_history, history_with_context
from train import make_pipeline


def stream(n=60, seed=0):
    rng = np.random.RandomState(seed)
    start = pd.Timestamp('2020-01-01')
    times = start + pd.to_timedelta(np.sort(rng.randint(0, 20 * 86400, n)) + np.arange(n), unit='s')
    return pd.DataFrame({
        'trans_date_trans_time': times.strftime('%Y-%m-%d %H:%M:%S'), 'dob': '1980-01-01',
        'cc_num': rng.choice(['a', 'b', 'c'], n), 'merchant': rng.choice(['m1', 'm2', 'm3'], n),
        'amt': rng.uniform(1, 500, n).round(2), 'city_pop': 1000, 'lat': 30., 'long': -80.,
        'merch_lat': 30 + rng.uniform(-1, 1, n), 'merch_long': -80 + rng.uniform(-1, 1, n),
        'category': rng.choice(['grocery_pos', 'misc_net'], n), 'gender': rng.choice(['F', 'M'], n),
        'state': 'FL'})


class HistoryTests(unittest.TestCase):
    def test_known_values(self):
        df = pd.DataFrame({
            'trans_date_trans_time': ['2020-01-01 00:00:00', '2020-01-01 01:00:00', '2020-01-03 00:00:00'],
            'cc_num': ['a'] * 3, 'merchant': ['m1', 'm1', 'm2'], 'category': ['x', 'x', 'y'],
            'amt': [10., 30., 40.], 'merch_lat': [0., 0., 0.], 'merch_long': [0., 0., 0.]})
        h = add_history(df)
        self.assertEqual(h.hist_prior_count.tolist(), [0, 1, 2])
        np.testing.assert_allclose(h.hist_amt_to_prior_mean, [1.0, 3.0, 2.0])
        np.testing.assert_allclose(h.hist_amt_to_prior_max, [1.0, 3.0, 40 / 30])
        self.assertEqual(h.hist_count_24h.tolist(), [0, 1, 0])
        self.assertEqual(h.hist_new_merchant.tolist(), [1, 0, 1])
        self.assertEqual(h.hist_new_category.tolist(), [1, 0, 1])
        self.assertAlmostEqual(h.hist_log_secs_since_prev[1], np.log1p(3600))

    def test_recent_activity_and_habit_values(self):
        df = pd.DataFrame({
            'trans_date_trans_time': ['2020-01-01 10:00:00', '2020-01-01 10:30:00', '2020-01-01 23:00:00',
                                      '2020-01-01 23:10:00'],
            'cc_num': ['a'] * 4, 'merchant': ['m1'] * 4, 'category': ['x', 'x', 'y', 'x'],
            'amt': [10., 30., 20., 50.], 'merch_lat': [0.] * 4, 'merch_long': [0.] * 4})
        h = add_history(df)
        self.assertEqual(h.hist_count_1h.tolist(), [0, 1, 0, 1])
        self.assertEqual(h.hist_count_6h.tolist(), [0, 1, 0, 1])
        self.assertEqual(h.hist_night_count_72h.tolist(), [0, 0, 0, 1])
        np.testing.assert_allclose(h.hist_category_share, [0, 1, 0, 2 / 3])
        np.testing.assert_allclose(h.hist_hour_share, [0, 1, 0, 1 / 3])
        np.testing.assert_allclose(h.hist_amt_to_category_mean, [1.0, 3.0, 1.0, 50 / 20])
        self.assertEqual(h.hist_amt_zscore[0], 0.0)
        self.assertAlmostEqual(h.hist_amt_zscore[2], (20 - 20) / 10)  # prior amounts 10, 30: mean 20, std 10
        self.assertAlmostEqual(h.hist_log_amt_sum_24h[3], np.log1p(10 + 30 + 20))

    def test_no_lookahead(self):
        df = stream()
        full = add_history(df)
        cut = 30
        altered = df.copy()
        altered.loc[cut:, 'amt'] *= 1000  # change only the future
        altered.loc[cut:, 'merchant'] = 'zzz'
        pd.testing.assert_frame_equal(full.iloc[:cut], add_history(altered).iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], add_history(df.iloc[:cut]))

    def test_independent_of_row_order(self):
        df = stream()
        shuffled = df.sample(frac=1, random_state=3)
        a = add_history(df)
        b = add_history(shuffled)
        pd.testing.assert_frame_equal(a.loc[shuffled.index].reset_index(drop=True), b.reset_index(drop=True))

    def test_context_matches_full_stream(self):
        df = stream()
        context, new = df.iloc[:40], df.iloc[40:]
        pd.testing.assert_frame_equal(history_with_context(context, new), add_history(df).iloc[40:])

    def test_batch_mode_matches_precomputed_mode(self):
        df = stream(200)
        y = (np.arange(200) % 7 == 0).astype(int)
        validated = validate_input(df, history=True)
        pre = make_pipeline(LogisticRegression(solver='liblinear'), history=True)
        pre.fit(validated.join(add_history(validated)), y)
        expected = pre.predict_proba(validated.join(add_history(validated)))[:, 1]
        pre.set_params(features__history='batch')
        scored = score_transactions(df, {'pipeline': pre, 'threshold': .5, 'metadata': {}})
        np.testing.assert_allclose(scored.sort_values('row_number').model_score, expected)

    def test_history_inputs_required(self):
        df = stream()
        for column in ['cc_num', 'merchant']:
            with self.assertRaises(ValueError):
                validate_input(df.drop(columns=column), history=True)
            validate_input(df.drop(columns=column), history=False)
        self.assertEqual(len(set(HISTORY_FEATURES)), len(HISTORY_FEATURES))


if __name__ == '__main__':
    unittest.main()
