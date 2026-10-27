"""All 17 history features against a slow, obviously-correct reference.

The reference follows the rule literally: a row may use only rows of the same card
with a STRICTLY EARLIER timestamp. Rows sharing a timestamp never see each other.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import (DAY, GAP_CAP, HISTORY_FEATURES, HOUR, _haversine_km, add_history,
                     history_with_context)


def tied_stream(n=160, seed=1):
    """Random card activity on an hourly grid so many rows share a (card, timestamp)."""
    rng = np.random.RandomState(seed)
    slots = np.sort(rng.randint(0, 6 * 24, n))
    times = pd.Timestamp('2020-01-01') + pd.to_timedelta(slots * HOUR + rng.choice([0, 0, 0, 1800], n), unit='s')
    return pd.DataFrame({
        'trans_date_trans_time': times.strftime('%Y-%m-%d %H:%M:%S'),
        'cc_num': rng.choice(['a', 'b', 'c'], n), 'merchant': rng.choice(['m1', 'm2', 'm3', 'm4'], n),
        'category': rng.choice(['x', 'y', 'z'], n), 'amt': rng.uniform(1, 400, n).round(2),
        'merch_lat': 30 + rng.uniform(-1, 1, n), 'merch_long': -80 + rng.uniform(-1, 1, n)})


def reference(df):
    df = df.reset_index(drop=True)
    time = pd.to_datetime(df.trans_date_trans_time, utc=True).to_numpy('datetime64[s]').astype(np.int64)
    hour = (time % DAY) // HOUR

    def earlier(i):
        return np.flatnonzero((df.cc_num.to_numpy() == df.cc_num[i]) & (time < time[i]))

    prior = [earlier(i) for i in range(len(df))]
    amt = df.amt.to_numpy()
    high = np.zeros(len(df))
    for i, s in enumerate(prior):
        high[i] = len(s) >= 5 and amt[i] / amt[s].mean() > 3
    night = ((hour >= 22) | (hour <= 3)).astype(float)
    rows = []
    for i, s in enumerate(prior):
        row = dict.fromkeys(HISTORY_FEATURES, 0.0)
        row['hist_prior_count'] = len(s)
        if len(s):
            last = time[s].max()
            tail = s[time[s] == last]
            row['hist_amt_to_prior_mean'] = amt[i] / amt[s].mean()
            row['hist_amt_to_prior_max'] = amt[i] / amt[s].max()
            row['hist_log_secs_since_prev'] = np.log1p(min(time[i] - last, GAP_CAP))
            row['hist_dist_from_prev_km'] = float(_haversine_km(
                df.merch_lat[tail].mean(), df.merch_long[tail].mean(), df.merch_lat[i], df.merch_long[i]))
            same_category = s[df.category.to_numpy()[s] == df.category[i]]
            row['hist_category_share'] = len(same_category) / len(s)
            row['hist_hour_share'] = (hour[s] == hour[i]).sum() / len(s)
            row['hist_amt_to_category_mean'] = amt[i] / amt[same_category].mean() if len(same_category) else 1.0
            if len(s) >= 2:
                spread = np.sqrt(max((amt[s] ** 2).mean() - amt[s].mean() ** 2, 0))
                row['hist_amt_zscore'] = float(np.clip((amt[i] - amt[s].mean()) / max(spread, 1e-9), -10, 50))
        else:
            row['hist_amt_to_prior_mean'] = row['hist_amt_to_prior_max'] = row['hist_amt_to_category_mean'] = 1.0
            row['hist_log_secs_since_prev'] = np.log1p(GAP_CAP)
        within = lambda seconds: s[time[s] >= time[i] - seconds]
        row['hist_count_24h'], row['hist_count_1h'], row['hist_count_6h'] = (len(within(x)) for x in (DAY, HOUR, 6 * HOUR))
        row['hist_log_amt_sum_24h'] = np.log1p(amt[within(DAY)].sum())
        row['hist_high_amt_count_72h'] = high[within(3 * DAY)].sum()
        row['hist_night_count_72h'] = night[within(3 * DAY)].sum()
        row['hist_new_merchant'] = float(not (df.merchant.to_numpy()[s] == df.merchant[i]).any())
        row['hist_new_category'] = float(not (df.category.to_numpy()[s] == df.category[i]).any())
        rows.append(row)
    return pd.DataFrame(rows, columns=HISTORY_FEATURES)


class StrictHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = tied_stream()
        pairs = cls.frame.groupby(['cc_num', 'trans_date_trans_time']).size()
        cls.tied_groups = int((pairs > 1).sum())

    def test_data_really_contains_ties(self):
        self.assertGreater(self.tied_groups, 10)

    def test_all_17_features_match_the_literal_rule(self):
        for seed in (1, 2, 3):
            frame = tied_stream(seed=seed)
            actual, expected = add_history(frame).reset_index(drop=True), reference(frame)
            for column in HISTORY_FEATURES:
                with self.subTest(seed=seed, column=column):
                    np.testing.assert_allclose(actual[column], expected[column], rtol=1e-9, atol=1e-9)

    def test_row_order_never_matters(self):
        shuffled = self.frame.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(add_history(self.frame).loc[shuffled.index].reset_index(drop=True),
                                      add_history(shuffled).reset_index(drop=True))

    def test_changing_a_simultaneous_row_changes_nothing_for_its_twin(self):
        groups = self.frame.groupby(['cc_num', 'trans_date_trans_time']).indices
        first, second = next(v for v in groups.values() if len(v) > 1)[:2]
        altered = self.frame.copy()
        altered.loc[second, ['amt', 'merchant', 'category']] = [9999., 'zzz', 'qqq']
        pd.testing.assert_series_equal(add_history(self.frame).iloc[first], add_history(altered).iloc[first])

    def test_no_lookahead_for_any_feature(self):
        cut = 100
        times = pd.to_datetime(self.frame.trans_date_trans_time)
        earlier = self.frame[times < times.iloc[cut]]
        altered = self.frame.copy()
        future = times >= times.iloc[cut]
        altered.loc[future, 'amt'] *= 500
        altered.loc[future, 'merchant'] = 'zzz'
        pd.testing.assert_frame_equal(add_history(self.frame).loc[earlier.index],
                                      add_history(altered).loc[earlier.index])

    def test_context_batches_match_one_continuous_stream(self):
        times = pd.to_datetime(self.frame.trans_date_trans_time)
        boundary = times.sort_values().iloc[80]
        context, review = self.frame[times < boundary], self.frame[times >= boundary]
        pd.testing.assert_frame_equal(history_with_context(context, review),
                                      add_history(self.frame).loc[review.index])


if __name__ == '__main__':
    unittest.main()
