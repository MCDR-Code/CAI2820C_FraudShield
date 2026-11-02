"""Correct a sampling bug in upstream Sparkov (commit b5eb45c), applied at run time.

Upstream ``Profile.closest_rand_parallel(r, i, j, obj)`` always compares column 1
(the date random number) instead of column ``i``. Category and shopping time are
therefore derived from the SAME random number as the date: early dates in the
simulation range always get the first categories in the profile (gas_transport,
grocery_*) and late dates the last ones (kids_pets, personal_care, travel). In a
multi-year run each year then contains only 4-6 legitimate categories, while fraud
(generated in a 1-day window) contains all 14. Upstream also assigns per-category
amounts after sorting by the shopping-time column instead of the category column.

This module replaces ``Profile.sample_from`` with a version that:
  * draws three INDEPENDENT uniform numbers per transaction (date, time of day, category);
  * maps each through its own cumulative-weight table;
  * assigns gamma-distributed amounts to the rows of the matching category.
Everything else (transaction counts, weights, fraud hours, travel, output format)
is unchanged. The upstream checkout itself is not modified.
"""

import numpy as np

PATCH_VERSION = 'sparkov-independent-sampling-v1'


def _lookup(values, cumsum_map):
    """Smallest cumulative key >= value (same rule as upstream), vectorized."""
    keys = np.array(sorted(cumsum_map.keys()))
    index = np.searchsorted(keys, values, side='left')
    index = np.minimum(index, len(keys) - 1)  # guard float rounding at 1.0
    return keys[index]


def sample_from(self, is_fraud):
    from datetime import datetime, time
    num_trans = int((self.end - self.start).days *
                    np.random.randint(self.profile['avg_transactions_per_day']['min'],
                                      self.profile['avg_transactions_per_day']['max'] + 1))
    travel_max = self.profile['travel_max_dist']
    is_traveling = False
    date_keys = _lookup(np.random.random(num_trans), self.proportions['date_prop'])
    time_keys = _lookup(np.random.random(num_trans), self.proportions['shopping_time'])
    cat_keys = _lookup(np.random.random(num_trans), self.proportions['categories_wt'])
    amounts = np.zeros(num_trans)
    for cat_key in np.unique(cat_keys):
        rows = np.flatnonzero(cat_keys == cat_key)
        specs = self.amt_specs[self.proportions['categories_wt'][cat_key]]
        draws = np.random.gamma(specs['shape'], specs['scale'], len(rows))
        lower = np.random.uniform(1.00, 10.00, len(rows))
        amounts[rows] = np.where(draws < 1, lower, draws)
    output, fraud_dates = [], []
    for i in range(num_trans):
        trans_num = self.fake.md5(raw_output=False)
        chosen_date = self.proportions['date_prop'][date_keys[i]]
        chosen_date_str = chosen_date.strftime('%Y-%m-%d')
        if is_fraud == 1:
            fraud_dates.append(chosen_date_str)
        chosen_cat = self.proportions['categories_wt'][cat_keys[i]]
        chosen_amt = '{:.2f}'.format(amounts[i])
        chosen_daypart = self.proportions['shopping_time'][time_keys[i]]
        hr, mn, sec = self.sample_time(chosen_daypart, is_fraud)
        stamp = datetime.combine(chosen_date, time(hour=hr, minute=mn, second=sec))
        output.append([str(trans_num), chosen_date_str, f'{hr:02d}:{mn:02d}:{sec:02d}', str(int(stamp.timestamp())),
                       str(chosen_cat), str(chosen_amt), str(is_fraud)])
    return output, is_traveling, travel_max, fraud_dates


def apply():
    """Patch the already-importable upstream Profile class in this process."""
    import profile_weights
    profile_weights.Profile.sample_from = sample_from
    return PATCH_VERSION
