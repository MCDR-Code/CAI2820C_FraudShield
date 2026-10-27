"""Per-card history features built only from strictly earlier transactions."""

import numpy as np
import pandas as pd

HISTORY_INPUTS = ['cc_num', 'merchant']
HISTORY_FEATURES = ['hist_prior_count', 'hist_amt_to_prior_mean', 'hist_amt_to_prior_max',
                    'hist_log_secs_since_prev', 'hist_count_24h', 'hist_new_merchant',
                    'hist_new_category', 'hist_dist_from_prev_km',
                    'hist_count_1h', 'hist_count_6h', 'hist_log_amt_sum_24h', 'hist_high_amt_count_72h',
                    'hist_night_count_72h', 'hist_category_share', 'hist_hour_share',
                    'hist_amt_zscore', 'hist_amt_to_category_mean']
HOUR = 3600
DAY = 86400
GAP_CAP = 30 * DAY
KEY_SPAN = 2 ** 40


def _haversine_km(lat1, lon1, lat2, lon2):
    a, b, c, d = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def add_history(df):
    """Return HISTORY_FEATURES for validated rows, aligned to df's row order.

    A row sees only earlier transactions on its card: no labels, no future rows.
    Rows are ordered by card, timestamp, then input position, so the result does
    not depend on how the input happens to be sorted.
    """
    n = len(df)
    ts = pd.to_datetime(df.trans_date_trans_time).to_numpy('datetime64[s]').astype(np.int64)
    card = pd.factorize(df.cc_num.astype(str))[0]
    order = np.lexsort((np.arange(n), ts, card))
    s_card, s_ts = card[order], ts[order]
    amt = df.amt.to_numpy(float)[order]
    mlat, mlon = df.merch_lat.to_numpy(float)[order], df.merch_long.to_numpy(float)[order]
    merchant = pd.factorize(df.merchant.astype(str))[0][order]
    category = pd.factorize(df.category.astype(str))[0][order]

    index = np.arange(n)
    first = np.r_[True, s_card[1:] != s_card[:-1]]
    start = np.maximum.accumulate(np.where(first, index, 0))
    prior_count = index - start

    cumulative = np.cumsum(amt)
    prior_sum = cumulative - amt - (cumulative[start] - amt[start])
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ratio = np.where(prior_count > 0, amt / (prior_sum / np.maximum(prior_count, 1)), 1.0)
    running_max = pd.Series(amt).groupby(s_card).cummax()
    prior_max = running_max.groupby(s_card).shift(1).to_numpy()
    max_ratio = np.where(np.isnan(prior_max), 1.0, amt / np.where(np.isnan(prior_max), 1.0, prior_max))

    gap = pd.Series(s_ts).groupby(s_card).diff().to_numpy()
    log_gap = np.log1p(np.where(np.isnan(gap), GAP_CAP, np.minimum(gap, GAP_CAP)))

    keys = s_card.astype(np.int64) * KEY_SPAN + (s_ts - s_ts.min())
    before = np.searchsorted(keys, keys, 'left')

    def window(seconds):
        return before, np.searchsorted(keys, keys - seconds, 'left')

    def window_sum(values, seconds):
        # Sum of `values` over this card's earlier rows inside the trailing window.
        prefix = np.r_[0.0, np.cumsum(values)]
        upper, lower = window(seconds)
        return prefix[upper] - prefix[lower]

    ones = np.ones(n)
    count_24h = window_sum(ones, DAY)
    count_1h, count_6h = window_sum(ones, HOUR), window_sum(ones, 6 * HOUR)
    amt_sum_24h = window_sum(amt, DAY)
    hour = (s_ts % DAY) // HOUR
    is_night = ((hour >= 22) | (hour <= 3)).astype(float)
    is_high = ((prior_count >= 5) & (mean_ratio > 3)).astype(float)
    high_72h, night_72h = window_sum(is_high, 3 * DAY), window_sum(is_night, 3 * DAY)

    with np.errstate(divide='ignore', invalid='ignore'):
        prior_mean = prior_sum / np.maximum(prior_count, 1)
        prior_sq = np.cumsum(amt ** 2)
        prior_sq = prior_sq - amt ** 2 - (prior_sq[start] - amt[start] ** 2)
        std = np.sqrt(np.clip(prior_sq / np.maximum(prior_count, 1) - prior_mean ** 2, 0, None))
        zscore = np.where(prior_count >= 2, np.clip((amt - prior_mean) / np.maximum(std, 1e-9), -10, 50), 0.0)

    frame = pd.DataFrame({'c': s_card, 'm': merchant, 'k': category})
    new_merchant = frame.groupby(['c', 'm']).cumcount().eq(0).to_numpy()
    new_category = frame.groupby(['c', 'k']).cumcount().eq(0).to_numpy()

    habit = pd.DataFrame({'c': s_card, 'k': category, 'h': hour, 'a': amt})
    category_n = habit.groupby(['c', 'k']).cumcount().to_numpy()
    hour_n = habit.groupby(['c', 'h']).cumcount().to_numpy()
    category_sum = habit.groupby(['c', 'k']).a.cumsum().to_numpy() - amt
    with np.errstate(divide='ignore', invalid='ignore'):
        category_share = np.where(prior_count > 0, category_n / np.maximum(prior_count, 1), 0.0)
        hour_share = np.where(prior_count > 0, hour_n / np.maximum(prior_count, 1), 0.0)
        category_ratio = np.where(category_n > 0, amt / (category_sum / np.maximum(category_n, 1)), 1.0)

    prev_lat = pd.Series(mlat).groupby(s_card).shift(1).to_numpy()
    prev_lon = pd.Series(mlon).groupby(s_card).shift(1).to_numpy()
    distance = np.nan_to_num(_haversine_km(prev_lat, prev_lon, mlat, mlon), nan=0.0)

    out = np.empty((n, len(HISTORY_FEATURES)))
    out[order] = np.column_stack([prior_count, mean_ratio, max_ratio, log_gap, count_24h,
                                  new_merchant, new_category, distance,
                                  count_1h, count_6h, np.log1p(amt_sum_24h), high_72h, night_72h,
                                  category_share, hour_share, zscore, category_ratio])
    return pd.DataFrame(out, columns=HISTORY_FEATURES, index=df.index)


def history_with_context(context, new):
    """History for `new` rows that may also see earlier `context` rows (e.g. the training period)."""
    joined = pd.concat([context, new], ignore_index=True)
    return add_history(joined).iloc[len(context):].set_axis(new.index)
