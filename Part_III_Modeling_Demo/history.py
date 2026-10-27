"""Juan's per-card features, using strictly earlier timestamp groups only.

Simultaneous purchases never become one another's history. If the previous
timestamp has several merchants, distance uses their geographic centroid.
Labels and direct identifiers never enter the feature matrix.
"""

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
HISTORY_VERSION = 'strict-timestamp-groups-v3'


def _haversine_km(lat1, lon1, lat2, lon2):
    a, b, c, d = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _is_night(hour):
    return (hour >= 22) | (hour <= 3)


def add_history(df):
    """Return aligned features; every row sees only earlier times on its card."""
    if df.empty:
        return pd.DataFrame(index=df.index, columns=HISTORY_FEATURES, dtype=float)
    frame = pd.DataFrame({
        'card': df.cc_num.astype(str).to_numpy(),
        'time': pd.to_datetime(df.trans_date_trans_time, utc=True).to_numpy('datetime64[s]').astype(np.int64),
        'amount': df.amt.to_numpy(float), 'lat': df.merch_lat.to_numpy(float),
        'lon': df.merch_long.to_numpy(float), 'merchant': df.merchant.astype(str).to_numpy(),
        'category': df.category.astype(str).to_numpy(),
    })
    frame['square'] = frame.amount ** 2
    frame['hour'] = (frame.time % DAY) // HOUR
    grouped = frame.groupby(['card', 'time'], sort=True)
    group = grouped.agg(n=('amount', 'size'), total=('amount', 'sum'), maximum=('amount', 'max'),
                        square=('square', 'sum'), lat=('lat', 'mean'), lon=('lon', 'mean')).reset_index()
    gid = grouped.ngroup().to_numpy()
    cards = group.groupby('card', sort=False)
    prior_n = (cards.n.cumsum() - group.n).to_numpy(float)
    prior_sum = (cards.total.cumsum() - group.total).to_numpy(float)
    prior_square = (cards.square.cumsum() - group.square).to_numpy(float)
    maximum = cards.maximum.cummax().groupby(group.card, sort=False).shift().to_numpy()
    previous_lat = cards.lat.shift().to_numpy()
    previous_lon = cards.lon.shift().to_numpy()
    gap = cards.time.diff().to_numpy()
    group_times, card_positions = group.time.to_numpy(), cards.indices.values()

    def windowed(values, seconds):
        """Per timestamp group: sum of `values` over this card's strictly earlier groups in the trailing window."""
        values, result = np.asarray(values, float), np.zeros(len(group))
        for positions in card_positions:
            times = group_times[positions]
            prefix = np.r_[0.0, np.cumsum(values[positions])]
            result[positions] = prefix[:-1] - prefix[np.searchsorted(times, times - seconds, side='left')]
        return result

    def prior_by(keys):
        """Per row: count and amount total of strictly earlier timestamps with the same keys on this card."""
        parts = frame.groupby(keys + ['time'], sort=True)
        stats = parts.agg(n=('amount', 'size'), total=('amount', 'sum'))
        within = stats.groupby(level=list(range(len(keys))))
        ids = parts.ngroup().to_numpy()
        return ((within.n.cumsum() - stats.n).to_numpy(float)[ids],
                (within.total.cumsum() - stats.total).to_numpy(float)[ids])

    amount = frame.amount.to_numpy()
    count = prior_n[gid]
    mean = np.divide(prior_sum[gid], count, out=np.ones(len(df)), where=count > 0)
    mean_ratio = np.divide(amount, mean, out=np.ones(len(df)), where=count > 0)
    max_ratio = np.divide(amount, maximum[gid], out=np.ones(len(df)), where=np.isfinite(maximum[gid]))
    first_merchant = frame.groupby(['card', 'merchant']).time.transform('min')
    first_category = frame.groupby(['card', 'category']).time.transform('min')
    distance = np.nan_to_num(_haversine_km(previous_lat[gid], previous_lon[gid], frame.lat, frame.lon), nan=0.)

    # Recent activity: sums over strictly earlier timestamp groups inside a trailing window.
    is_high = ((count >= 5) & (mean_ratio > 3)).astype(float)
    high_per_group = np.bincount(gid, weights=is_high, minlength=len(group))
    night_per_group = group.n.to_numpy(float) * _is_night((group_times % DAY) // HOUR)
    count_1h, count_6h, count_24h = (windowed(group.n, s)[gid] for s in (HOUR, 6 * HOUR, DAY))
    log_amount_24h = np.log1p(windowed(group.total, DAY))[gid]
    high_72h, night_72h = windowed(high_per_group, 3 * DAY)[gid], windowed(night_per_group, 3 * DAY)[gid]

    # Card habits: how usual this category/hour/amount is for the card, from strictly earlier groups.
    category_count, category_total = prior_by(['card', 'category'])
    hour_count, _ = prior_by(['card', 'hour'])
    category_share = np.divide(category_count, count, out=np.zeros(len(df)), where=count > 0)
    hour_share = np.divide(hour_count, count, out=np.zeros(len(df)), where=count > 0)
    category_ratio = np.divide(amount, category_total / np.maximum(category_count, 1),
                               out=np.ones(len(df)), where=category_count > 0)
    average = prior_sum[gid] / np.maximum(count, 1)
    spread = np.sqrt(np.clip(prior_square[gid] / np.maximum(count, 1) - average ** 2, 0, None))
    zscore = np.where(count >= 2, np.clip((amount - average) / np.maximum(spread, 1e-9), -10, 50), 0.)

    values = np.column_stack([
        count, mean_ratio, max_ratio,
        np.log1p(np.where(np.isnan(gap[gid]), GAP_CAP, np.minimum(gap[gid], GAP_CAP))),
        count_24h, frame.time.eq(first_merchant), frame.time.eq(first_category), distance,
        count_1h, count_6h, log_amount_24h, high_72h, night_72h,
        category_share, hour_share, zscore, category_ratio,
    ])
    return pd.DataFrame(values, columns=HISTORY_FEATURES, index=df.index)


def history_with_context(context, new):
    """Context and review batches stay separate; labels are never consumed."""
    joined = pd.concat([context, new], ignore_index=True)
    return add_history(joined).iloc[len(context):].set_axis(new.index)


def validate_context_period(context, new):
    """Reject overlapping periods rather than count the same upload as its own past."""
    if context.empty:
        return
    latest = pd.to_datetime(context.trans_date_trans_time, utc=True).max()
    earliest = pd.to_datetime(new.trans_date_trans_time, utc=True).min()
    if latest >= earliest:
        raise ValueError('Earlier history must end strictly before the first transaction to review. '
                         'Do not upload the same or overlapping transactions in both inputs.')
