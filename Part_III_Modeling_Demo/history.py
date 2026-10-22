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
                    'hist_new_category', 'hist_dist_from_prev_km']
DAY = 86400
GAP_CAP = 30 * DAY
HISTORY_VERSION = 'strict-timestamp-groups-v2'


def _haversine_km(lat1, lon1, lat2, lon2):
    a, b, c, d = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


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
    grouped = frame.groupby(['card', 'time'], sort=True)
    group = grouped.agg(n=('amount', 'size'), total=('amount', 'sum'), maximum=('amount', 'max'),
                        lat=('lat', 'mean'), lon=('lon', 'mean')).reset_index()
    gid = grouped.ngroup().to_numpy()
    cards = group.groupby('card', sort=False)
    prior_n = (cards.n.cumsum() - group.n).to_numpy(float)
    prior_sum = (cards.total.cumsum() - group.total).to_numpy(float)
    maximum = cards.maximum.cummax().groupby(group.card, sort=False).shift().to_numpy()
    previous_lat = cards.lat.shift().to_numpy()
    previous_lon = cards.lon.shift().to_numpy()
    gap = cards.time.diff().to_numpy()
    count_24h = np.zeros(len(group), dtype=float)
    for positions in cards.indices.values():
        times = group.time.to_numpy()[positions]
        prefix = np.r_[0, np.cumsum(group.n.to_numpy()[positions])]
        left = np.searchsorted(times, times - DAY, side='left')
        count_24h[positions] = prefix[:-1] - prefix[left]
    count = prior_n[gid]
    mean = np.divide(prior_sum[gid], count, out=np.ones(len(df)), where=count > 0)
    mean_ratio = np.divide(frame.amount.to_numpy(), mean, out=np.ones(len(df)), where=count > 0)
    max_ratio = np.divide(frame.amount.to_numpy(), maximum[gid], out=np.ones(len(df)),
                          where=np.isfinite(maximum[gid]))
    first_merchant = frame.groupby(['card', 'merchant']).time.transform('min')
    first_category = frame.groupby(['card', 'category']).time.transform('min')
    distance = np.nan_to_num(_haversine_km(previous_lat[gid], previous_lon[gid], frame.lat, frame.lon), nan=0.)
    values = np.column_stack([
        count, mean_ratio, max_ratio,
        np.log1p(np.where(np.isnan(gap[gid]), GAP_CAP, np.minimum(gap[gid], GAP_CAP))),
        count_24h[gid], frame.time.eq(first_merchant), frame.time.eq(first_category), distance,
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
