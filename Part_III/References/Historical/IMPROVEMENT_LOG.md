# Model improvement log

Goal: push validation **precision and recall both toward 95%**.
Rules for every step: measured on the chronological **validation** window only; the official test
file is never read; choices are made on an inner split of the training period where possible
(see `tune.py`); each step records what changed and why.

> **Status note.** Steps 0 to 3 below were measured with the earlier row-order history code (PR #1) and
> a batch-only demo. The integrated branch uses strict timestamp groups and separately loaded earlier
> history (`HISTORY_VERSION = 'strict-timestamp-groups-v3'`). Teammate's corrected figures for the
> 8-feature model on that contract are precision 90.85%, recall 85.24%, F1 0.880 (the 90.4% / 86.4%
> row below was the earlier implementation and is superseded for that model). The 17-feature figures
> below must be re-measured by retraining on the integrated code (Step 4); expect them to be close
> because ties are very rare, but they are not yet confirmed.

Reading the numbers: *precision* = of the transactions we flag, how many are fraud;
*recall* = of the real fraud, how much we flag; *average precision (AP)* = how well the
model ranks fraud above legitimate (threshold-free). Validation has 259,335 rows, 1,538 fraud.

## Scoreboard (validation)

| Step | Model | Precision | Recall | F1 | AP | False alerts | Missed fraud |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Random Forest (recorded baseline) | 73.1% | 80.0% | 0.764 | 0.844 | 453 | 307 |
| 0 | Gradient Boosting + 8 history features (PR #1, row-order code; superseded) | 90.4% | 86.4% | 0.883 | 0.942 | 142 | 209 |
| 2 | Gradient Boosting + 17 history features (row-order code, reproduced by teammate: 96.56% / 96.68%) | 96.6% | 96.7% | 0.966 | 0.989 | 53 | 51 |
| 3 | Same, tuned settings | 97.1% | 96.0% | 0.965 | 0.989 | 44 | 62 |

## Step 1: error analysis of the current best model

**Question:** where does Gradient Boosting + history still fail? (`analyze_errors.py`)

**Findings (209 missed fraud, 142 false alerts):**
- Fraud comes in two groups. *Large night-time* fraud (about 22:00–03:59, amounts over $250) is
  caught well (recall 83–99%). *Small daytime* fraud is not: recall is only 37–75% between
  05:00 and 21:00, and 58% for amounts of $10–50 and 36% for $100–250.
- The median missed fraud is **$22**; the median caught fraud is **$685**.
- Weak categories: travel (17% recall), food_dining (25%), health_fitness (30%), kids_pets (35%),
  personal_care (41%). Strong: grocery_pos (99%), shopping_net (97%), gas_transport (97%).
- Misses are spread over 91 different cards, so this is not one odd card.
- Missed fraud has model scores up to 0.87, just under the threshold (0.877 is the lowest caught
  score), so many misses are close calls, not total blind spots.

**Interpretation:** a small purchase looks normal on its own. The current history features describe
one transaction against the card's average, but not *what else the card did recently*. In this
synthetic data a compromised card tends to show several suspicious transactions close together
(e.g. a big night purchase, then small ones). Features that summarise the card's last hours/days
should help.

**Decision:** Step 2 adds recent-activity and card-habit features (see below).

## Step 2: recent-activity and card-habit features

**Hypothesis:** small purchases look normal alone, but a card that just had unusual activity
(a big or night-time purchase in the last hours/days) is more likely compromised, so summarising
the card's recent past should recover the missed small fraud.

**Change** (`history.py`, nine new causal features; each still sees only earlier rows of the same card):
`hist_count_1h`, `hist_count_6h` (activity bursts), `hist_log_amt_sum_24h` (recent spend),
`hist_high_amt_count_72h` (large purchases vs the card's norm in 3 days), `hist_night_count_72h`
(late-night purchases in 3 days), `hist_category_share` and `hist_hour_share` (how usual this
category/hour is for the card), `hist_amt_zscore` (spend vs the card's own spread) and
`hist_amt_to_category_mean` (amount vs the card's usual spend in this category).
Model and settings unchanged from the earlier Gradient Boosting run (200 iterations, 31 leaves,
learning rate 0.1, class-weighted); trained with
`python train.py --data-dir ../data --output-dir artifacts_gb_history2 --models "Gradient Boosting" --history`.

**Result (validation):** precision 90.4% -> **96.6%**, recall 86.4% -> **96.7%**, F1 0.883 -> **0.966**,
AP 0.942 -> 0.989. False alerts 142 -> 53; missed fraud 209 -> 51. Threshold 0.818. Fit time 39 s.
Both precision and recall are above the 95% goal on the validation window.

**Why it worked:** the missed fraud was small daytime purchases on cards that were already showing
other suspicious activity. The recent-activity counts expose that context, which a single
transaction does not contain.

**Checks that these gains are not leakage:** the tests (`tests/test_history.py`, 13 passing) prove that
changing future rows never changes a row's features, that results do not depend on input order, and
that adding earlier context behaves like one continuous stream. No label is used by any feature.

**Caveats (read before trusting the number):**
1. *Validation-informed design.* The new features were chosen after looking at validation errors in
   Step 1, so 96.6/96.7 is optimistic. The threshold was also chosen on validation.
2. *Test will be lower.* The earlier baseline fell from 80% recall (validation) to 75% (test) because
   the later period behaves differently. Expect a similar gap; the official test has not been read.
3. *Synthetic data.* Sparkov generates fraud with clear patterns (night, large, bursts). Real-bank
   performance would be lower. This is a course-project result, not production evidence.
4. *Dashboard.* History is computed from the uploaded batch only. The 24h/72h features need several
   transactions per card in the batch; demo data keeps whole card histories for this reason. A short
   upload sees little history and will score worse than this validation result.
5. *Scores remain uncalibrated*; flags mean "review", not "confirmed fraud".

**Decision:** keep. Target reached on validation, so tuning (Step 3) is optional; its value now is
robustness, not reaching 95%.

## Step 3: tune the gradient boosting settings

**Question:** do better settings lift the 17-feature model, or is it already near its ceiling?

**Method (`tune.py`):** random search of 12 settings (learning rate, iterations, leaves, minimum samples
per leaf, L2, share of features per split, class weighting). Each was fitted on the earliest 75% of the
training period and scored by average precision on the latest 25% of it. The validation window and the
test file were not used. To fit in the available memory (2 GB free) only 25% of legitimate rows were
kept in the fitting part; the hold-out was complete.

**Search result:** hold-out AP ranged 0.973 to 0.986. Eight of twelve settings were within 0.003 of each
other, i.e. the model is not sensitive to these settings. Best: learning rate 0.1, 200 iterations,
63 leaves, min 100 samples per leaf, no L2, half the features per split, balanced weights.

**Final comparison (`train.py --gb-params artifacts_tuning/tuning.json`, full training period, validation window):**

| | Precision | Recall | F1 | AP | False alerts | Missed fraud | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untuned (31 leaves, min 20, all features) | 96.6% | 96.7% | 0.9662 | 0.9886 | 53 | 51 | 0.818 |
| Tuned | 97.1% | 96.0% | 0.9653 | 0.9894 | 44 | 62 | 0.716 |

**Reading it:** the tuned model has slightly higher precision and AP, slightly lower recall and F1; it
moves along the same precision/recall trade-off instead of improving it. A difference of 9 false alerts
and 11 missed cases out of 1,538 fraud is within what a different threshold or random seed would give.
Tuning did not help meaningfully.

**Decision:** keep the simpler untuned settings as the default (same quality, fewer moving parts).
`--gb-params` stays available. The features, not the settings, produced the gain (Step 2).

**Next ideas** (unchanged caveats apply: validation is optimistic, test not yet read): average several
seeds for stability, try LightGBM/XGBoost, or evaluate once on the official test file.

## Step 4: port the nine features to strict timestamp groups

**Why:** the teammate's integration rule is that transactions with the same timestamp are simultaneous:
each may use only strictly earlier timestamps, and neither becomes history for the other. Our PR #1 code
used row order for some calculations (running totals and counts), so one of two simultaneous
transactions could become the other's history.

**How common is it?** In the 1,296,675-row development file only **40 rows (0.003%, 20 pairs)** share a
card and timestamp, and none are fraud. So correctness matters, but the effect on the metrics should be tiny.

**Change** (`history.py`, version `strict-timestamp-groups-v3`): all nine new features are computed per
(card, timestamp) group from strictly earlier groups, in the same style as the existing eight:
window counts and sums use per-group totals; habit features (category/hour share, amount z-score, amount vs
the card's category average) use prior counts, sums and sums of squares of earlier groups. v2 saved models
must be retrained; `evaluate.py` already rejects a different history version.

**Checks:**
1. `tests/test_history_strict.py` compares **all 17 features** with a literal brute-force reference ("every
   earlier-timestamp row of the same card") on random data with many ties, three seeds. They match.
   Also tested: row order never matters, changing a simultaneous twin changes nothing, no look-ahead for any
   feature, and context + review batches equal one continuous stream. 30 tests pass in total.
2. Old code versus new code on the **real development file** (no training): at a relative tolerance of 1e-4
   exactly 40 rows differ: the 20 second rows of tied pairs, and the 20 rows right after a tie (the
   "previous location" is now the centroid of all simultaneous merchants). No other row differs. Smaller
   differences (about 4,000 rows at 1e-9, 45 at 1e-6) are floating-point noise: the old code took one global
   running total across all cards; the new code takes one per card, which is more precise.

**Not done yet:** retraining on the integrated code and recording the new validation numbers (teammate's
step). Nothing here was evaluated on the official test file.
