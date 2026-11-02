> Current layout: generation/training tools are in Part_III/Modeling; local data
> are in Data and frozen pipelines in Models, all relative to repository root.
> Use Part_III/README.md for current reconstruction commands. Paths in the
> original experiment narrative below identify the pre-reorganization run.

# Sparkov sampling bug and corrected regeneration â€” October 31, 2026

## What was wrong

Upstream Sparkov (commit `b5eb45c`, also unchanged on upstream `master`) samples each
transaction's date, time of day and category in `Profile.closest_rand_parallel(r, i, j, obj)`.
The function sorts by column `i` but always compares column **1**, the random number used
for the date. Category and AM/PM were therefore a fixed function of the date's position in
the simulation range. A second issue assigned per-category amounts after sorting by the
time-of-day column instead of the category column.

Effects in our generated data (legitimate transactions only; fraud is generated in a
one-day window, so it still covered all 14 categories):

| File | Legitimate categories |
|---|---|
| Original Kaggle 2019â€“2020 | all 14 every month (not affected) |
| 2021 full year | 2â€“4 per month, rotating through the year |
| 2022 / 2023 / 2024 / 2025 / 2026 | 4 / 6 / 5 / 5 / 4 for the whole year |

In the 2026 test, 1,928 of 2,095 fraud cases fell in categories with no legitimate purchases
at all. Models trained and evaluated on this data (`expanded_through_2025_run1`,
`expanded_through_2025_rf_baseline`) are kept as a record but must not be reported as
final results.

## Fix

`sparkov_fixes.py` replaces `Profile.sample_from` at run time (the upstream checkout stays
unmodified). It draws three independent random numbers per transaction for date, time of
day and category, and assigns amounts to the rows of the matching category. Transaction
counts, weights, fraud hours and output format are unchanged.
`generate_sparkov_pilot.generate_profile` applies it, and `generation.json` records the
patch version and hash.

Sandbox check (4 customers, one profile, 2021â€“2026): upstream produced 2â€“4 legitimate
categories per year and AM share of 0% or 100% per category; the fix produced all 14
categories every year, about 30% AM in every category, and per-category mean amounts
matching the profile specification (for example grocery_pos 200.2 vs. 200).

## Regeneration and retraining

`run_fixed_pipeline.ps1` regenerates 2021 and 2022â€“2026 with the SAME 999 customers and
seeds as before, to retain the population and nominal seeds. Changed random-number consumption also changes realized transaction counts and fraud windows; this is not a transaction-matched causal comparison. It checks that every month has legitimate
transactions in every category (`check_generated_categories.py`), rebuilds
`data/expanded_training_fixed` (original training + 2021â€“2025; 2026 reserved as test),
trains the Random Forest, Logistic Regression/Gradient Boosting and Gradient Boosting +
history models into `artifacts/expanded_fixed_run1/`, and executes
`Team_A_Final_Evaluation_2026.ipynb`. Nothing existing is overwritten. Log:
`results/fixed_pipeline_log.txt`.

## Final evidence and limitations

Corrected development: 6,293,357 rows; fitting: 5,034,685; validation: 1,258,672.
Corrected test: 956,022 rows, 2,032 fraud. History model: 95.33% precision,
86.42% recall, 90.66% F1, 86 false alerts and 276 missed fraud.
All models and thresholds were saved before corrected-test scoring, but the
regeneration followed analysis of the earlier defective test. The whole process
is therefore disclosed as iterative development, not a pristine independent study.
Fraud covering all categories did not establish immunity to the sampling defects;
the correction applies to legitimate and fraudulent sampling.

The old experiment is historical evidence only and has been relocated outside
the repository. Exact reproduction needs the trusted saved customer file; supply
it with `run_fixed_pipeline.ps1 -CustomersFile <path>` in a clean workspace.
