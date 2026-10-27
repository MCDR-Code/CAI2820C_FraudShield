# Sparkov pilot: new customers, January 2021

This pilot generates customers and transactions across all **12 profiles in
Sparkov's main configuration**, using each profile's paired fraud rules, then
combines the outputs. It is separate from the original Kaggle files and does not
train, tune or replace any model. New cardholders are used; original training
history does not provide history for these cards.

## Completed run — October 27, 2026

- January 1–31, 2021; five customers per profile; **60 customers, 4,597 rows**.
- **610 fraud transactions (13.27%)** across the complete generated file.
- All 12 customer profiles have transactions and exactly five customers each.
- No missing cells, duplicate rows/transaction IDs, invalid labels, dates,
  numeric values or coordinates. No card IDs overlap either original CSV.
- Earlier history: **3,493 rows**, January 1–23.
- Review batch: **1,104 rows**, January 24–31, including **25 fraud transactions**.
- Repeated generation with seed 42 produced an identical customer file and all
  12 raw transaction files on this Windows environment.

Saved models, unchanged thresholds, evaluated on the SAME 1,104 review rows:

| Saved model | Fraud caught | Fraud missed | False alerts | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Original Random Forest | 11 | 14 | 1 | 91.67% | 44.00% |
| Gradient Boosting without history | 9 | 16 | 4 | 69.23% | 36.00% |
| History v3 from PR #2 | Not evaluated | — | — | — | — |

The v3 artifact does not exist yet; retraining remains a separate step. These
are new synthetic-pilot results, not original validation/test results. Only 25
fraud cases are in the review window, so this is a small preliminary experiment.
Fraud labels are used for evaluation, never as scoring inputs.

**The pilot's fraud prevalence is different from the original data.** Current
upstream `datagen_transaction.py` schedules a short fraud window for nearly every
customer (`randint(0,100) < 99`). One such window occupies a substantial part of
a one-month simulation. Most generated fraud in this run falls in the earlier
history window. We kept that behavior unchanged and did not move fraud rows to
improve the review results. Using the same project does not reproduce the Kaggle
author's precise generator version/settings or dataset distribution.

## Files to use in the demo

All generated data stays locally in:

`Part_III_Modeling_Demo/data/synthetic_pilot_2021_01/`

- `pilot_transactions.csv`: complete compatible file, suitable for the ordinary
  Batch review tab with the existing Random Forest/transaction-only model.
- `pilot_history.csv`: load this in Customer history after v3 is trained.
- `pilot_review.csv`: then upload this as transactions to review. Its timestamps
  are strictly later than the history. Review transactions also provide history
  for subsequent review transactions through the existing strict-time pipeline.
- `customers.psv`: reusable generated customer records and profile assignments;
  pipe-delimited, not a transaction upload.
- `raw/`: unchanged pipe-delimited Sparkov transaction outputs, one per profile.
- `generation.json`: source revision, seed, environment, settings and source hashes.
- `pilot_report.json`: quality checks, counts, model hashes, metrics and output hashes.
- `baseline_review_results.csv` / `boosting_review_results.csv`: saved review scores.

Only code and this explanation belong in GitHub; generated customer/transaction
files and model binaries stay outside it. The pilot is well below existing demo
upload limits. This work does not remove those limits or add multi-file uploads.

## Reproduce or try a longer period

Run from `Part_III_Modeling_Demo` with the existing Python 3.12 project environment.
Keep generation packages isolated because upstream has old dependency pins:

```powershell
git clone https://github.com/namebrandon/Sparkov_Data_Generation.git data/Sparkov_Data_Generation
git -C data/Sparkov_Data_Generation checkout b5eb45c89d36f2aa4ef16044a42945bed8b96d93
python -m venv data/sparkov_env
python -m pip --python data/sparkov_env/Scripts/python.exe install -r requirements-sparkov.txt
& data/sparkov_env/Scripts/python.exe generate_sparkov_pilot.py --worker
python generate_sparkov_pilot.py
```

Skip clone/environment setup when already present. Output folders must be new;
the program refuses to overwrite a completed run. Preparation scans the original
CSV pair under `../../01_Assignments/fraud` to check for card collisions; use
`--original-dir` if yours is elsewhere. Those originals are not modified.

For a six-month experiment, use a new folder and pass the same output path to
both commands:

```powershell
& data/sparkov_env/Scripts/python.exe generate_sparkov_pilot.py --worker --start 2021-01-01 --end 2021-06-30 --customers-per-profile 5 --seed 42 --output data/synthetic_pilot_2021_h1
python generate_sparkov_pilot.py --output data/synthetic_pilot_2021_h1
```

Only the one-month experiment above has been run. The preparation stage splits
at the beginning of the final quarter of the requested calendar period. It
scores available baseline/boosting/v3 artifacts without fitting or tuning them.

## Compatibility and reproducibility details

The worker calls upstream customer and transaction logic from a clean pinned
checkout. Customer DOB sampling is explicitly adapted: dates are anchored to
simulation start, with adults aged 18–95, rather than using the computer's current
date. Customers are sampled until each profile has five; equal profile coverage
is deliberate and does not represent population proportions. Transaction/fraud
generation rules are unchanged. Python, NumPy and Faker are seeded, and profiles
run sequentially to avoid multiprocessing randomness.

Generation uses Faker 13.12.0 and NumPy 1.26.4 on Python 3.12; upstream's NumPy
1.21.6 pin cannot be used with Python 3.12. This is a documented environment
adaptation. Reproduction was checked here, not on a Mac.

Conversion combines `trans_date` and `trans_time`, adds the Kaggle-style `fraud_`
merchant prefix, retains the target, and exports the 22 named Kaggle data fields.
There is no unused CSV index column. Card IDs remain strings. Generator-only
SSN/account/profile fields are kept in local raw/customer files and excluded from
model inputs. The redundant `unix_time` is normalized from the ISO timestamp to
UTC epoch seconds for portable output; raw files retain upstream values. Our
models use `trans_date_trans_time`, not `unix_time`. Raw epoch values themselves
can differ across computer time zones.

Source: [Sparkov pinned generation code](https://github.com/namebrandon/Sparkov_Data_Generation/tree/b5eb45c89d36f2aa4ef16044a42945bed8b96d93).
