# Expanded training results — October 27, 2026

The notebook completed successfully in **28 minutes 45 seconds**, from 4:09:20
to 4:38:04 p.m. America/New_York. This includes reading/validating data, constructing
features, fitting, scoring validation, saving models and running consistency/
error-analysis checks. CPU training ran on Mariela's Windows computer; no GPU.

Data: 6,325,651 development rows, comprising original training plus generated
2021–2025. The chronological split used **5,060,519 fitting rows** and **1,265,132
validation rows**, including 2,267 validation fraud cases. Equal timestamps stay
together. **The 960,579-row 2026 test was not evaluated; its labels remain unread.**

| Model | Fit time | Validation precision | Recall | F1 | False alerts | Missed fraud |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1m 04s | 16.52% | 44.51% | 24.10% | 5,098 | 1,258 |
| Random Forest | 13m 13s | 19.23% | 80.11% | 31.01% | 7,629 | 451 |
| Gradient Boosting | 2m 20s | 23.11% | 80.02% | 35.86% | 6,037 | 453 |
| Gradient Boosting + 17 history features | 2m 52s | **95.87%** | **90.12%** | **92.91%** | **88** | **224** |

Fit times total approximately 19m 30s; remaining time is preparation, prediction,
history construction and verification. All models used the SAME validation rows.
Thresholds were selected on validation using the recorded selection rule. The
high precision of history v3 is a validation result on synthetic data, not a
2026 test result or evidence of banking performance. Comparisons with historical
original-model metrics use different validation populations and are not direct
estimates of improvement. Larger training volume alone did not improve the
transaction-only validation metrics; history supplies the useful customer context.

## Saved artifacts and verification

New artifacts are under `artifacts/expanded_through_2025_run1/`:

- `transaction_models/`: selected transaction-only Gradient Boosting, 78 features.
- `history_v3/`: selected Gradient Boosting with history, 95 features.

Each directory contains selected_model.joblib, experiment.json, validation
comparison, example inputs/expected scores, and a sharing manifest. Only the
selected pipeline per experiment is saved; the candidate Random Forest and
Logistic Regression fits are represented by their comparison metrics, not
separate saved artifacts. The original saved Random Forest remains unchanged.

Saved-model copy/source/environment/example prediction checks passed for both
selected models. Full history validation/inference parity passed with maximum
score difference **0.0**. Validation error analysis completed. The executed
notebook contains outputs with no error cells. Full dataset CSVs and model
binaries stay local; only notebook and aggregate evidence are shared.

History v3 model SHA-256:
`7c9b4dda61c677933e2730f922959838705cc5180cf5240a1acd773b5ef8911e`

Transaction-only model SHA-256:
`4b31d922944c2ac99bfcae639bc4a0237551926f0d9c21747389cc97fb2842f5`

Training CSV SHA-256:
`fee5140205817b8e0980c9cd8b1bd016a9b331cee4ccbb6ebd2a7b4b14a0ace6`

The demo still points at previously installed model folders. These new models
have not silently replaced them. Keep models/thresholds frozen before the
separate 2026 test. Follow Training_and_Model_Sharing.md to transfer trusted
artifacts, substituting the new paths above.
