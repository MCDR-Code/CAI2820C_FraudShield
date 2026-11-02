# FraudShield model development and demo

## Corrected final experiment

The October 31 experiment replaces the initial generated-data experiment, which
had upstream Sparkov sampling defects. See [Generator Fix](Generator_Fix.md).
All results below use the corrected data. The original Kaggle-only experiment
and the Phase II preparation checkpoint remain separate historical evidence.

Development: **6,293,357 transactions** (original January 2019-June 2020 training
plus corrected generated 2021-2025). Chronological fitting uses 5,034,685 rows,
ending November 9, 2024; validation uses 1,258,672 later rows through December 2025.
Equal timestamps stay together. The original Kaggle test is excluded.
Corrected 2026 test: **956,022 rows**, including **2,032 fraud cases**.

| Frozen model | Precision | Recall | F1 | Accuracy | False alerts | Missed fraud |
|---|---:|---:|---:|---:|---:|---:|
| Expanded Random Forest | 42.84% | 81.84% | 56.24% | 99.7293% | 2,219 | 369 |
| Expanded transaction-only Gradient Boosting | 47.54% | 83.32% | 60.54% | 99.7691% | 1,868 | 339 |
| Gradient Boosting with 17 history features | 95.33% | 86.42% | 90.66% | 99.9621% | 86 | 276 |

Gradient Boosting is scikit-learn HistGradientBoostingClassifier. Thresholds were
chosen on validation only: maximum F1 meeting recall >= 0.80 and precision >= 0.10
where feasible. No refit on validation. The history model has 95 transformed features;
transaction-only models have 78. Its saved threshold is 0.9609706329345121.
The corrected test was scored after all models were saved. However, regeneration
followed investigation of the earlier defective test; disclose that development
history rather than claiming a pristine independent study.

Read [executed final evaluation](Team_A_Final_Evaluation_2026_ran.ipynb),
[aggregate comparison](results/fixed_final_evaluation_summary.csv), and
[corrected evidence](results/corrected_2026/). Saved evidence can be read without
models or datasets. Scores are not calibrated probabilities; flags require human
review. These synthetic-data results do not establish real-bank performance.

## Start the demo

Use Python 3.12 and the recorded environment:

```powershell
python -m pip install -r requirements-tested.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Open http://127.0.0.1:8766. Install trusted local model folders under
`artifacts/expanded_fixed_run1/{rf_baseline,transaction_models,history_v3}`.
The original Kaggle Random Forest is preserved separately in `artifacts/baseline`.
Only trusted joblib files should be loaded. Datasets and model binaries are not on GitHub.

- **Batch review:** selects expanded Random Forest or expanded transaction-only
  boosting. Switching models reuses the uploaded file and caches each result.
- **Customer history:** always uses expanded boosting with history. Load server
  history 2021-2025, or add CSVs yourself. Each file has an Unload button; overlapping
  transactions remain while another loaded file contains them. Then upload the
  corrected 2026 review file, or use the server-review control.
- **Model evaluation:** displays the selected saved experiment's validation and
  recorded 2026 test. Uploaded-file metrics describe that upload separately.

Server history uses `data/synthetic_2021_fixed/transactions_2021.csv` and
`data/synthetic_2022_2026_fixed/transactions_2022.csv` through 2025. Corrected review
is `data/synthetic_2022_2026_fixed/transactions_2026.csv`; an identical copy is
`data/expanded_training_fixed/fraudTest.csv`. Never use 2026 as earlier context
for the full-year 2026 review. History must end strictly before the review starts.

History includes only strictly earlier timestamps on the same card, including
earlier review rows. Simultaneous transactions cannot affect each other. Labels
and numeric card identities are not predictors. Review uploads do not automatically
append to history. Add the earlier review file as context before a later batch.
History remains in server memory for one hour per browser session; at most four
sessions are retained. Large files consume memory and processing time. No fixed
row or byte cap is imposed. Original/generated customers are different populations.

Downloads omit names, addresses, birth dates and card IDs, and escape spreadsheet
formulas. Optional 0/1 labels produce upload metrics after scoring. Failed history
additions preserve existing history. The browser shows 100 queue rows; downloads
include all scored rows. There is no implemented case-management or monitoring system.

## Reproduction and trusted handoff

1. Download the original pair from
   [Kartik2112](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
   into `../../01_Assignments/fraud` relative to this folder.
2. Set up the isolated generator environment using `requirements-sparkov.txt`.
   Clone [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation) into
   `data/Sparkov_Data_Generation` at commit
   `b5eb45c89d36f2aa4ef16044a42945bed8b96d93`.
3. Obtain the saved 999-customer `customers.psv` from a trusted teammate. Its checksum
   is recorded in the generation evidence. The local corrected data folder retains it.
   The file is deliberately excluded from GitHub. Exact regeneration requires this
   population; generating new customers creates a different experiment.
4. In a clean workspace with no completed fixed-output folders, run:

```powershell
powershell -ExecutionPolicy Bypass -File run_fixed_pipeline.ps1 -CustomersFile "C:\trusted-handoff\customers.psv"
```

The script creates `data/synthetic_2021_fixed`, `data/synthetic_2022_2026_fixed`,
`data/expanded_training_fixed`, and `artifacts/expanded_fixed_run1`, then executes
`Team_A_Final_Evaluation_2026.ipynb`. It refuses to overwrite completed runs.
The sampler patch is recorded in generation manifests; profile weights and fraud
scheduling are retained, while random-number consumption and realized counts change.

Use `python check_model.py --artifacts artifacts/expanded_fixed_run1/history_v3`
for trusted artifact checks. `evaluate.py` writes a test record once and refuses
an existing record. Read saved results rather than repeatedly tuning against test.
`prepare_expanded_training.py` rebuilds the corrected combined data only; it does
not train. `run_notebook.py` preserves executed notebooks by refusing overwrites.

For deliberate remote presentation, `START_DEMO_REMOTE.bat` listens on port 5001
and requests an HTTP Basic password. The planned address is http://b2518067.eero.online:5100, with external TCP 5100
forwarded to internal TCP 5001 on the presentation computer. Forwarding is currently
off and must be enabled and tested by the presenter. The TTS/STT demo shares this
mapping, so stop that app before starting FraudShield; only one app can use port
5001 at a time.
The local address is not a remote demo link. Bind to localhost for ordinary use.

## Verification and submission status

```powershell
python -m unittest discover -s tests -v
```

Tests use small fixtures, not the final data. The completed-model evidence is
preserved; testing does not retrain the project models or rerun 2026 evaluation.
See [submission forms](../Docs_README.md). Team-specific reflections still require human input; remote reachability
requires a presentation-time check. Proposed monitoring,
retraining and analyst outcome collection are not implemented prototype features.
The retrospective look-back analysis measures delayed recovery, not immediate
model detection; it is exploratory and requires later validation.

Earlier notebooks and result files are historical. In particular,
`Expanded_Training_Results.md`, `Expanded_Training.md`, `Synthetic_Data_2022_2026.md`,
and `results/expanded_training_2025_validation.json` describe the defective initial
generation and must not supply final submission metrics. The obsolete bulky data,
models and browser exports were relocated outside the repository on November 1.
