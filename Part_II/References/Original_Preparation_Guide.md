# Team A credit card fraud data preparation

This folder supports the Part II data-preparation report and executed
notebook submitted for CAI2820C. It contains only the materials needed to
understand and reproduce that preparation checkpoint. Modeling and the demo
are kept separately in the sibling Part_III_Modeling_Demo folder.

## Files

- `Team_A_Data_Exploration_and_Preparation_ran.ipynb`: the single executed
  notebook, including EDA, quality checks, feature engineering, chronological
  splitting and training-only preprocessing. Its preparation source is embedded.
- `code/reduce_fraud_by_card.py`: standalone standard-library Python program;
  reconstructs the exact historical subset or previews/accepts a new selection.
- `evidence/existing_sample_cards.csv`: one `cc_num` column containing the exact
  324 synthetic card IDs used in the report. It is NOT a transaction dataset.
- `evidence/sampling_audit.json`: source/sample counts, completeness checks and
  SHA-256 fingerprints documenting the historical subset.
- `README.md`: this pipeline guide. The completed report is submitted separately;
  its Appendices A and B explain how these repository files support the results.

Original/reduced transaction CSVs, ZIPs, duplicate notebooks, generated outputs,
reference templates, archives, credentials and later modeling work are excluded.
The small card-ID CSV is deliberately included: a seed alone cannot reconstruct
the original historical draw whose selection algorithm was different.

## Obtain the original data

- Dataset: [Credit Card Transactions Fraud Detection, Kartik2112](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
- Synthetic generator provenance: [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation).

Download and extract `fraudTrain.csv` and `fraudTest.csv` into the ignored
`data` directory at the repository root (`../data` from this phase folder).
Kaggle may require an account for
manual downloading. The code does not require `kaggle.json` or configured API
credentials. Use the published CSV files; do not run the Sparkov simulator to
generate a different dataset.

Expected originals: 1,296,675 training/development rows and 555,719 test rows,
23 columns per file. The source is synthetic, not private banking records.
Source metadata recorded for the report identifies Kaggle version 1, updated
August 5, 2020, and the CC0 dataset license; the generator code has a separate
MIT license. Retain source attribution and do not claim real-bank performance.

## Pipeline and exact historical reproduction

1. Obtain the two original CSV files. Keep their contents unchanged.
2. Reconstruct the report's reduced pair from its saved 324 card IDs. The program
   verifies each original file's fingerprint and the audited row/card counts.
3. Keep ALL transactions for those IDs in BOTH files, preserving original row
   order and train/test separation. No new sampling occurs in reproduce mode.
4. Open the executed notebook in Colab, Jupyter or VS Code. Supply that reduced
   pair and run from the first cell. Colab accepts CSV uploads or a downloaded
   dataset ZIP; dataset files are never committed to the repository.
5. Inspect schema, target, nulls, duplicates, types, amounts, categories and time;
   create derived copies with hour, weekday, month, age and geographic distance.
6. Sort development rows chronologically: earlier 80% for training preprocessing,
   later 20% for validation. The separate test file is not part of that 20%.
7. Fit scaling/encoding on training only, apply to validation/test inputs, and
   inspect the notebook's plots, tables and saved preparation evidence.

From this phase folder, with originals in `../data`:

```powershell
python code/reduce_fraud_by_card.py reproduce --train ../data/fraudTrain.csv --test ../data/fraudTest.csv --cards evidence/existing_sample_cards.csv --audit evidence/sampling_audit.json --output-dir ../data/reproduced
```

The output directory must not already exist. Outputs are `fraudTrain.csv`,
`fraudTest.csv` and `sampling_report.json`. Adjust input/output paths to your
computer. Source hash mismatches stop the operation instead of silently using
different data. No selected history is truncated to achieve a row percentage.

Expected reconstructed totals: **453,733 development rows** and **194,320 test
rows**, both covering the same **324 cards**. These represent 34.9920% and
34.9673% of original rows, respectively. The historical selection used a
near-35%-of-test-ROWS rule and recorded seed 42; replay uses the saved IDs,
not an attempted recreation of the missing original shuffle order.

Repeated reconstruction from the same inputs yields identical newly written
transaction files. An older CSV export may have different quoting/line endings
or numeric formatting; the audit's sample hashes identify those older exact
files, not a promise that a reserialized reconstruction has the same bytes.
Use transaction identity, values, row/card totals and complete histories to
compare data content. Do not edit long card IDs in spreadsheet numeric format.

Verification on the actual project sources: two independent reconstructions
produced byte-identical train/test outputs. Both matched the historical card IDs,
row order, text values and counts; numeric values matched within 1e-12 tolerance.

## Run the notebook

Open `Team_A_Data_Exploration_and_Preparation_ran.ipynb` in Google Colab and
Run All. If necessary, upload BOTH reconstructed CSV files when prompted.
For local use, set `DATA_DIR` in the setup cell to your reconstructed directory.
The original/sample provenance comparison is optional: set `ORIGINAL_DIR` to
the original CSV directory to repeat that comparison. Saved notebook outputs
remain readable without having the datasets locally.

The notebook supplies its own source fallback when `code/project_data.py` is
not present. Do not add a second copy of the notebook or external preparation
module just to run it. The Python reducer needs Python 3.9+ with no external
libraries. The notebook was tested with Python 3.12.0, pandas 3.0.3, NumPy 2.5.4,
SciPy 1.16.3, scikit-learn 1.8.0, Matplotlib 3.10.8 and joblib 1.5.3. A local
Jupyter environment also needs Jupyter/ipykernel. Versions can change notebook
display/serialization; use the recorded environment for closest replication.

Expected reduced-development results: 451,296 legitimate and 2,437 fraud rows,
zero missing cells/duplicate transactions, 362,986 training rows, 90,747
validation rows, and 73 transformed columns. The 194,320 test inputs remain
separate; this preparation notebook does not load their fraud labels or train
a classifier. Derived outputs are created locally, not committed here.

## New selections are a different workflow

Use preview/write only to explore a NEW percentage of unique TEST cards:

```powershell
python code/reduce_fraud_by_card.py preview --train ../data/fraudTrain.csv --test ../data/fraudTest.csv --percent 35 --seed 42
python code/reduce_fraud_by_card.py write --plan sampling_plans/PLAN_FOLDER/selection.json --output-dir ../data/new_subset
```

Review preview counts before executing write and replace PLAN_FOLDER with the
actual saved path. Fixed percentage/seed and identical inputs repeat a new
draw. Omitting the seed deliberately draws again. Preview saves IDs and actual
row percentages without writing reduced data; write accepts those exact IDs.
This card-percentage method does NOT regenerate the report's historical IDs.

## Safeguards and limitations

Card IDs group complete histories; they are not numerical model predictors.
No future transactions or labels should be used in any future customer-history
features. The current preparation does not compute individual spending baselines.
Selecting from test excludes 75 original training-only cards; the later final
modeling phase therefore uses full originals. Synthetic behavior, rare fraud,
demographic/geographic proxies and known customers crossing periods limit
generalization. This checkpoint is preparation evidence, not a trained model.

Submit the report and executed notebook through the instructor's assignment
upload. Supporting repository: https://github.com/MCDR-Code/CAI2820C_FraudShield
Include the submission commit/release in the report. The repository link
supplements the required report and notebook attachments.
