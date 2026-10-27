# Expanded training: separate from submitted work

Original notebooks, original train/test files and saved models are unchanged.

Training begins in **2019**, not 2021: it includes the original January 2019–June
2020 training data plus generated 2021–2025 data. The folder name describes the
generated period. The 2026 file is test only. All dataset CSVs stay local;
GitHub contains the code, notebook, instructions and aggregate results.

Ready locally:

- `data/expanded_training_2021_2026/fraudTrain.csv`: **6,325,651 transactions**:
  original training data plus complete new 2021–2025 data. Earlier combined
  versions are archived. The folder was renamed from its former 2022 start.
- `data/expanded_training_2021_2026/fraudTest.csv`: 960,579 transactions from
  2026, excluded from training and validation.
- `data/expanded_training_2021_2026/training_data_manifest.json`: source hashes,
  counts, combined checksum and explicit 2026 test split.
- `Team_A_Expanded_Data_Model_Training.ipynb`: separate unexecuted notebook,
  importing current v3 code. The original executed notebook is unchanged.
- `prepare_expanded_training.py`: reproducible streaming combination; refuses
  overwrites and verifies input hashes and chronological year boundaries.

The complete additional 2021 year is ready under
`data/synthetic_2021_full/transactions_2021.csv`: 1,000,384 transactions, 999
customers, 10,277 fraud, about 267 MB. Customer records/profile assignments match
the 2022–2026 population exactly. It is a full-year run, separate from the old
60-customer January pilot. See `results/synthetic_2021_full.json` for settings.

## Simple joining command

From this phase folder, the following joins the ORIGINAL training file, the full
2021 file, and 2022–2025. It does NOT include either test file or any 2026 rows:

```powershell
python join_csvs.py --inputs "../../01_Assignments/fraud/fraudTrain.csv" "data/synthetic_2021_full/transactions_2021.csv" "data/synthetic_2022_2026/transactions_202[2-5].csv" --output "data/expanded_training_2021_2026/fraudTrain.csv" --drop-export-index --replace-existing
```

Verified combined count: **6,325,651 rows**. This larger join has been completed.
The previous output is archived;
original source files are never modified. The unused original CSV index column
is removed, headers are aligned, and values/card-number strings are preserved.
The utility writes `fraudTrain.join.json` with counts, inputs and checksum. The
copied notebook prefers that manifest when present. Existing 2026 `fraudTest.csv`
is unaffected. Plain generic usage is `python join_csvs.py --inputs a.csv b.csv
--output combined.csv`. Quoted wildcards and `--exclude` patterns are supported.
The tool only joins; it does not sort rows, deduplicate, train or choose a split.
The supplied command keeps these already sorted sources in chronological order.

For the complete project preparation (including the separate 2026 test copy),
run `python prepare_expanded_training.py --replace-existing`. It calls the same
joining tool, validates generated-file hashes and updates the project manifest.
The standalone joining tool is [join_csvs.py](join_csvs.py).

The notebook offers Logistic Regression, Random Forest and Gradient Boosting
for comparison on the same chronological validation period, plus Gradient
Boosting with 17 v3 history features as a separate experiment. Edit EXPERIMENTS
to choose algorithms. These are training options, not additional saved demo
selector entries. Each experiment saves its selected model. New artifacts go
under `artifacts/expanded_through_2025_run1/`, never over the original baseline.

Expanded data requires explicit opt-in (`expanded_data=True` or
`--expanded-data`). Original defaults retain their original size checks.
Approximately 80% earlier rows fit models; 20% later rows are validation, keeping
equal timestamps together. An absent future test is recorded with null test
counts/hashes. If supplied, a test file's chronology and ID isolation are checked
without reading fraud labels during training.

No full expanded training was performed. A tiny automated-test fit verified
the opt-in, missing-test metadata and saved-model sharing check. Run the copied
notebook locally from the Phase III folder, or use its Colab setup and point
DATA_DIR at a Drive/upload folder containing the combined file. Choose a new
RUN_NAME on subsequent runs; completed model folders are not overwritten.

The 2026 file is reserved for testing, outside fitting and validation. Since
year files have been available for exploratory scoring, this is not claimed as
a pristine untouched holdout. No 2027 generation is needed. Do not place the old
original fraudTest.csv in the new folder. Old published model metrics use a different validation set and
cannot directly measure this experiment's improvement. New artifacts are not
automatically installed into the demo. The original v3 history model also still
needs its training run.

Generated data stays outside GitHub. Code, notebook and instructions are shared;
Phase II and the original executed notebook remain reproducible checkpoints.
