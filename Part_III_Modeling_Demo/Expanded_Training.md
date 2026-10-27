# Expanded training: separate from submitted work

Original notebooks, original train/test files and saved models are unchanged.

Ready locally:

- `data/expanded_training_2022_2026/fraudTrain.csv`: all 4,989,171 newly generated
  transactions for 999 new cards, combined chronologically. No original data.
- `data/expanded_training_2022_2026/training_data_manifest.json`: source hashes,
  counts, combined checksum and pending future-test status.
- `Team_A_Expanded_Data_Model_Training.ipynb`: separate unexecuted notebook,
  importing current v3 code. The original executed notebook is unchanged.
- `prepare_expanded_training.py`: reproducible streaming combination; refuses
  overwrites and verifies input hashes and chronological year boundaries.

Combined file: 1,331,921,246 bytes, 10,336 fraud transactions.
SHA-256: `5a69ab2f7ae0645f393e9e76e480e5dacf1e10b6fcae002db2d7ac5d83218c3d`.

The notebook offers Logistic Regression, Random Forest and Gradient Boosting
for comparison on the same chronological validation period, plus Gradient
Boosting with 17 v3 history features as a separate experiment. Edit EXPERIMENTS
to choose algorithms. These are training options, not additional saved demo
selector entries. Each experiment saves its selected model. New artifacts go
under `artifacts/expanded_2022_2026_run1/`, never over the original baseline.

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

The 2022–2026 files have been explored/scored and are development data. A future
final test using the SAME saved customers/profiles must still be generated after
freezing models and thresholds. Do not place the old original fraudTest.csv in
the new folder. Old published model metrics use a different validation set and
cannot directly measure this experiment's improvement. New artifacts are not
automatically installed into the demo. The original v3 history model also still
needs its training run.

Generated data stays outside GitHub. Code, notebook and instructions are shared;
Phase II and the original executed notebook remain reproducible checkpoints.
