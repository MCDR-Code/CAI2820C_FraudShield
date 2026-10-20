# Part III model development and local demo

This phase uses the FULL original Sparkov CSV pair. Part II's historical
324-card preparation checkpoint remains separate and unchanged.

## Included files

- `Team_A_Model_Development_and_Demo_ran.ipynb`: the single executed modeling
  notebook; source functions are embedded for standalone Colab execution.
- `train.py`: chronological split, training-only transformations, two baselines,
  Logistic Regression/Random Forest comparison and validation threshold selection.
- `evaluate.py`: evaluate the saved model/threshold on the official test; rejects
  an accidental repeat evaluation in the same experiment directory.
- `core.py` and `shared/`: the shared input contract and feature transformations
  used during both training and application inference.
- `app.py` and `static/`: local CSV-upload review dashboard; no retraining on upload.
- `history.py`: optional per-card history features.
- `tests/`: focused regression tests for validation, identity/target exclusion,
  threshold selection, history features, and valid/invalid upload handling.
- `results/`: original recorded aggregate metrics and charts. These are evidence,
  not runnable model artifacts or transaction datasets.
- `Development_Results.md` and `Demo_Checklist.md`: interpretation and demo sequence.
- `requirements-tested.txt`: exact tested packages; `requirements.txt`: compatible
  dependency ranges for other environments.

Model binaries and the demo transaction sample are NOT committed. They are
generated locally during training, inside the ignored `artifacts/` directory.
This avoids publishing original, reduced or sample transaction CSVs.

## Obtain data and reproduce locally

Use Python 3.12. Download the original `fraudTrain.csv` and `fraudTest.csv` from
[Kartik2112's dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
and extract them to the ignored root `data/` directory. Generator provenance:
[Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation). No Kaggle
API credentials are assumed. Do not use the Part II reduced pair for this phase.

From this phase folder:

```powershell
python -m pip install -r requirements-tested.txt
python train.py --data-dir ../data --output-dir artifacts
python evaluate.py --artifacts artifacts --data-dir ../data
python -m unittest discover -s tests -v
python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765. Once generated, saved artifacts are reused by the
dashboard; no retraining or full-data upload is needed for the live demo.
Training refuses a nonempty output directory. Evaluation refuses an existing
`final_test.json`; use the recorded result instead of repeatedly reopening labels.
Full-data training needs sufficient RAM and may take several minutes.

For notebooks, open this folder as the working directory or use Google Colab.
Set `DATA_DIR` to your original CSV directory when automatic discovery cannot
find it. The saved executed notebook can be inspected without the data. A fresh
Colab runtime can upload both CSVs or dataset ZIPs; ZIPs are not repository files.

## Method and recorded results

Sort original development transactions by timestamp; fit on the earlier
1,037,340 rows and select candidates/thresholds on the later 259,335 rows.
The separate official test has 555,719 rows. No card sampling is applied.
All learned transforms fit on training only; class weights replace resampling.
The same saved estimator, without refitting on validation, is tested after locking
the validation-selected threshold. Direct identifiers and the target are not inputs.

The recorded run selected Random Forest at threshold 0.5769612893644895.
Final-test precision was **68.93%**, recall **75.10%**, with 726 false alerts and
534 missed fraud cases. The preliminary 80% recall target was NOT met. Class-
weighted scores are not calibrated fraud probabilities. See results/ and the
executed notebook for supporting evidence.

**The official test has already been observed.** A faithful rerun reproduces
an existing experiment; it is not a new independent holdout. Do not tune against
the recorded test outcomes and present a later rerun on that file as unseen.
Synthetic historical U.S. data, repeated customers and demographic/geographic
proxies limit generalization. Subgroup checks are descriptive, not fairness
certification or evidence of production readiness.

## Optional: history features and gradient boosting (experimental)

`python train.py --data-dir ../data --output-dir artifacts_new --history --models "Gradient Boosting"`
adds eight per-card history features (`history.py`; each row sees only earlier
transactions on its card) and/or a gradient-boosting candidate. Defaults are
unchanged, so the recorded run above still reproduces. On validation only,
Gradient Boosting + history reached precision 90.4%, recall 86.4%, F1 0.883
(Random Forest baseline: 73.1% / 80.0% / 0.764). It has NOT been evaluated on
the official test. A history model's dashboard uploads also need `cc_num` and
`merchant`, and history is computed from the uploaded batch only.

## Upload contract and safety

The dashboard requires trans_date_trans_time, dob, amt, city_pop, lat, long,
merch_lat, merch_long, category, gender and state. Extra labels/identifiers are
ignored as predictors. It rejects missing/blank fields, invalid dates/numbers,
nonfinite values, invalid coordinates, nonpositive amounts, negative population
and age outside 0-120 years. Maximum batch: 50,000 rows / 20 MB.

The default page starts empty. Load demo sample is an explicit action using
locally generated validation records. Failed uploads clear results and do not
substitute sample data. Uploads are processed in memory and not saved by the
application. Downloads omit names, addresses, DOB, card IDs and fraud labels.
Flags support human review; they are not confirmed fraud or banking actions.

Keep the server on localhost. Only load locally generated joblib files, never
untrusted uploaded models. Authentication, production deployment and independent
teammate usability testing are not claimed. Confirm the instructor's final
requirements before producing the final report/presentation.
