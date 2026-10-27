# Part III model development and local demo

Part II is an unchanged preparation checkpoint. This phase uses the full original
Sparkov train/test pair, with no card reduction. Juan Franceschi contributed the
history-feature and Gradient Boosting proposal in PR #1. The integration retains
those ideas with strictly earlier timestamp groups and explicit scoring context.

**October 27 update:** PR #2 is incorporated into `phase-iii-history-context`.
History v3 has 17 history features (95 total on the full dataset). It has NOT
been retrained yet. The v2 model/results remain historical evidence, not v3
results. Follow [Training and Model Sharing](Training_and_Model_Sharing.md) for
the next run and Mac handoff. `main` and Phase II remain unchanged.

[Synthetic Pilot](Synthetic_Pilot.md) documents the separate January 2021
experiment: 60 new customers across all 12 Sparkov profiles, 4,597 generated
transactions, and paired earlier-history/review CSVs. Generated data stays local;
the original data and trained models are unchanged.

[Full Synthetic Data 2022–2026](Synthetic_Data_2022_2026.md) describes the separate
five-year generation for 999 new customers, matching the original dataset's
actual distinct-card count, with one compatible labeled CSV per year.

[Expanded Training](Expanded_Training.md) describes the separate combined
training file and copied unexecuted notebook, with all three training algorithms.

## Saved models and results

| Experiment | Validation precision | Validation recall | Validation F1 | Official test |
| --- | ---: | ---: | ---: | --- |
| Original Random Forest | 73.10% | 80.04% | 0.76412 | Previously measured: 68.93% precision, 75.10% recall |
| Gradient Boosting | 87.72% | 81.73% | 0.84618 | Not evaluated |
| Corrected Gradient Boosting + 8 history features (v2 record) | 90.85% | 85.24% | 0.87957 | Not evaluated |
| Gradient Boosting + 17 history features (v3) | Awaiting training | Awaiting training | Awaiting training | Not evaluated |

These are validation comparisons, not new independent holdout results. The
official test was already observed in the original project. Do not tune against
it or present a later retrospective evaluation as unseen. The original baseline
missed its 80% final-test recall target. Scores are not calibrated probabilities;
review flags are not confirmed fraud. Synthetic results are not banking evidence.

## Obtain the original data

Download `fraudTrain.csv` and `fraudTest.csv` from
[Kartik2112's dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
Generator provenance: [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
Extract both original CSVs into the ignored repository-root `data/` directory.
No Kaggle API credentials are assumed. Do not combine train and test or use the
Part II reduced pair for these experiments.

## Reproduce locally

Use Python 3.12. From this phase folder, with sufficient RAM:

```powershell
python -m pip install -r requirements-tested.txt
python train.py --data-dir ../data --output-dir artifacts/baseline
python train.py --data-dir ../data --output-dir artifacts/boosting --models "Gradient Boosting"
python train.py --data-dir ../data --output-dir artifacts/history_v3 --history --models "Gradient Boosting"
python verify_context.py --data-dir ../data --artifacts artifacts/history_v3
python analyze_errors.py --data-dir ../data --artifacts artifacts/history_v3
python check_model.py --artifacts artifacts/history_v3
python -m unittest discover -s tests -v
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Open http://127.0.0.1:8766. Saved models are loaded, not retrained, by the demo.
Training refuses nonempty experiment directories. Use another NEW directory for
a deliberate experiment, not an overwrite of recorded artifacts. The original
baseline's published aggregate results are in `results/`; a new baseline fit
does not automatically recreate its recorded final-test artifact.

The executed notebook currently preserves the October 22 v2 run and embedded
v2 source. It is historical evidence, NOT the executable v3 training workflow.
Do not use its Run All to train v3. Use the commands above until we refresh its
source and measured outputs after the new training run. It never evaluated the
new models on official test labels.

`evaluate.py` remains a separate deliberate tool. New GB/history candidates
require `--acknowledge-observed-test` for a retrospective evaluation. That
acknowledgment does not make the old test independent. No such evaluation was
performed for this integration.

## Demo workflow

- **Batch review:** Choose the original Random Forest or transaction-only
  Gradient Boosting. Upload a compatible CSV or load that model's sample.
- **Customer history:** Load earlier history first, then a separate review CSV.
  The example buttons load a matched pseudonymous history/review pair. An
  explicit No prior history option supports a disclosed cold start. The example
  contains a few whole card histories and is not a representative performance
  sample.
- **Model evaluation:** Choose a saved experiment. Validation tables, chart,
  threshold and limitations belong to that selected model. Only the preserved
  original baseline has the previously recorded final-test results.

History must end strictly before the first review transaction. Overlapping files
are rejected rather than double-counted. Within review batches, a transaction
sees earlier review transactions on its card, never later ones or same-time
transactions. Splitting a review stream into separate batches requires including
earlier reviewed transactions in the next context file if their history should
carry forward. Scoring does NOT automatically append transactions to context.

## Features and same-time behavior

The history model adds eight features: prior count, amount/prior-mean ratio,
amount/prior-maximum ratio, capped log time since previous timestamp, prior
24-hour count, new merchant, new category, and distance from the previous
timestamp's merchant location. For several merchants at the previous timestamp,
distance uses their centroid; this deterministic convention is documented and
tested. All features exclude current-time peers. Card IDs connect histories;
their numerical values and fraud labels never become predictors.

Unseen cards are shown as No prior history. Cold-start performance is not claimed
to match the contextual validation table. Validation uses the earlier training
period as context and the later development period as the review batch. The
same `history_with_context` function and saved precomputed-feature pipeline are
used by inference. `verify_context.py` measured zero score difference on all
259,335 validation rows for v2. Run it again for v3 after retraining; no new
parity result is claimed until then.

## Upload schema and safety

Transaction-only inputs require `trans_date_trans_time`, `dob`, `amt`, `city_pop`,
`lat`, `long`, `merch_lat`, `merch_long`, `category`, `gender`, `state`. History
and history-review inputs additionally require `cc_num` and `merchant`.
Card IDs are imported as text. Timestamp normalization has second resolution.
Invalid dates, blank values, nonfinite numbers, bad coordinates, nonpositive
amounts, negative population and ages outside 0-120 are rejected.

History and review uploads have no fixed byte or transaction-count limit.
CSV parsing reads directly from the spooled upload, avoiding full-file byte and
decoded-text copies. Available memory and processing time still determine the
practical file size. The browser shows the first 100 matching results; the full
scored result is available in the download. History supports sequential additions
or selecting several CSVs together. Updated counts follow each upload. Repeated
transaction IDs count once; conflicts are rejected. Failed additions preserve
existing history; Clear history starts over. Review accepts one CSV at a time.

Context is browser-session-scoped, stored only in local server RAM, expires
after one hour, and is explicitly clearable. At most four sessions are retained;
the oldest is evicted when another is added. Invalid legacy replacement clears
the previous context; invalid browser additions preserve it. Review errors never substitute demo results. Downloads
omit names, addresses, birth dates and card IDs and escape spreadsheet formula
text. When the review CSV supplies `is_fraud`, downloads include the actual
label and evaluation outcome. Uploads are not persisted by the application.

The upload screen evaluates this file's predictions against optional `is_fraud`
labels AFTER scoring. Labels must be 0/1 on every row when supplied; missing or
invalid labels are rejected rather than reported as a partial evaluation.
Precision, recall, F1, accuracy and all four confusion counts are shown, with
undefined measures displayed as N/A. Row labels are aligned using original row
numbers after score sorting. Unlabeled files still score normally, with a clear
message that correctness cannot be measured. This is separate from the saved
validation/test records under Model evaluation, and does not train the model.

Browser result state is separate for Random Forest, Gradient Boosting and the
history view. Switching tabs/models restores each batch, metrics, error and
filter state. Clear affects only the current batch; a successful history change
invalidates only the history review batch. State lasts for this page session,
not a page refresh or server restart. Backend history expiry still applies.

No original/reduced/demo transaction files, context files, binaries, ZIPs or
credentials are committed. Local generated files stay under ignored `artifacts/`.
Use only locally generated trusted joblib files and bind the server to localhost.
This is a classroom prototype, not an authenticated production banking service.

## Evidence and code

- `results/`: original baseline evidence, new validation-only experiment records,
  corrected context parity report and comparison table.
- `Team_A_Model_Development_and_Demo_ran.ipynb`: historical v2 executed notebook;
  refresh after v3 training rather than reuse its old numbers.
- `train.py`, `history.py`, `core.py`, `verify_context.py`: development and shared
  inference logic. `shared/project_data.py` retains Part II feature functions.
- `tests/`: input/privacy checks, threshold logic, strictly earlier history,
  same-time/order handling, context parity, cold starts and session isolation.
- `History_Integration_Results.md`: provenance, changes and measured limitations.
- `analyze_errors.py`: validation error breakdown with model-version, source
  fingerprint, metadata and split checks. `tune.py` and `--gb-params` remain
  optional; new v3 training keeps the untuned settings by default.
- `model_sharing.py` / `check_model.py`: file/source checksums, environment checks
  and matching example predictions for trusted team-generated models.

The Phase II report/notebook are not changed by these Phase III improvements.
