# Part III modeling and final evaluation

Modeling contains training, generation, joining, verification and evaluation tools,
plus the final evaluation notebook. Results contains aggregate evidence. Deliverables
contains the two current Word forms and the architecture source used in their appendix.
References contains the rubrics and Generator_Fix.md. Historical references are
explicitly separate from the corrected experiment. Shared resources are ../Shared,
../Data and ../Models. The application is ../Final_Demo.

## Read the completed experiment

Start with Modeling/Team_A_Final_Evaluation_2026_ran.ipynb and
Results/fixed_final_evaluation_summary.csv. The selected model is history boosting,
with 95.33% precision, 86.42% recall, 86 false alerts and 276 misses on 956,022 rows.
Modeling/Team_A_Final_Evaluation_2026.ipynb is the unexecuted reconstruction source.
Existing saved evaluation records are read instead of repeating a test.

On the corrected 2026 test set (956,022 transactions, 2,032 fraud), adding 17 strictly earlier per-card history features to the boosting model raised precision from 47.54% to 95.33% (+47.8 points) and recall from 83.32% to 86.42% (+3.1 points). F1 rose from 60.54% to 90.66%. False alerts fell from 1,868 to 86 (a 95.4% reduction), and missed fraud fell from 339 to 276 (63 fewer, an 18.6% reduction). Both models use the same model family and the same rows. The gain is therefore attributable to the history features, though the history model’s threshold (0.961) was tuned separately on validation. These results come from synthetic Sparkov data in which fraud is generated in a one-day window, so the history features may partly exploit that generator pattern. They should not be read as expected real-world performance. The corrected test is also not fully independent, because regeneration followed our analysis of an earlier defective test.

All commands below run from repository root. Use Python 3.12:

```powershell
python -m pip install -r requirements-tested.txt
python -m unittest discover -s Tests -v
python Part_III/Modeling/check_model.py --artifacts Models/expanded_fixed_run1/history_v3
```

The model check uses small saved examples, not another 2026 evaluation. Keep only
trusted team-generated joblib files; model deserialization can execute code.

## Reconstruct in a clean workspace

1. Obtain the original Kaggle fraudTrain.csv and fraudTest.csv from
   https://www.kaggle.com/datasets/kartik2112/fraud-detection and put them in Data/Original.
2. Clone https://github.com/namebrandon/Sparkov_Data_Generation into
   Data/Sparkov_Data_Generation and check out commit
   b5eb45c89d36f2aa4ef16044a42945bed8b96d93.
3. Create Data/sparkov_env using Python 3.12 and install requirements-sparkov.txt
   with that environment's python.exe -m pip. This is separate from the model environment.
4. Obtain the trusted saved 999-customer customers.psv file from a teammate.
   Exact regeneration needs this file; fresh customers create a different experiment.
5. With no completed fixed-output folders, run:

```powershell
powershell -ExecutionPolicy Bypass -File Part_III/Modeling/run_fixed_pipeline.ps1 -CustomersFile "C:\trusted-handoff\customers.psv"
```

The pipeline generates corrected 2021 and 2022-2026 files, checks monthly category
coverage, joins original development plus 2021-2025, reserves 2026, trains the
comparison/history models, then executes the final notebook. It refuses completed
output folders. Do not run it just to launch the demo.

Training defaults use the original data. --expanded-data explicitly enables the
expanded experiment. Thresholds are chosen on chronological validation, not test.
The original test remains historical. New models intended for a fresh independent
assessment require a separate later test after development decisions are finished.

The moved models preserve their original experiment metadata and binary hashes.
Use explicit --data-dir Data/expanded_training_fixed after relocation. Their local
model_manifest_before_reorganization.json preserves the original source record;
current manifests record relocated source paths and verified example parity.

## Demo and report limitations

Read ../Final_Demo/README.md for local/remote launchers and preloading history before
the presentation. The corrected server history is 2021-2025; review is corrected 2026.
All data are synthetic; no real-bank performance is claimed. Personal reflections
still require the team. Monitoring, outcome recording and scheduled retraining are
proposed operations. The exploratory look-back rule measures delayed recovery,
not the frozen model's immediate detection rate.
