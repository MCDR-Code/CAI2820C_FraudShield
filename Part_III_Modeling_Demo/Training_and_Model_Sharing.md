# Train history v3 and share it with the team

PR #2 is incorporated on `phase-iii-history-context`. No full v3 model has been
trained yet. It adds all 17 timestamp-safe history features, for 95 total input
features on the full original dataset. Keep the original Random Forest and v2
files unchanged. Old 17-feature row-order results are not confirmed v3 results.

## Mariela's Windows computer

Open PowerShell in the repository's phase folder:

```powershell
cd "G:\1MDC_Education\CAI2820C_Artificial_Intelligence_App\CAI2820C_FraudShield\Part_III_Modeling_Demo"
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
python train.py --data-dir "../../01_Assignments/fraud" --output-dir artifacts/history_v3 --history --models "Gradient Boosting"
python verify_context.py --data-dir "../../01_Assignments/fraud" --artifacts artifacts/history_v3
python analyze_errors.py --data-dir "../../01_Assignments/fraud" --artifacts artifacts/history_v3
python check_model.py --artifacts artifacts/history_v3
```

Use the full original CSV pair, not the reduced sample. Training is separate from
the demo and refuses to overwrite a nonempty output directory. If a prior attempt
left partial files, preserve it and select a new output directory; do not delete
old evidence merely to rerun. The demo expects the completed `artifacts/history_v3`
folder, so resolve partial-run locations before starting it.

The first command fits the model and selects a threshold using validation. The
second verifies that inference reproduces those validation scores. The third
describes false alerts and missed fraud without using official test outcomes.
The fourth checks the generated sharing files and example predictions. No command
above evaluates the model on official test labels or changes the original CSVs.

After completion, restart the demo from this phase folder so it loads v3 source
and the new artifact. Do not reuse an old server that still has v2 cached:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Stop the previous demo first, or use another available port. The baseline and
transaction-only GB artifacts remain separate. If the v3 folder does not exist,
the new history model is unavailable; it must not silently fall back to v2.

## What to send Juan

Send the WHOLE completed `artifacts/history_v3` folder privately. It contains:

- `selected_model.joblib`: fitted feature/preprocessing/classifier pipeline and threshold.
- `experiment.json`: validation setup, parameters, environment and dataset fingerprints.
- `validation_comparison.csv` / `validation_pr.png`: measured validation evidence.
- `demo_transactions.csv` / `demo_history.csv`: a small paired example, with pseudonymous card IDs.
- `demo_expected_scores.json`: expected predictions for checking the copy on another computer.
- `model_manifest.json`: checksums for artifacts and normalized inference/training source.
- `context_verification.json`: full validation/inference parity result, after verification.

Original training/test CSVs are NOT required to load the model or run its paired
example. Juan needs them only for full-data analysis, additional scoring or his
own training. The model itself does not contain the original card histories.

Keep the trained binaries and transaction files outside GitHub, as before. Share
the code through GitHub and the artifact folder through a private transfer. A
trusted model is important: joblib can execute code when loaded; a checksum
checks copying, not who created a file.

## Juan's Mac

A Windows-trained model can often be loaded on a Mac without retraining, but this
must be checked, not assumed. Use the SAME repository code version and matching
Python major/minor and package versions. His earlier Python 3.14 environment
should not be used for a model trained with our Python 3.12 environment.

After obtaining the current branch and entering `Part_III_Modeling_Demo`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python check_model.py --artifacts artifacts/history_v3
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Place the received folder at `Part_III_Modeling_Demo/artifacts/history_v3` before
checking. If the training environment differs from the pinned requirements, use
the actual versions recorded in `experiment.json`; do not dismiss compatibility
warnings. Checksums tolerate Windows/Mac source line endings, while model file
bytes must match exactly. The smoke test compares scores with a small floating-
point tolerance and requires the same review flags. No Mac run is claimed until
Juan completes this check on his computer. No GPU is required by these models.

If he needs to use Git commands to update an existing clean checkout:

```bash
git fetch origin
git switch phase-iii-history-context
git pull --ff-only
```

Do not discard his local changes to switch branches; preserve or commit his work
first. Tell him the exact code commit used for training when sharing the folder.

## After training

Record the new metrics, model hash and parity check, then refresh the executed
notebook and results summary. Do not substitute the older 96.56%/96.68% figures.
PR #1 can be closed as superseded by Juan. Merge into `main` only after the new
run and team checks. Fixed upload-limit removal and generating extra Sparkov data
remain separate pending work; they are not included in this PR #2 integration.

Reference: https://scikit-learn.org/stable/model_persistence.html
