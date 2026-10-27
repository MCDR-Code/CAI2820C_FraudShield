# CAI2820C FraudShield

Team A's Credit Card Fraud Detection and Risk Assessment System for CAI2820C.
The project asks: **Which credit-card transactions should a fraud analyst
review first?** It uses synthetic Sparkov transactions distributed by Kartik2112.

Team: Jorge Alvarez V, Mariela Del Rio, Javier Elizalde Ramirez and Juan Franceschi.

## Latest development update

On `phase-iii-history-context`, the [October 27 change summary](Part_III_Modeling_Demo/README.md#latest-changes)
lists the model/history integration, demo improvements, CSV joining tool, and
expanded training/test split. Expanded training completed in 28 minutes 45 seconds;
see [results](Part_III_Modeling_Demo/Expanded_Training_Results.md). Dataset CSVs
and model binaries remain local and excluded from GitHub.

## Project phases

| Folder | Purpose | Start here |
| --- | --- | --- |
| [Part II Data Preparation](Part_II_Data_Preparation/) | Reproduce the historical 324-card subset, inspect it, engineer features and prepare chronological training/validation/test inputs | [Pipeline and reproduction guide](Part_II_Data_Preparation/README.md) |
| [Part III Modeling and Demo](Part_III_Modeling_Demo/) | Full-data model comparison, validation-only threshold selection, recorded final-test evaluation and local review dashboard | [Development and demo guide](Part_III_Modeling_Demo/README.md) |

Part II is a preparation checkpoint using the historical reduced sample. Part III
uses all original transactions/cards while preserving training, validation and
test roles. Do not substitute Part III results for the Part II report's counts
or describe preparation as classifier training.

The completed Part II DOCX and its executed notebook are submitted directly to
the instructor. The report links to this repository for supporting files. The
DOCX is not duplicated here. Historical executed notebooks are preserved; the
expanded training experiment has its own separate executed notebook.

## Data sources

- [Kartik2112 Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [Sparkov synthetic generator provenance](https://github.com/namebrandon/Sparkov_Data_Generation)

Download and extract `fraudTrain.csv` and `fraudTest.csv` manually. Put them in
a local `data/` directory at this repository's root, or adjust the documented
paths. The directory is ignored by Git and is not uploaded. Original sizes are
1,296,675 development rows and 555,719 test rows, with 23 columns in each CSV.
The code does not assume Kaggle API credentials or require `kaggle.json`.
Use the published CSVs for historical Part II and the original-model checkpoint.
The separate expanded-data experiment also uses newly generated transactions;
follow [Expanded Training](Part_III_Modeling_Demo/Expanded_Training.md) for that run.

No original or reduced transaction CSVs, ZIPs, saved model binaries, caches,
credentials, archive copies or duplicate notebook versions are committed.
The small Part II selected-card CSV is an inclusion list, not transaction data.
Part III result CSVs contain aggregate metrics only, not transaction records.

## Reproduction sequence

1. Obtain the original CSV pair from the dataset link above.
2. Follow Part II's guide to recreate the exact historical subset using the
   saved card IDs and audit. Run its notebook on the reconstructed pair.
3. For development/demo work, use the full original pair and Part III's guide.
   Train/preprocess on the earlier development period, choose the candidate
   and threshold on later validation, then evaluate the saved approach.
4. Start the local dashboard from saved artifacts generated on your computer.

Part II reconstruction has been checked twice: the resulting new train/test
files were identical across runs and matched the historical dataset content.
Recorded library versions and methodological limitations are documented in
each phase. Original data remains unchanged.

## Status and limitations

Part III selected Random Forest on validation. The recorded reserved-test
result was 68.93% precision and 75.10% recall; the preliminary 80% recall target
was not met. These are synthetic historical data results, not evidence of
production banking performance. Model scores are not calibrated fraud
probabilities and flags require human review.

The official test has now been observed. Do not tune against its outcomes and
present a rerun on that same file as a new independent test. The later final
report/presentation and teammate usability checks remain separate work.
