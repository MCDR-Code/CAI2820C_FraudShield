# Part II data preparation

Authoritative report: Deliverables/Team_A_Final_Project_Proposal_Data_Preparation.docx.
Executed notebook: Notebooks/Team_A_Data_Exploration_and_Preparation_ran.ipynb.
Instructor materials are in References. code and evidence retain the reducer,
selected-card inclusion list and sampling audit. Earlier assignment duplicates
were moved to ../Archive/Part_II_Previous_Workspace.

From project root, reconstruct the historical 324-card sample:

```powershell
python Part_II/code/reduce_fraud_by_card.py reproduce --train Data/Original/fraudTrain.csv --test Data/Original/fraudTest.csv --cards Part_II/evidence/existing_sample_cards.csv --audit Part_II/evidence/sampling_audit.json --output-dir Data/reproduced
```

The destination must be new. Set the notebook's DATA_DIR to the absolute path of
Data/reproduced, or the appropriate path relative to its notebook working directory.
Its preparation source is embedded. No classifier training is required here.
This preserved checkpoint uses 453,733 development and 194,320 test rows; do not
substitute expanded-model counts from Part III.

References/Original_Preparation_Guide.md preserves the original detailed guide.
Its historical relative paths must be interpreted using the current Data folder.
Original Kaggle data: Data/Original at repository root. Continue with ../Part_III.
