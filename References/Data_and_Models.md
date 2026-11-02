# One home for shared data and models

| Local folder | Contents |
|---|---|
| Data/Original | Original unchanged Kaggle train/test pair |
| Data/synthetic_2021_fixed | Corrected 2021 transactions |
| Data/synthetic_2022_2026_fixed | Corrected yearly 2022-2026 files |
| Data/expanded_training_fixed | Original development plus corrected 2021-2025, and separate 2026 test copy |
| Data/demo_2026 | Alternative January-November history and December review split |
| Data/Sparkov_Data_Generation | Pinned upstream checkout |
| Data/sparkov_env | Isolated generator environment; use python.exe -m pip |
| Models/expanded_fixed_run1/rf_baseline | Corrected expanded Random Forest |
| Models/expanded_fixed_run1/transaction_models | Corrected transaction-only boosting |
| Models/expanded_fixed_run1/history_v3 | Selected corrected history boosting |
| Models/baseline | Preserved original Kaggle-only Random Forest |

Data and Models are ignored by Git. Exact regeneration requires the trusted saved
customer population. Saved experiment metadata retains old paths as provenance;
use explicit data paths after moving. The algorithms, binaries, thresholds and
recorded test results were preserved; only source/import/resource paths changed.
