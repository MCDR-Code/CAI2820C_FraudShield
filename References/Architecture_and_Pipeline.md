# Architecture and experiment pipeline

## Architecture

```mermaid
flowchart LR
  U[Analyst browser] --> A[Final_Demo FastAPI app]
  A --> S[Shared validation, features and scoring]
  T[Part_III modeling and evaluation] --> S
  D[(Data: original and corrected years)] --> T
  D --> S
  T --> M[(Models: frozen pipelines and thresholds)]
  M --> S
  S --> O[Ranked queue, upload metrics and download]
  T --> R[Part_III Results: recorded evidence]
  R --> F[Deliverables and final presentation]
```

The demo loads saved models. Training is separate. Both use the same feature code.
Card identifiers connect histories; labels and numeric IDs are not predictors.
Data and Models stay local; code, notebooks and aggregate evidence are published.
Monitoring, analyst case management and retraining schedules are proposed future
operations, not implemented services in this diagram.

## Pipeline

```mermaid
flowchart TD
  K[Original Kaggle development: 2019-June 2020] --> C[Combine development]
  G[Corrected Sparkov 2021-2026 generation] --> Y[2021-2025]
  Y --> C
  G --> X[2026 reserved chronological test]
  C --> E[Earlier 5,034,685 rows: fit preprocessing and models]
  E --> V[Later 1,258,672 rows: validate models and choose thresholds]
  V --> L[Freeze chosen model and threshold]
  L --> X
  X --> Q[Recorded test results and error/subgroup analysis]
  L --> DEMO[Load frozen model in demo]
  Y --> H[Earlier customer history]
  H --> DEMO
  X --> DEMO
```

The original Kaggle test is not part of expanded development. At scoring time,
each transaction sees only strictly earlier timestamps on the same card, including
earlier review rows. A later flag cannot retroactively change the model's earlier
prediction. The exploratory look-back rule is delayed recovery, not this pipeline's
immediate detection rate. The generator correction followed inspection of the
earlier defective test, which is disclosed in the evaluation history.

Standalone SVG versions are in Diagrams for local viewing and reuse.
