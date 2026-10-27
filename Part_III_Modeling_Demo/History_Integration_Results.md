# Phase III integration of Juan's improvements

**Historical v2 record:** PR #2 was incorporated on October 27 for history v3
(17 history features). The v3 model has not been retrained; numbers and hashes
below describe v2 only. See `Training_and_Model_Sharing.md` for the new run.

October 22, 2026. Based on Juan Franceschi's PR #1 commit
7a730784334cb12e981100f4376dd40ccc9c05d0. Local integration branch:
`phase-iii-history-context`. No merge to main or remote publication is implied.

## Incorporated

- Gradient Boosting as an independently selectable transaction-only candidate.
- Juan's eight per-card history feature concepts and original six added tests.
- A separately trained Gradient Boosting history model with explicit earlier
  context plus review inputs, matching validation and deployment.
- Strict timestamp-group history: simultaneous transactions never influence one
  another. If a prior timestamp has several merchant locations, use its centroid.
- New tests for simultaneous rows, score parity, new cards, label exclusion,
  period overlap, browser-session isolation, repeat scoring, invalid context
  replacement, explicit cold start and context expiry.
- Independent evaluation selectors and validation-only labels; no new test claims.

## Corrected validation run

Full original development file: 1,296,675 rows. Earlier training period:
1,037,340. Later validation: 259,335, including 1,538 frauds. Seed 42;
training-only preprocessing; no refit on validation; no oversampling.

Corrected GB + history: precision 90.8524%, recall 85.2406%, F1 0.879571,
average precision 0.940012; TP 1,311, FP 132, FN 227, TN 257,665.
Threshold: 0.9010280974092135. History schema: strict-timestamp-groups-v2.
The corrected results replace the earlier PR's approximate 90.4% / 86.4%
figures for this implementation. Differences do not isolate the causal impact
of one bug fix on accuracy.

The transaction-only GB experiment retained from the independent investigation
has precision 87.7181%, recall 81.7295%, F1 0.846180, average precision 0.905165.
Its unchanged transaction-only features do not depend on history source.

Validation-versus-deployment parity was checked with the exact saved history
pipeline, threshold and earlier training context across all 259,335 validation
transactions. Maximum absolute score difference: **0.0**. This is functional
parity, not an independent performance evaluation.

Model SHA-256:
`7d3f19981c909b2a710681769c962b6ac698f438a72332f21d34bac443447cbb`.
Original training SHA-256:
`fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d`.

## Preserved and not claimed

The original Random Forest model and its previously recorded final-test evidence
are retained separately. Original-model hash:
`3dc619c459692de7c6e04fdf78d0f68b20b4eaa9633682d2efae1ad471610808`.
Its test precision/recall remain 68.93% / 75.10%; that evidence does not belong
to either new model. Neither new candidate was evaluated on official test labels.

Both new thresholds were selected using validation, which is already development
evidence and not a fresh holdout. No claim is made that cold starts, short
histories or real banking traffic achieve the contextual validation metrics.
All data are synthetic; subgroup fairness and independent teammate acceptance
remain separate checks. The original working demo and Phase II deliverables
are not overwritten. New demo artifacts are local and ignored by Git.
