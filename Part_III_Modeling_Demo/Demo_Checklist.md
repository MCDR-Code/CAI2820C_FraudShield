# Team A demo checklist

1. Start the local server and open the FraudShield page. No retraining is needed.
2. In Model evaluation, explain the chronological training/validation/test split,
   two baselines, candidate comparison, and validation-selected threshold.
3. Report final-test precision/recall, false alerts and missed fraud. Explicitly
   mention that final-test recall did not reach the preliminary 80% target.
4. Return to Batch review and Load demo sample. Identify the source as a
   validation-only sample, not a fresh test or live bank transactions.
5. Review the highest-scoring transactions and switch Flagged only on/off.
6. Download results. Explain that scores are uncalibrated model outputs;
   flags prioritize human review and do not establish confirmed fraud.
7. Upload artifacts/demo_transactions.csv. Then try an invalid CSV and confirm
   the error appears with no previous/sample results displayed as its analysis.
8. Clear the batch. Explain that originals remain unchanged, identity fields
   are excluded from model inputs and downloaded results, and the prototype
   is local only, without banking actions.

## Remaining team checks

- Have a teammate run the ZIP on a second computer with tested dependencies.
- Confirm each teammate understands threshold selection versus final testing.
- Record actual user feedback and any presentation-specific timing measurements.
- Confirm final instructor rubric, report/slides requirements and team details.
- Consider future feature/fairness improvements using development data only;
  the observed official test must not become a tuning set presented as unseen.

These are remaining checks, not claims of completed user testing.
