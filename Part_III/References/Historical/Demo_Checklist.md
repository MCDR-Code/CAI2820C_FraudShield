# Team A Phase III demo checklist

1. Start the local server on port 8766. Saved models are loaded; no training occurs.
2. In Batch review, load the Random Forest sample. Switch to Gradient Boosting
   and load its sample. Samples are validation interaction examples, not proof
   of accuracy or comparable random test batches.
3. In Customer history, choose Load example history, then Load example transactions.
   The first input provides earlier context; the second supplies transactions to
   review. Show prior counts and any new-card/no-prior-history status.
4. Explain that each row uses only strictly earlier timestamps. Purchases at the
   same time cannot become one another's history, regardless of CSV row order.
5. Show Flagged only and download results. Scores are not calibrated fraud
   probabilities; flags are for human review, not automatic blocking.
6. Clear history and attempt scoring: it must ask for context. Select No prior
   history to explicitly demonstrate a cold start, not silently substitute history.
7. Try invalid input. Previous results must clear, with no sample fallback. A
   failed history replacement also removes the previous context.
8. In Model evaluation, select each saved experiment. New GB/history results are
   validation-only. Only the original Random Forest displays its previously
   recorded final-test result (75.10% recall, below the 80% preliminary target).
9. Explain that the official test has already been observed. No new official
   test evaluation was run for these improvements. Phase II stays unchanged.

Review uploads: up to 50,000 rows / 20 MB. Earlier-history uploads: up to
2,000,000 rows / 512 MB. An original training file can be loaded as history for
strictly later transactions, not for overlapping transactions from that file.
Both history inputs require the same card IDs as text and the merchant column.
For multiple review batches, explicitly extend earlier context as needed;
the application does not append scored batches automatically.

## Remaining team checks

- Have a teammate reproduce training and run the demo on another computer.
- Record actual usability feedback, presentation timing and machine requirements.
- Confirm final instructor rubric and report/slides requirements.
- Discuss synthetic-data and subgroup limitations without claiming fairness certification.

These are remaining checks, not claims of completed teammate acceptance.
