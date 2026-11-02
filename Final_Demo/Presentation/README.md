# Final presentation and rehearsal

FraudShield_Final_Presentation.pptx contains 15 editable slides: 13 main slides and
two appendix slides. Speaker notes include suggested pacing, explanations and
repository source references. Aim for about eight minutes of speaking and two
minutes of live demonstration, then questions. Rehearse with the whole team; each
member must confirm their own contributions and learning before report submission.

## Before presenting

1. Stop the TTS/STT service on port 5001. Start ../START_DEMO_REMOTE.bat and enter
   the session password. Enable the existing 5100 -> 5001 forwarding and verify
   http://b2518067.eero.online:5100 from the presentation device.
2. In the same browser tab, load corrected server history 2021-2025. Wait for the
   file list, then start corrected 2026 review. Switch to the PowerPoint while
   scoring continues; leave the tab open without refreshing.
3. Slide 11 is the live-demo cue. Return to that browser tab. Show the loaded
   history, review metrics, analyst queue and download. Explain that the model
   is already trained and scores are not calibrated probabilities.
4. If processing is not finished, use slide 9 or the executed evaluation notebook
   as explicitly recorded evidence; do not describe it as a new live result.
5. Disable forwarding and stop the server after presenting.

## Content checks

Use only corrected results: 956,022 test rows and 2,032 fraud cases. History model:
95.33% precision, 86.42% recall, 90.66% F1, 99.9621% accuracy, 86 false alerts and
276 misses. Old 960,579-row results are superseded. The correction followed
earlier-test investigation; disclose that development history. The look-back
experiment is delayed recovery, not immediate model recall.

The appendix contains the folder map and launch checklist. No personal reflections
were invented for teammates. All comparison tables and diagrams remain editable.
