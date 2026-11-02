# Run the FraudShield demo

Use Python 3.12. From project root, install `requirements-tested.txt` with
`python -m pip install -r requirements-tested.txt`. Trusted saved models must be
under Models/expanded_fixed_run1 and corrected yearly files under Data. They stay
local and are not supplied by cloning GitHub. Starting the app never trains models.

## Local use

Double-click **START_DEMO_LOCAL.bat** in this folder. Keep its terminal open.
Open http://127.0.0.1:8766. Close the terminal or press Ctrl+C to stop.
Equivalent command from project root:

```powershell
python -m uvicorn app:app --app-dir Final_Demo --host 127.0.0.1 --port 8766
```

## Remote presentation

1. Stop the TTS/STT demo; it shares internal port 5001.
2. Double-click **START_DEMO_REMOTE.bat** and enter the session password.
3. Keep the presentation computer awake and its terminal running. The launcher
   does not change Windows Firewall or router rules.
4. Enable the existing external TCP 5100 to presentation-computer TCP 5001 mapping.
5. Open http://b2518067.eero.online:5100 on the presentation device. Use any username
   and the session password. Test from another network before presentation day.
6. Disable forwarding and stop the server afterward. Forwarding is normally off.

In remote mode, host-computer access is http://127.0.0.1:5001. This temporary HTTP
demo does not provide HTTPS; use synthetic data and a separate session password.
The README does not claim external reachability has already been tested.

## Present while the data processes

Keep the same browser address and tab throughout. In Customer history, click
**Load server history 2021-2025**, or upload those five files. Wait for the five-file
list and history end date. Then upload corrected 2026 as the review file, or use
the server-review button. Only the review file is evaluated. Switch to PowerPoint
while scoring finishes; keep the tab open and do not refresh. Return when metrics
appear. History expires after one hour per browser session.

Batch review compares expanded RF and transaction-only boosting. Changing models
reuses the uploaded file. Customer history uses the history model. Model evaluation
shows recorded validation/test evidence, without training or threshold selection.
Download scored results to preserve them beyond the page session.

The presentation deck and rehearsal notes are in Presentation. Missing personal
reflections in the report still require team input.
