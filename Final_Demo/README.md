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

Same approach as the TTS/STT phone app (Start-Phone.ps1):

1. One-time setup: cloudflared at `G:\Tools\cloudflared.exe` (already used by the
   TTS/STT project) and, for the QR code, `python -m pip install "qrcode[pil]"`.
2. Double-click **START_DEMO_REMOTE.bat** (it runs `Start-Demo-Remote.ps1`). It starts
   the demo server on this computer (port 5001, local only) in its own window, then a
   Cloudflare quick tunnel.
3. It prints a link like `https://some-words.trycloudflare.com/?key=...`, copies it to
   the clipboard and opens a QR code. Open it on the presentation computer; the key is
   then remembered by that browser, so no password is typed.
4. Keep this computer awake and the launcher window open while presenting.
5. Press Enter in the launcher afterward: the tunnel closes and the demo server stops.
   The address and key change every start, so an old link never works again.

Options: `-Cloudflared <path>`, `-NoTunnel` (key-protected server only, for testing).

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
