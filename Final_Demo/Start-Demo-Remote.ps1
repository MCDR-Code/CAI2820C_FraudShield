# Starts the FraudShield demo for a remote presentation (same approach as Sky's Start-Phone.ps1):
#   1. the demo server on this computer (port 5001, local only) in its own window, with a fresh access key
#   2. a Cloudflare quick tunnel, then prints the link to open on the presentation computer
#      (also copied to the clipboard, and shown as a QR code)
#
#   .\Start-Demo-Remote.ps1
#   .\Start-Demo-Remote.ps1 -Cloudflared C:\Tools\cloudflared.exe
#   .\Start-Demo-Remote.ps1 -NoTunnel        server with key only (test at http://localhost:5001)
#
# Press Enter (or Ctrl+C) in this window to close the tunnel AND stop the demo server.
param(
    [string]$Cloudflared = 'G:\Tools\cloudflared.exe',
    [string]$Python = 'python',
    [int]$Port = 5001,
    [switch]$NoTunnel
)
$ErrorActionPreference = 'Stop'

# ---- access key: the tunnel address is public, so the link carries a key.
# A new key is made for every run and is never saved, so an old link stops working.
$bytes = New-Object byte[] 18
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:FRAUDSHIELD_ACCESS_KEY = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_')

# ---- demo server (bound to this computer only; the tunnel reaches it locally)
$health = "http://127.0.0.1:$Port/health"
$serverUp = { try { Invoke-RestMethod $health -TimeoutSec 3 | Out-Null; $true } catch { $false } }
if (& $serverUp) { throw "Something is already running on port $Port. Close it first (for example a previous demo window)." }
Write-Host 'Starting the FraudShield demo server in a new window...'
$server = Start-Process $Python -ArgumentList '-m', 'uvicorn', 'app:app', '--app-dir', "`"$PSScriptRoot`"", '--host', '127.0.0.1', '--port', "$Port" `
    -WorkingDirectory $PSScriptRoot -PassThru
$waited = 0
while (!(& $serverUp) -and $waited -lt 90) {
    if ($server.HasExited) { throw 'The demo server stopped. Run START_DEMO_LOCAL.bat to see the error.' }
    Start-Sleep -Seconds 2; $waited += 2
}
if (!(& $serverUp)) { throw 'The demo server did not start within 90 seconds.' }
$localLink = "http://localhost:$Port/?key=$($env:FRAUDSHIELD_ACCESS_KEY)"
Write-Host "Demo server is ready. On this computer: $localLink"

$tunnel = $null
try {
    if ($NoTunnel) { Read-Host 'Press Enter to stop the demo server'; return }

    # ---- Cloudflare quick tunnel
    if (!(Test-Path -LiteralPath $Cloudflared)) { throw "cloudflared was not found at $Cloudflared. Pass -Cloudflared <path>." }
    $log = Join-Path $env:TEMP 'fraudshield-cloudflared.log'
    Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue
    $tunnel = Start-Process $Cloudflared -ArgumentList 'tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port" `
        -RedirectStandardError $log -NoNewWindow -PassThru
    $url = $null
    for ($i = 0; $i -lt 60 -and !$url; $i++) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { throw "cloudflared stopped. Its log: $log" }
        if (Test-Path -LiteralPath $log) {
            $match = Select-String -LiteralPath $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
            if ($match) { $url = $match.Matches[0].Value }
        }
    }
    if (!$url) { throw "No tunnel address after 60 seconds. Its log: $log" }
    $link = "$url/?key=$($env:FRAUDSHIELD_ACCESS_KEY)"
    Set-Clipboard -Value $link
    Write-Host ''
    Write-Host 'Open this on the presentation computer (copied to the clipboard):' -ForegroundColor Cyan
    Write-Host "  $link" -ForegroundColor Green
    Write-Host 'The address and key change every time this script starts.'
    $qr = Join-Path $env:TEMP 'fraudshield-demo-link.png'
    Remove-Item -LiteralPath $qr -ErrorAction SilentlyContinue
    $ErrorActionPreference = 'Continue'      # the QR code is optional
    & $Python -c "import sys, qrcode; qrcode.make(sys.argv[1], box_size=12, border=3).save(sys.argv[2])" $link $qr 2>$null
    $ErrorActionPreference = 'Stop'
    if (Test-Path -LiteralPath $qr) { Start-Process $qr; Write-Host 'A QR code of the link is open: scan it with a phone camera.' }
    else { Write-Host 'No QR code (install it once with: python -m pip install "qrcode[pil]").' -ForegroundColor Yellow }
    Write-Host ''
    Read-Host 'Press Enter to close the tunnel and stop the demo'
} finally {
    if ($tunnel -and !$tunnel.HasExited) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue }
    if ($server -and !$server.HasExited) { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
    Write-Host 'Tunnel closed and demo server stopped. The link no longer works.'
}
