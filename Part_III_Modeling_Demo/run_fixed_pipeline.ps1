# Regenerate the synthetic 2021-2026 data with the corrected Sparkov sampler, rebuild the
# expanded training/test files, retrain all models and run the final evaluation notebook.
# Run from Part_III_Modeling_Demo:  powershell -ExecutionPolicy Bypass -File run_fixed_pipeline.ps1
# Nothing existing is overwritten: every output goes to a NEW *_fixed folder.
param([string]$CustomersFile = 'data/synthetic_2022_2026_fixed/customers.psv')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force results | Out-Null
$Sparkov = 'data/sparkov_env/Scripts/python.exe'
$Customers = (Resolve-Path -LiteralPath $CustomersFile).Path # trusted saved 999-customer population

$Data = 'data/expanded_training_fixed'
$Art = 'artifacts/expanded_fixed_run1'
foreach ($Existing in @('data/synthetic_2021_fixed','data/synthetic_2022_2026_fixed',$Data,$Art)) {
    if (Test-Path -LiteralPath $Existing) { throw "Output exists: $Existing. Use a clean workspace; completed runs are preserved." }
}

function Step($Name, [scriptblock]$Command) {
    Write-Host "`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Name ====="
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "STEP FAILED: $Name (exit $LASTEXITCODE)" }
    Write-Host "----- done: $Name ($(Get-Date -Format 'HH:mm:ss'))"
}

Start-Transcript -Path results/fixed_pipeline_log.txt -Append
try {
    Step 'Generate 2021 (fixed sampler)' { & $Sparkov generate_sparkov.py --worker --customer-file $Customers --workers 4 --seed 20211027 --start 2021-01-01 --end 2021-12-31 --output data/synthetic_2021_fixed }
    Step 'Prepare 2021' { python generate_sparkov.py --output data/synthetic_2021_fixed }
    Step 'Generate 2022-2026 (fixed sampler)' { & $Sparkov generate_sparkov.py --worker --customer-file $Customers --workers 4 --seed 20221027 --start 2022-01-01 --end 2026-12-31 --output data/synthetic_2022_2026_fixed }
    Step 'Prepare 2022-2026' { python generate_sparkov.py --output data/synthetic_2022_2026_fixed }
    Step 'Check category coverage' { python check_generated_categories.py data/synthetic_2021_fixed data/synthetic_2022_2026_fixed }
    New-Item -ItemType Directory -Force $Data | Out-Null
    Step 'Join training file (original + 2021-2025)' { python join_csvs.py --inputs '../../01_Assignments/fraud/fraudTrain.csv' 'data/synthetic_2021_fixed/transactions_2021.csv' 'data/synthetic_2022_2026_fixed/transactions_202[2-5].csv' --output "$Data/fraudTrain.csv" --drop-export-index }
    Step 'Copy reserved 2026 test file' { Copy-Item 'data/synthetic_2022_2026_fixed/transactions_2026.csv' "$Data/fraudTest.csv" }
    Step 'Train Random Forest baseline' { python train.py --data-dir $Data --output-dir "$Art/rf_baseline" --models 'Random Forest' --expanded-data }
    Step 'Train Logistic Regression + Gradient Boosting' { python train.py --data-dir $Data --output-dir "$Art/transaction_models" --models 'Logistic Regression,Gradient Boosting' --expanded-data }
    Step 'Train Gradient Boosting + history' { python train.py --data-dir $Data --output-dir "$Art/history_v3" --history --models 'Gradient Boosting' --expanded-data }
    Step 'Run final evaluation notebook' { python run_notebook.py Team_A_Final_Evaluation_2026.ipynb }
    Write-Host "`nPIPELINE COMPLETED $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
finally {
    Stop-Transcript
}
