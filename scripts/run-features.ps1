param(
    [string]$TransPath = "data/raw/HI-Medium_Trans.csv",
    [string]$AccountsPath = "data/raw/HI-Medium_accounts.csv",
    [string]$OutputDir = "aml_features",
    [double]$Sample = $null
)

$sampleArg = if ($PSBoundParameters.ContainsKey('Sample') -and $Sample -ne $null) { "--sample $Sample" } else { "" }

Write-Host "Running feature pipeline with:`n  TransPath: $TransPath`n  AccountsPath: $AccountsPath`n  OutputDir: $OutputDir" -ForegroundColor Cyan

python experiments/run_feature_pipeline.py --trans-path $TransPath --accounts-path $AccountsPath --output-dir $OutputDir $sampleArg
