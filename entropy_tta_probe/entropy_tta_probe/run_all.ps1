# usage (PowerShell, inside the entropy_tta_probe folder):
#   uv run powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Data "D:\DATA" -Arch RN50
# or, after activating .venv:
#   .\run_all.ps1 -Data "D:\DATA" -Arch RN50
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$Arch = "RN50",
    [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

function Run([string[]]$PyArgs) {
    Write-Host ">> python $($PyArgs -join ' ')" -ForegroundColor Cyan
    & $Py @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "failed: $($PyArgs -join ' ')" }
}

$C = @("--data_root", $Data, "--arch", $Arch, "--workers", "$Workers")
$Main = @("swapprompt")
$HasC = Test-Path (Join-Path $Data "imagenet-c")
$Extra = if ($HasC) { @("imagenet_c:all") } else { @() }

Run (@("experiments/cache_features.py") + $C + @("--datasets") + $Main + $Extra)
Run (@("experiments/e1_entropy_vs_correctness.py") + $C + @("--datasets") + $Main + $Extra)

Run (@("experiments/run_episodic.py") + $C + @("--datasets") + $Main + @("--n_samples", "1000", "--steps", "4"))
Run (@("experiments/e2_overconfidence.py") + $C)
Run (@("experiments/e3_view_selection.py") + $C)

$Stream = if ($HasC) { @("imagenet_c:sev=5") } else { @("imagenet_a", "imagenet_r", "imagenet_sketch", "imagenet_v2") }
foreach ($Order in @("iid", "dirichlet", "class_sorted")) {
    Run (@("experiments/e4_continual_collapse.py") + $C + @("--datasets") + $Stream + @("--order", $Order))
}
Run (@("experiments/e4_continual_collapse.py") + $C + @("--datasets") + $Stream + @("--order", "dirichlet", "--objective", "ent_div"))
Run (@("experiments/e4_continual_collapse.py") + $C + @("--datasets") + $Stream + @("--order", "iid", "--params", "norm", "--lr", "1e-3"))

Run (@("experiments/e5_scale_sensitivity.py") + $C + @("--datasets") + $Main)
Run (@("experiments/e6_semantic_blindness.py") + $C + @("--datasets") + $Main)
Write-Host "done -> results\" -ForegroundColor Green
