#requires -Version 5.1
<#
ZM-ZEGRID-RESILIENT-RUN-RESUME M0 — Windows REAL-GPU baseline witness launcher.

Creates a FRESH output folder under
  $env:USERPROFILE\Desktop\zgprobe
and runs the Python witness with an owned-process timeout (<=300 s) and a
1 GiB new-artifact cap. It never edits the installed product and never keeps a
background service/task.

Example:
  .\run_m0_windows_witness.ps1 `
    -SrcRoot "C:\worktrees\zemosaic-resilient-run-resume\src" `
    -InstalledModule "C:\Users\JarvisJunior\AppData\Local\ZeMosaic\runtime\zemosaic_zegrid_mode.py"

The installed module is used ONLY for provenance (SHA-256 equality). The
witness imports the supplied -SrcRoot, so it cannot silently run a different
source tree.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SrcRoot,
    [Parameter(Mandatory = $true)][string]$InstalledModule,
    [string]$OutputRoot = "",
    [int]$MaxSeconds = 300,
    [long]$MaxNewBytes = 1073741824,
    [int]$PressureAfter = 2,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $env:USERPROFILE "Desktop\zgprobe\m0_witness_$stamp"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path -LiteralPath $SrcRoot)) { throw "SrcRoot not found: $SrcRoot" }
if (-not (Test-Path -LiteralPath $InstalledModule)) { throw "InstalledModule not found: $InstalledModule" }

$witness = Join-Path $PSScriptRoot "m0_windows_witness.py"
if (-not (Test-Path -LiteralPath $witness)) { throw "Harness not found: $witness" }

Write-Host "M0 witness mode=WINDOWS_REAL_GPU (synthetic pressure labelled)"
Write-Host "OutputRoot=$OutputRoot"
Write-Host "InstalledModule(provenance)=$InstalledModule"

& $PythonExe $witness `
    --src-root $SrcRoot `
    --installed-module $InstalledModule `
    --output-root $OutputRoot `
    --max-seconds $MaxSeconds `
    --max-new-bytes $MaxNewBytes `
    --pressure-after $PressureAfter `
    --python-exe $PythonExe

$code = $LASTEXITCODE
Write-Host "ExitCode=$code"
if (Test-Path -LiteralPath (Join-Path $OutputRoot "run_summary.json")) {
    Write-Host "Summary: $(Join-Path $OutputRoot 'run_summary.json')"
}
exit $code
