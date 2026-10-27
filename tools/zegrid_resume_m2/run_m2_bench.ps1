param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$PythonExe = "python",
    [string]$ZgprobeRoot = "C:\Users\JarvisJunior\Desktop\zgprobe",
    [int]$MaxSeconds = 1800,
    [long]$MaxArtifactsBytes = 1073741824
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$src = (Resolve-Path $SourceRoot).Path
$bench = Join-Path $src "tools\zegrid_resume_m2\m2_bench.py"
if (-not (Test-Path $bench)) {
    Write-Error "M2 bench helper missing from SourceRoot: $bench"
    exit 2
}

$python = (Get-Command $PythonExe -ErrorAction SilentlyContinue)
if (-not $python) {
    if (-not (Test-Path $PythonExe)) {
        Write-Error "Python executable not found: $PythonExe"
        exit 2
    }
    $pythonPath = $PythonExe
} else {
    $pythonPath = $python.Source
}

# The helper enforces: absolute NEW output under ZgprobeRoot, no reuse, bounded
# timeout/artifacts, source-tree provenance (hashes before/after), no installed
# edits. It writes its own probe_summary.json and returns
# 0 PASS / 2 NOT_RUN / 3 FAIL / 4 TIMEOUT / 5 CAP_EXCEEDED / 6 GUARD.
& $pythonPath $bench `
    --source-root $src `
    --output-root $OutputRoot `
    --python-exe $pythonPath `
    --zgprobe-root $ZgprobeRoot `
    --max-seconds $MaxSeconds `
    --max-artifacts-bytes $MaxArtifactsBytes
exit $LASTEXITCODE
