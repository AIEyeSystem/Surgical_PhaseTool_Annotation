[CmdletBinding()]
param(
    [switch]$SkipCheckpoint,
    [switch]$SkipSmokeTest,
    [switch]$Clean,
    [switch]$SkipDependencyInstall,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$VenvRoot = Join-Path $RepoRoot ".venv"
$SourceEnvironment = "labeling"
$BuildRoot = Join-Path $RepoRoot "build"
$DestinationRoot = Join-Path $BuildRoot "IVAS"
$StageRoot = Join-Path $RepoRoot ("tmp\portable-" + [guid]::NewGuid().ToString("N"))
$ReleaseRoot = Join-Path $StageRoot "IVAS"
$WorkRoot = Join-Path $RepoRoot "tmp\pyinstaller"
$Checkpoint = Join-Path $RepoRoot "sam3.1_multiplex_bf16.pt"

# A Conda prefix invoked without `conda activate` otherwise sees the current
# Windows user's site-packages. Portable builds must be derived from .venv only.
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

function Assert-ChildPath([string]$Path) {
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $Resolved.StartsWith($RepoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the repository: $Resolved"
    }
}

function Invoke-Checked([string]$Description, [scriptblock]$Command) {
    Write-Host "`n==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-FrozenSmoke([string]$Description, [string]$Arguments, [int]$TimeoutMs) {
    Write-Host "`n==> $Description" -ForegroundColor Cyan
    $Timer = [System.Diagnostics.Stopwatch]::StartNew()
    $SmokeProcess = Start-Process -FilePath $ExePath -ArgumentList $Arguments -PassThru -WindowStyle Hidden
    if (-not $SmokeProcess.WaitForExit($TimeoutMs)) {
        $SmokeProcess.Kill()
        throw "$Description timed out after $([math]::Round($TimeoutMs / 1000)) seconds."
    }
    $Timer.Stop()
    if ($SmokeProcess.ExitCode -ne 0) {
        throw "$Description failed with exit code $($SmokeProcess.ExitCode)."
    }
    Write-Host "$Description completed in $([math]::Round($Timer.Elapsed.TotalSeconds, 2)) s"
}

Write-Host "IVAS portable Windows build" -ForegroundColor White
Write-Host "Repository: $RepoRoot"

New-Item -ItemType Directory -Path (Join-Path $RepoRoot "tmp") -Force | Out-Null
$BuildLog = Join-Path $RepoRoot "tmp\portable-build.log"
Start-Transcript -Path $BuildLog -Force | Out-Null
trap {
    Write-Host "`n[FAILED] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Build log: $BuildLog"
    Stop-Transcript | Out-Null
    exit 1
}

if ($PythonExe) {
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} else {
    foreach ($Candidate in @((Join-Path $VenvRoot "Scripts\python.exe"), (Join-Path $VenvRoot "python.exe"))) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExe = $Candidate
            break
        }
    }
}
if (-not $PythonExe) {
    $Conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $Conda) {
        throw "Conda is required once to clone the '$SourceEnvironment' environment into .venv."
    }
    if (Test-Path -LiteralPath $VenvRoot) {
        throw "No Python found in .venv. Use -PythonExe to select an existing environment."
    }
    Invoke-Checked "Clone Conda environment '$SourceEnvironment' into .venv" {
        conda create --prefix $VenvRoot --clone $SourceEnvironment --copy --yes
    }
    $PythonExe = Join-Path $VenvRoot "python.exe"
}
Write-Host "Python: $PythonExe"

$RequirementFiles = @("requirements.txt", "requirements-portable.txt", "requirements-build.txt") |
    ForEach-Object { Join-Path $RepoRoot $_ }
if (-not $SkipCheckpoint -and -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "SAM 3.1 checkpoint not found: $Checkpoint. Use -SkipCheckpoint for a runtime-only build."
}

if (-not $SkipDependencyInstall) {
    & $PythonExe (Join-Path $RepoRoot "tools\check_build_environment.py") @RequirementFiles
    if ($LASTEXITCODE -ne 0) {
        $InstallArgs = @("-m", "pip", "install", "--disable-pip-version-check", "--no-user")
        foreach ($RequirementFile in $RequirementFiles) {
            $InstallArgs += @("-r", $RequirementFile)
        }
        Invoke-Checked "Install missing runtime and build dependencies" {
            & $PythonExe @InstallArgs
        }
    }
}

Invoke-Checked "Verify build runtime" {
    & $PythonExe -c "import PyInstaller, PyQt5, scipy, torch, torchvision; print('Python runtime OK'); print('PyInstaller=' + PyInstaller.__version__); print('torch=' + torch.__version__); print('cuda=' + str(torch.version.cuda))"
}

Assert-ChildPath $DestinationRoot
Assert-ChildPath $StageRoot
Assert-ChildPath $ReleaseRoot
Assert-ChildPath $WorkRoot
if ($Clean -and (Test-Path -LiteralPath $WorkRoot)) {
    Remove-Item -LiteralPath $WorkRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $WorkRoot -Parent) -Force | Out-Null

if ($SkipCheckpoint) {
    $env:IVAS_BUNDLE_CHECKPOINT = ""
} else {
    $env:IVAS_BUNDLE_CHECKPOINT = $Checkpoint
}

Push-Location $RepoRoot
try {
    $PyInstallerArgs = @(
        "--noconfirm",
        "--distpath", $StageRoot,
        "--workpath", $WorkRoot,
        "build_exe.spec"
    )
    if ($Clean) {
        $PyInstallerArgs = @("--clean") + $PyInstallerArgs
    }
    Invoke-Checked "Build the self-contained portable application" {
        & $PythonExe -m PyInstaller @PyInstallerArgs
    }
} finally {
    Pop-Location
    Remove-Item Env:IVAS_BUNDLE_CHECKPOINT -ErrorAction SilentlyContinue
}

$ExePath = Join-Path $ReleaseRoot "IVAS.exe"
if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
    throw "PyInstaller completed without producing $ExePath"
}

# PyQt5 5.15 may carry an older private MSVC runtime beside Qt. If those DLLs
# load before Torch's newer runtime, c10.dll fails with WinError 1114. Keep the
# current runtime collected at _internal and remove only the conflicting copies.
$QtBin = Join-Path $ReleaseRoot "_internal\PyQt5\Qt5\bin"
foreach ($Pattern in @("msvcp140*.dll", "vcruntime140*.dll")) {
    Get-ChildItem -LiteralPath $QtBin -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

$CheckpointPath = Join-Path $ReleaseRoot "_internal\IVAS\checkpoints\sam3.1_multiplex_bf16.pt"
if (-not $SkipCheckpoint -and -not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
    throw "The portable build is missing its bundled SAM 3.1 checkpoint."
}

if (-not $SkipSmokeTest) {
    $PreviousPlatform = $env:QT_QPA_PLATFORM
    $PreviousSmokeLog = $env:IVAS_BUILD_SMOKE_LOG
    $env:QT_QPA_PLATFORM = "offscreen"
    $env:IVAS_BUILD_SMOKE_LOG = Join-Path $ReleaseRoot "SMOKE_TEST.log"
    Remove-Item -LiteralPath $env:IVAS_BUILD_SMOKE_LOG -Force -ErrorAction SilentlyContinue
    try {
        Invoke-FrozenSmoke "Run the frozen GUI startup smoke test" "--build-smoke-test" 120000
        Invoke-FrozenSmoke "Run the frozen IVAS brand smoke test" "--build-brand-smoke-test" 120000
        Invoke-FrozenSmoke "Run the frozen fused video workbench smoke test" "--build-video-workbench-smoke-test" 120000
        if (-not $SkipCheckpoint) {
            Invoke-FrozenSmoke "Run the frozen SAM 3.1 BF16 inference smoke test" "--build-sam31-smoke-test" 300000
        }
    } finally {
        if ($null -eq $PreviousPlatform) {
            Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        } else {
            $env:QT_QPA_PLATFORM = $PreviousPlatform
        }
        if ($null -eq $PreviousSmokeLog) {
            Remove-Item Env:IVAS_BUILD_SMOKE_LOG -ErrorAction SilentlyContinue
        } else {
            $env:IVAS_BUILD_SMOKE_LOG = $PreviousSmokeLog
        }
    }
}

$BuildSize = (Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File | Measure-Object Length -Sum).Sum
$BuildInfo = @(
    "IVAS portable build"
    "Built: $([DateTimeOffset]::Now.ToString('o'))"
    "Python: $(& $PythonExe -c 'import platform; print(platform.python_version())')"
    "Torch: $(& $PythonExe -c 'import torch; print(torch.__version__)')"
    "CUDA runtime: $(& $PythonExe -c 'import torch; print(torch.version.cuda)')"
    "SAM 3.1 checkpoint bundled: $(-not $SkipCheckpoint)"
    "Size bytes: $BuildSize"
) -join [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $ReleaseRoot "BUILD_INFO.txt"), $BuildInfo, [System.Text.Encoding]::UTF8)

# Only replace the release after a successful build and smoke tests. Preserve
# the old folder, including any user configuration or annotations stored there.
$PreviousRelease = $null
if (Test-Path -LiteralPath $DestinationRoot) {
    $PreviousRelease = Join-Path $BuildRoot ("IVAS.previous-" + [DateTime]::Now.ToString("yyyyMMdd-HHmmss-fff"))
    Assert-ChildPath $PreviousRelease
    Move-Item -LiteralPath $DestinationRoot -Destination $PreviousRelease
}
try {
    Move-Item -LiteralPath $ReleaseRoot -Destination $DestinationRoot
} catch {
    if ($PreviousRelease -and -not (Test-Path -LiteralPath $DestinationRoot)) {
        Move-Item -LiteralPath $PreviousRelease -Destination $DestinationRoot
    }
    throw
}
Remove-Item -LiteralPath $StageRoot
$ExePath = Join-Path $DestinationRoot "IVAS.exe"

Write-Host "`nBuild succeeded." -ForegroundColor Green
Write-Host "Portable folder: $DestinationRoot"
Write-Host "Executable:      $ExePath"
if ($PreviousRelease) {
    Write-Host "Previous build:  $PreviousRelease"
}
Write-Host "Build log:       $BuildLog"
Write-Host "Distribute the entire IVAS folder; no Python or Conda installation is required."
Stop-Transcript | Out-Null
