$ErrorActionPreference = "Stop"

Write-Host "PHOBOS NAVIGATION V4.3 DEPLOYMENT" -ForegroundColor Cyan
Write-Host "This deployment changes only Controller.py and experimental\phobos_nav." -ForegroundColor Yellow
Write-Host "It does NOT change JetSelectionActuator.py, Radar.py, or any sim configuration." -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Close SpaceTeams Pro, stop simulations, then type YES to continue"
if ($confirmation -cne "YES") {
    Write-Host "Deployment cancelled. No files changed." -ForegroundColor Yellow
    exit 0
}

$repoRoot = Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Assets\Repos\STU4-Solution"
$reposRoot = Split-Path $repoRoot -Parent
$simRepoRoot = Join-Path $reposRoot "STU4-Sim"
$actuatorRelative = "Systems/JetSelectionActuator.py"
$actuatorPath = Join-Path $simRepoRoot "Systems\JetSelectionActuator.py"
$simconfPath = Join-Path $simRepoRoot "Sims\COMPETITION_SIM.simconf"
$radarPath = Join-Path $simRepoRoot "Systems\Radar.py"
$jetCsvPath = Join-Path $simRepoRoot "Data\SpaceBuggyJetConfigNoMains.csv"

$sourceController = Join-Path $PSScriptRoot "Controller.py"
$sourcePackage = Join-Path $PSScriptRoot "experimental\phobos_nav"
$targetController = Join-Path $repoRoot "Controller.py"
$targetPackage = Join-Path $repoRoot "experimental\phobos_nav"

foreach ($path in @(
    $sourceController,
    (Join-Path $sourcePackage "__init__.py"),
    $targetController,
    $actuatorPath,
    $simconfPath,
    $radarPath,
    $jetCsvPath
)) {
    if (!(Test-Path -LiteralPath $path)) {
        throw "Required file/path not found: $path"
    }
}

$simconfText = Get-Content -LiteralPath $simconfPath -Raw
if ($simconfText -notmatch 'Local/Repos/STU4-Sim/Data/SpaceBuggyJetConfigNoMains[.]csv') {
    throw "COMPETITION_SIM.simconf no longer references the expected NoMains thruster asset. Review the configuration; no files were changed."
}

# V4.1 previously replaced the simulator actuator. Do not proceed while that
# tracked simulator file is modified: restore the official repository version
# first. This script deliberately never overwrites it.
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required for the safety check. Restore/verify the official STU4-Sim actuator before deploying."
}
if (!(Test-Path (Join-Path $simRepoRoot ".git"))) {
    throw "STU4-Sim is not a Git checkout; cannot verify the official actuator. Do not deploy yet."
}
$remoteUrl = (& git -C $simRepoRoot remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or $remoteUrl -notmatch "SimDynamX[:/]STU4-Sim") {
    throw "STU4-Sim origin is not the expected SimDynamX/STU4-Sim repository."
}
$upstream = (& git -C $simRepoRoot rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null)
if ($LASTEXITCODE -ne 0 -or !$upstream) {
    throw "STU4-Sim has no upstream branch configured; cannot verify the official actuator."
}
$officialBlob = (& git -C $simRepoRoot rev-parse "$($upstream):$actuatorRelative" 2>$null)
if ($LASTEXITCODE -ne 0 -or !$officialBlob) {
    throw "Could not resolve the upstream JetSelectionActuator.py."
}
$localBlob = (& git -C $simRepoRoot hash-object -- $actuatorPath)
if ($LASTEXITCODE -ne 0 -or !$localBlob) {
    throw "Could not hash the installed JetSelectionActuator.py."
}
if ($localBlob.Trim() -ne $officialBlob.Trim()) {
    throw "Installed JetSelectionActuator.py is not identical to the upstream official file. Run restore_official_actuator.ps1 first. No V4.3 files were copied."
}

$sourceModules = Get-ChildItem -LiteralPath $sourcePackage -File -Filter "*.py"
if ($sourceModules.Count -ne 9) {
    throw "Expected 9 navigation modules; found $($sourceModules.Count)."
}

$actuatorHashBefore = (Get-FileHash -LiteralPath $actuatorPath -Algorithm SHA256).Hash
$simconfHashBefore = $null
$radarHashBefore = $null
$jetCsvHashBefore = (Get-FileHash -LiteralPath $jetCsvPath -Algorithm SHA256).Hash
if (Test-Path -LiteralPath $simconfPath) {
    $simconfHashBefore = (Get-FileHash -LiteralPath $simconfPath -Algorithm SHA256).Hash
}
if (Test-Path -LiteralPath $radarPath) {
    $radarHashBefore = (Get-FileHash -LiteralPath $radarPath -Algorithm SHA256).Hash
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path $env:USERPROFILE "Documents\STPro-Phobos-V4.3-Backup-$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null

Copy-Item -LiteralPath $targetController -Destination (Join-Path $backup "Controller.py") -Force
$hadPackage = Test-Path -LiteralPath $targetPackage
if ($hadPackage) {
    Copy-Item -LiteralPath $targetPackage -Destination (Join-Path $backup "phobos_nav") -Recurse -Force
}
Set-Content -LiteralPath (Join-Path $PSScriptRoot "V4_3_BACKUP_PATH.txt") -Value $backup
Set-Content -LiteralPath (Join-Path $PSScriptRoot "V4_3_BACKUP_HAD_PACKAGE.txt") -Value ([string]$hadPackage)

try {
    Copy-Item -LiteralPath $sourceController -Destination $targetController -Force
    if (Test-Path -LiteralPath $targetPackage) {
        Remove-Item -LiteralPath $targetPackage -Recurse -Force
    }
    New-Item -ItemType Directory -Path $targetPackage -Force | Out-Null

    foreach ($module in $sourceModules) {
        Copy-Item -LiteralPath $module.FullName -Destination (Join-Path $targetPackage $module.Name) -Force
    }

    $checks = @(
        @{ Source = $sourceController; Target = $targetController; Name = "Controller.py" }
    )
    foreach ($module in $sourceModules) {
        $checks += @{
            Source = $module.FullName
            Target = (Join-Path $targetPackage $module.Name)
            Name = ("phobos_nav\" + $module.Name)
        }
    }
    foreach ($check in $checks) {
        $sourceHash = (Get-FileHash -LiteralPath $check.Source -Algorithm SHA256).Hash
        $targetHash = (Get-FileHash -LiteralPath $check.Target -Algorithm SHA256).Hash
        if ($sourceHash -ne $targetHash) {
            throw "Hash verification failed for $($check.Name)."
        }
        Write-Host "VERIFIED: $($check.Name)" -ForegroundColor Green
    }

    $actuatorHashAfter = (Get-FileHash -LiteralPath $actuatorPath -Algorithm SHA256).Hash
    if ($actuatorHashAfter -ne $actuatorHashBefore) {
        throw "JetSelectionActuator.py changed unexpectedly."
    }
    if ($simconfHashBefore -and (Get-FileHash -LiteralPath $simconfPath -Algorithm SHA256).Hash -ne $simconfHashBefore) {
        throw "COMPETITION_SIM.simconf changed unexpectedly."
    }
    if ($radarHashBefore -and (Get-FileHash -LiteralPath $radarPath -Algorithm SHA256).Hash -ne $radarHashBefore) {
        throw "Radar.py changed unexpectedly."
    }
    if ((Get-FileHash -LiteralPath $jetCsvPath -Algorithm SHA256).Hash -ne $jetCsvHashBefore) {
        throw "Thruster configuration CSV changed unexpectedly."
    }
}
catch {
    Write-Warning "V4.3 deployment failed. Restoring the previous Controller.py and navigation package."
    Copy-Item -LiteralPath (Join-Path $backup "Controller.py") -Destination $targetController -Force
    if (Test-Path -LiteralPath $targetPackage) {
        Remove-Item -LiteralPath $targetPackage -Recurse -Force
    }
    if ($hadPackage) {
        New-Item -ItemType Directory -Path $targetPackage -Force | Out-Null
        Copy-Item -Path (Join-Path $backup "phobos_nav\*") -Destination $targetPackage -Recurse -Force
    }
    throw
}

Write-Host ""
Write-Host "V4.3 deployment completed." -ForegroundColor Green
Write-Host "Backup: $backup"
Write-Host "Verified unchanged: JetSelectionActuator.py, Radar.py, COMPETITION_SIM.simconf, and thruster CSV."
Write-Host "Run only a 20-30 second integration test first. Do not start a full competition run."
