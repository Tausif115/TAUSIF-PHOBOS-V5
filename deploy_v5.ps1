# PHOBOS V5 - guarded deployment of Controller.py and experimental\phobos_nav
#
# Safety properties:
#   * Never modifies STU4-Sim (actuator, radar, scoring, spawner, simconfs, CSV).
#   * Verifies the official STU4-Sim systems match their git upstream first.
#   * Backs up the CURRENTLY INSTALLED controller + package into a NEW
#     timestamped folder (never overwrites an existing backup) and verifies
#     every backed-up file by SHA256 before anything is replaced.
#   * Appends to an append-only BACKUP_HISTORY.log outside the git repo.
#   * Verifies SHA256 source == installed after copying; restores from the
#     verified backup automatically on any failure.
#   * Writes nothing into this V5 source tree.
#
# Use -WhatIf to run every check and print the plan without changing files.

param(
    [string]$ReposRoot = (Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Assets\Repos"),
    [string]$BackupRoot = (Join-Path $env:USERPROFILE "Documents\STPro-Phobos-Backups"),
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "phobos_deploy_common.ps1")

# ---------------------------------------------------------------------------
Write-Host "PHOBOS NAVIGATION V5 DEPLOYMENT" -ForegroundColor Cyan
Write-Host "Changes ONLY STU4-Solution\Controller.py and STU4-Solution\experimental\phobos_nav." -ForegroundColor Yellow
Write-Host "Never changes STU4-Sim (actuator, radar, scoring, spawner, simconfs, thruster CSV)." -ForegroundColor Yellow

$solutionRoot = Join-Path $ReposRoot "STU4-Solution"
$simRoot = Join-Path $ReposRoot "STU4-Sim"
$targetController = Join-Path $solutionRoot "Controller.py"
$targetPackage = Join-Path $solutionRoot "experimental\phobos_nav"
$sourceController = Join-Path $PSScriptRoot "Controller.py"
$sourcePackage = Join-Path $PSScriptRoot "experimental\phobos_nav"
$officialFiles = @(
    "Systems/JetSelectionActuator.py", "Systems/Radar.py", "Systems/Scoring.py",
    "Systems/SpawnDebris.py", "Sims/COMPETITION_SIM.simconf", "Data/SpaceBuggyJetConfigNoMains.csv"
)

foreach ($path in @($sourceController, (Join-Path $sourcePackage "__init__.py"), $targetController, $simRoot)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Required file/path not found: $path" }
}

# 1. Official simulator integrity (git upstream comparison).
if (!(Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required to verify the official STU4-Sim files." }
if (!(Test-Path (Join-Path $simRoot ".git"))) { throw "STU4-Sim is not a Git checkout; cannot verify official files." }
$remoteUrl = (& git -C $simRoot remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or $remoteUrl -notmatch "SimDynamX[:/]STU4-Sim") { throw "STU4-Sim origin is not SimDynamX/STU4-Sim." }
$upstream = (& git -C $simRoot rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null)
if ($LASTEXITCODE -ne 0 -or !$upstream) { throw "STU4-Sim has no upstream branch configured." }
$officialHashBefore = @{}
foreach ($relative in $officialFiles) {
    $full = Join-Path $simRoot ($relative -replace "/", "\")
    if (!(Test-Path -LiteralPath $full)) { throw "Official file missing: $full" }
    $officialBlob = (& git -C $simRoot rev-parse "$($upstream):$relative" 2>$null)
    $localBlob = (& git -C $simRoot hash-object -- $full)
    if (!$officialBlob -or $localBlob.Trim() -ne $officialBlob.Trim()) {
        throw "$relative differs from upstream $upstream. Restore the official file first. Nothing was changed."
    }
    $officialHashBefore[$relative] = Get-Sha256 $full
}
Write-Host "VERIFIED: official STU4-Sim files match $upstream" -ForegroundColor Green

# 2. Source manifest.
$sourceFiles = Get-InstallFiles $sourceController $sourcePackage
$sourceHashes = @{}
foreach ($relative in $sourceFiles.Keys) { $sourceHashes[$relative] = Get-Sha256 $sourceFiles[$relative] }
Write-Host "Source: $(Get-NavTag $sourceController), $($sourceFiles.Count) files, Controller.py $($sourceHashes['Controller.py'])"
Write-Host "Installed now: $(Get-NavTag $targetController), Controller.py $(Get-Sha256 $targetController)"

$validation = Join-Path $simRoot "Sims\COMPETITION_SIM_500_RADAR_VALIDATION.simconf"
if (Test-Path -LiteralPath $validation) {
    Write-Host "500-radar validation simconf present: $(Get-Sha256 $validation)"
} else {
    Write-Host "WARNING: COMPETITION_SIM_500_RADAR_VALIDATION.simconf is not installed (this script does not install it)." -ForegroundColor Yellow
}

if ($WhatIf) {
    Write-Host "WhatIf: all checks passed. No backup made and no files changed." -ForegroundColor Cyan
    exit 0
}

$confirmation = Read-Host "Close SpaceTeams Pro and stop all simulations, then type DEPLOY V5"
if ($confirmation -cne "DEPLOY V5") {
    Write-Host "Deployment cancelled. No files changed." -ForegroundColor Yellow
    exit 0
}

# 3. Verified backup of what is installed now.
$backup = New-VerifiedBackup $BackupRoot "pre-deploy-v5" $targetController $targetPackage
Add-History $BackupRoot "BACKUP pre-deploy-v5 | $backup | $(Get-NavTag $targetController) | $(Get-Sha256 $targetController)"
Write-Host "VERIFIED BACKUP: $backup" -ForegroundColor Green

# 4. Deploy and verify; restore automatically on failure.
try {
    Copy-Item -LiteralPath $sourceController -Destination $targetController -Force
    if (Test-Path -LiteralPath $targetPackage) { Remove-Item -LiteralPath $targetPackage -Recurse -Force }
    New-Item -ItemType Directory -Path $targetPackage -Force | Out-Null
    foreach ($relative in $sourceFiles.Keys) {
        if ($relative -eq "Controller.py") { continue }
        Copy-Item -LiteralPath $sourceFiles[$relative] -Destination (Join-Path $targetPackage (Split-Path $relative -Leaf)) -Force
    }

    $installedFiles = Get-InstallFiles $targetController $targetPackage
    if ($installedFiles.Count -ne $sourceFiles.Count) { throw "Installed file count $($installedFiles.Count) != source $($sourceFiles.Count)" }
    foreach ($relative in $sourceFiles.Keys) {
        if (!$installedFiles.Contains($relative)) { throw "Missing after deploy: $relative" }
        if ((Get-Sha256 $installedFiles[$relative]) -ne $sourceHashes[$relative]) { throw "Hash verification failed: $relative" }
        Write-Host "VERIFIED: $relative" -ForegroundColor Green
    }
    foreach ($relative in $officialFiles) {
        if ((Get-Sha256 (Join-Path $simRoot ($relative -replace "/", "\"))) -ne $officialHashBefore[$relative]) {
            throw "Official file changed unexpectedly: $relative"
        }
    }
}
catch {
    Write-Warning "V5 deployment failed: $($_.Exception.Message). Restoring the verified backup."
    Restore-FromBackup $backup $targetController $targetPackage
    Add-History $BackupRoot "DEPLOY-FAILED-RESTORED | $backup"
    throw
}

Add-History $BackupRoot "DEPLOYED | $(Get-NavTag $targetController) | $(Get-Sha256 $targetController) | rollback source: $backup"
Write-Host ""
Write-Host "V5 deployment completed and verified." -ForegroundColor Green
Write-Host "Backup of the previous install: $backup"
Write-Host "History: $(Join-Path $BackupRoot 'BACKUP_HISTORY.log')"
Write-Host "Run verify_installed_v5.ps1, then ONLY the 20-30 s COMPETITION_SIM_500_RADAR_VALIDATION check."
