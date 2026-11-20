# PHOBOS V5 - guarded rollback of Controller.py and experimental\phobos_nav
#
# Safety properties:
#   * Restores ONLY STU4-Solution\Controller.py and experimental\phobos_nav.
#   * Verifies the chosen backup's SHA256 manifest BEFORE using it.
#   * Backs up the CURRENTLY INSTALLED files into a NEW verified backup first,
#     so a rollback can never destroy the only copy of anything.
#   * Verifies every restored file against the backup manifest; if the
#     restore fails, the pre-rollback backup is restored automatically.
#   * Never deletes or overwrites backups; appends to BACKUP_HISTORY.log.
#
# Default source: the most recent "pre-deploy-v5" backup (the install that
# existed before the last V5 deployment). Use -BackupPath to choose another,
# -List to show all backups, -WhatIf to verify without changing anything.

param(
    [string]$ReposRoot = (Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Assets\Repos"),
    [string]$BackupRoot = (Join-Path $env:USERPROFILE "Documents\STPro-Phobos-Backups"),
    [string]$BackupPath = "",
    [switch]$List,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "phobos_deploy_common.ps1")

$solutionRoot = Join-Path $ReposRoot "STU4-Solution"
$targetController = Join-Path $solutionRoot "Controller.py"
$targetPackage = Join-Path $solutionRoot "experimental\phobos_nav"

if (!(Test-Path -LiteralPath $BackupRoot)) { throw "No backup root found: $BackupRoot" }
$backups = @(Get-ChildItem -LiteralPath $BackupRoot -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "BACKUP_MANIFEST.sha256") } |
    Sort-Object Name -Descending)

if ($List) {
    foreach ($item in $backups) {
        $info = Get-Content -LiteralPath (Join-Path $item.FullName "BACKUP_INFO.txt") -ErrorAction SilentlyContinue
        $tag = ($info | Where-Object { $_ -like "controller_tag=*" }) -replace "controller_tag=", ""
        Write-Host ("{0}  {1}" -f $item.Name, $tag)
    }
    exit 0
}

if ($BackupPath) {
    $source = (Resolve-Path -LiteralPath $BackupPath).Path
} else {
    $latest = $backups | Where-Object { $_.Name -like "*_pre-deploy-v5*" } | Select-Object -First 1
    if (!$latest) { throw "No pre-deploy-v5 backup found in $BackupRoot. Use -BackupPath." }
    $source = $latest.FullName
}

Test-BackupIntegrity $source
$sourceInfo = Get-Content -LiteralPath (Join-Path $source "BACKUP_INFO.txt") -ErrorAction SilentlyContinue
Write-Host "PHOBOS V5 ROLLBACK" -ForegroundColor Cyan
Write-Host "Backup to restore (manifest verified): $source"
$sourceInfo | ForEach-Object { Write-Host "  $_" }
Write-Host "Currently installed: $(Get-NavTag $targetController) $(Get-Sha256 $targetController)"

if ($WhatIf) {
    Write-Host "WhatIf: backup verified. No files changed." -ForegroundColor Cyan
    exit 0
}

$confirmation = Read-Host "Close SpaceTeams Pro and stop all simulations, then type ROLLBACK"
if ($confirmation -cne "ROLLBACK") {
    Write-Host "Rollback cancelled. No files changed." -ForegroundColor Yellow
    exit 0
}

# Preserve what is installed now before replacing it.
$safety = New-VerifiedBackup $BackupRoot "pre-rollback" $targetController $targetPackage
Add-History $BackupRoot "BACKUP pre-rollback | $safety | $(Get-NavTag $targetController) | $(Get-Sha256 $targetController)"
Write-Host "VERIFIED SAFETY BACKUP: $safety" -ForegroundColor Green

try {
    Restore-FromBackup $source $targetController $targetPackage
}
catch {
    Write-Warning "Rollback failed: $($_.Exception.Message). Restoring the pre-rollback install."
    Restore-FromBackup $safety $targetController $targetPackage
    Add-History $BackupRoot "ROLLBACK-FAILED-RESTORED | $safety"
    throw
}

Add-History $BackupRoot "ROLLED-BACK | restored $source | now $(Get-NavTag $targetController) $(Get-Sha256 $targetController) | safety backup $safety"
Write-Host "Rollback completed and verified against the backup manifest." -ForegroundColor Green
Write-Host "Installed now: $(Get-NavTag $targetController) $(Get-Sha256 $targetController)"
Write-Host "Previous install preserved at: $safety"
