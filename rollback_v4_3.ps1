$ErrorActionPreference = "Stop"

$backupFile = Join-Path $PSScriptRoot "V4_3_BACKUP_PATH.txt"
$hadPackageFile = Join-Path $PSScriptRoot "V4_3_BACKUP_HAD_PACKAGE.txt"
if (!(Test-Path $backupFile) -or !(Test-Path $hadPackageFile)) {
    throw "V4.3 backup metadata not found."
}
$backup = (Get-Content -LiteralPath $backupFile -Raw).Trim()
$hadPackage = ((Get-Content -LiteralPath $hadPackageFile -Raw).Trim() -eq "True")
if (!(Test-Path (Join-Path $backup "Controller.py"))) {
    throw "Controller backup missing: $backup"
}

$repoRoot = Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Assets\Repos\STU4-Solution"
$targetController = Join-Path $repoRoot "Controller.py"
$targetPackage = Join-Path $repoRoot "experimental\phobos_nav"

$confirmation = Read-Host "This restores only Controller.py and experimental\phobos_nav. Type ROLLBACK"
if ($confirmation -cne "ROLLBACK") {
    Write-Host "Rollback cancelled." -ForegroundColor Yellow
    exit 0
}

Copy-Item -LiteralPath (Join-Path $backup "Controller.py") -Destination $targetController -Force
if (Test-Path -LiteralPath $targetPackage) {
    Remove-Item -LiteralPath $targetPackage -Recurse -Force
}
if ($hadPackage) {
    New-Item -ItemType Directory -Path $targetPackage -Force | Out-Null
    Copy-Item -Path (Join-Path $backup "phobos_nav\*") -Destination $targetPackage -Recurse -Force
}
Write-Host "V4.3 rollback completed. Simulator actuator and sim configuration were not touched." -ForegroundColor Green
