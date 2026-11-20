# PHOBOS V5 - installed-version verifier (STRICTLY READ-ONLY)
#
# Reports exactly which controller STPro will load, compares it with this
# V5-DEVELOPMENT source tree, and checks the official simulator files.
# This script never writes, copies, deletes or modifies anything.
#
# Exit codes: 0 = installed controller matches this source tree
#             2 = installed controller differs (informational)
#             1 = a required file or path is missing

param(
    [string]$ReposRoot = (Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Assets\Repos"),
    [string]$ExpectedValidationSimconfSha256 = "3CD0FBD3959C8A3AB0A7D4AA71A30D9F84D381089190A02967822468EA57B930"
)

$ErrorActionPreference = "Stop"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-NavTag([string]$Path) {
    $match = Select-String -LiteralPath $Path -Pattern "PHOBOS_NAV_V[0-9]+" | Select-Object -First 1
    if ($match) { return $match.Matches[0].Value }
    return "<no PHOBOS_NAV tag>"
}

function Get-PackageHashes([string]$PackageDir) {
    $table = @{}
    if (Test-Path -LiteralPath $PackageDir) {
        Get-ChildItem -LiteralPath $PackageDir -File -Filter "*.py" | ForEach-Object {
            $table[$_.Name] = Get-Sha256 $_.FullName
        }
    }
    return $table
}

$solutionRoot = Join-Path $ReposRoot "STU4-Solution"
$simRoot = Join-Path $ReposRoot "STU4-Sim"
$installedController = Join-Path $solutionRoot "Controller.py"
$installedPackage = Join-Path $solutionRoot "experimental\phobos_nav"
$sourceController = Join-Path $PSScriptRoot "Controller.py"
$sourcePackage = Join-Path $PSScriptRoot "experimental\phobos_nav"

Write-Host "PHOBOS INSTALLED-VERSION VERIFIER (read-only)" -ForegroundColor Cyan
Write-Host "Repos root: $ReposRoot"

$missing = @()
foreach ($path in @($installedController, $sourceController, $sourcePackage)) {
    if (!(Test-Path -LiteralPath $path)) { $missing += $path }
}
if ($missing.Count -gt 0) {
    $missing | ForEach-Object { Write-Host "MISSING: $_" -ForegroundColor Red }
    exit 1
}

$installedSha = Get-Sha256 $installedController
$sourceSha = Get-Sha256 $sourceController
$installedItem = Get-Item -LiteralPath $installedController
Write-Host ""
Write-Host "Installed controller : $installedController"
Write-Host "  SHA256             : $installedSha"
Write-Host "  Tag                : $(Get-NavTag $installedController)"
Write-Host "  Size / modified    : $($installedItem.Length) bytes / $($installedItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
Write-Host "Source controller    : $sourceController"
Write-Host "  SHA256             : $sourceSha"
Write-Host "  Tag                : $(Get-NavTag $sourceController)"

$allMatch = ($installedSha -eq $sourceSha)
$installedModules = Get-PackageHashes $installedPackage
$sourceModules = Get-PackageHashes $sourcePackage
$names = @($installedModules.Keys) + @($sourceModules.Keys) | Sort-Object -Unique

Write-Host ""
Write-Host "Navigation package (experimental\phobos_nav):"
foreach ($name in $names) {
    $installed = $installedModules[$name]
    $source = $sourceModules[$name]
    if (!$installed) { $state = "MISSING_IN_INSTALL"; $allMatch = $false }
    elseif (!$source) { $state = "EXTRA_IN_INSTALL"; $allMatch = $false }
    elseif ($installed -eq $source) { $state = "MATCH" }
    else { $state = "DIFFERENT"; $allMatch = $false }
    $color = if ($state -eq "MATCH") { "Green" } else { "Yellow" }
    $shown = if ($installed) { $installed.Substring(0, 16) } else { "-" }
    Write-Host ("  {0,-24} {1,-20} installed={2}" -f $name, $state, $shown) -ForegroundColor $color
}

$legacyShadow = Join-Path $solutionRoot "phobos_nav"
if (Test-Path -LiteralPath $legacyShadow) {
    Write-Host ""
    Write-Host "NOTE: legacy package exists at $legacyShadow." -ForegroundColor Yellow
    Write-Host "      Controller.py puts experimental\ first on sys.path, so it is shadowed, not loaded."
}

Write-Host ""
Write-Host "Official simulator files (STU4-Sim):"
foreach ($relative in @("Systems\JetSelectionActuator.py", "Systems\Radar.py", "Systems\Scoring.py", "Systems\SpawnDebris.py", "Sims\COMPETITION_SIM.simconf", "Data\SpaceBuggyJetConfigNoMains.csv")) {
    $path = Join-Path $simRoot $relative
    if (Test-Path -LiteralPath $path) {
        Write-Host ("  {0,-40} {1}" -f $relative, (Get-Sha256 $path))
    } else {
        Write-Host ("  {0,-40} MISSING" -f $relative) -ForegroundColor Red
    }
}
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $simRoot ".git"))) {
    $tracked = & git -C $simRoot status --porcelain --untracked-files=no 2>$null
    if ($LASTEXITCODE -eq 0 -and !$tracked) {
        Write-Host "  git: no modifications to tracked STU4-Sim files" -ForegroundColor Green
    } else {
        Write-Host "  git: tracked STU4-Sim files are MODIFIED:" -ForegroundColor Red
        $tracked | ForEach-Object { Write-Host "    $_" }
    }
}

$validation = Join-Path $simRoot "Sims\COMPETITION_SIM_500_RADAR_VALIDATION.simconf"
Write-Host ""
if (Test-Path -LiteralPath $validation) {
    $validationSha = Get-Sha256 $validation
    $state = if ($validationSha -eq $ExpectedValidationSimconfSha256.ToUpperInvariant()) { "MATCHES baseline manifest" } else { "DOES NOT MATCH baseline manifest" }
    Write-Host "500-radar validation simconf: $validationSha ($state)"
} else {
    Write-Host "500-radar validation simconf: NOT INSTALLED" -ForegroundColor Yellow
}

Write-Host ""
if ($allMatch) {
    Write-Host "RESULT: installed controller and package MATCH this V5 source tree." -ForegroundColor Green
    exit 0
}
Write-Host "RESULT: installed controller DIFFERS from this V5 source tree." -ForegroundColor Yellow
exit 2
