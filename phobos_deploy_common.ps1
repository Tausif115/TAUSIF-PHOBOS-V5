# Shared helpers for deploy_v5.ps1 and rollback_v5.ps1 (dot-sourced).
# Defines functions only; performs no action when loaded.

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-NavTag([string]$Path) {
    $match = Select-String -LiteralPath $Path -Pattern "PHOBOS_NAV_V[0-9]+" | Select-Object -First 1
    if ($match) { return $match.Matches[0].Value }
    return "<no PHOBOS_NAV tag>"
}

function Get-InstallFiles([string]$ControllerPath, [string]$PackageDir) {
    # Relative path -> full path, excluding bytecode caches.
    $files = [ordered]@{ "Controller.py" = $ControllerPath }
    if (Test-Path -LiteralPath $PackageDir) {
        Get-ChildItem -LiteralPath $PackageDir -File -Filter "*.py" | Sort-Object Name | ForEach-Object {
            $files["experimental\phobos_nav\" + $_.Name] = $_.FullName
        }
    }
    return $files
}

function New-VerifiedBackup([string]$Root, [string]$Reason, [string]$ControllerPath, [string]$PackageDir) {
    # Always a NEW folder; an existing backup is never overwritten.
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $target = Join-Path $Root "$stamp`_$Reason"
    $suffix = 1
    while (Test-Path -LiteralPath $target) {
        $target = Join-Path $Root "$stamp`_$Reason`_$suffix"
        $suffix++
    }
    New-Item -ItemType Directory -Path $target | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $target "experimental\phobos_nav") -Force | Out-Null

    $files = Get-InstallFiles $ControllerPath $PackageDir
    $manifest = @()
    foreach ($relative in $files.Keys) {
        $source = $files[$relative]
        $destination = Join-Path $target $relative
        Copy-Item -LiteralPath $source -Destination $destination
        $sourceHash = Get-Sha256 $source
        if ((Get-Sha256 $destination) -ne $sourceHash) {
            throw "Backup verification failed for $relative"
        }
        $manifest += "$sourceHash  $relative"
    }
    $manifest | Set-Content -LiteralPath (Join-Path $target "BACKUP_MANIFEST.sha256") -Encoding ASCII
    @(
        "reason=$Reason",
        "created=$(Get-Date -Format o)",
        "controller_tag=$(Get-NavTag $ControllerPath)",
        "controller_sha256=$(Get-Sha256 $ControllerPath)",
        "had_package=$([bool](Test-Path -LiteralPath $PackageDir))",
        "file_count=$($files.Count)"
    ) | Set-Content -LiteralPath (Join-Path $target "BACKUP_INFO.txt") -Encoding ASCII
    Test-BackupIntegrity $target
    return $target
}

function Get-BackupManifest([string]$BackupDir) {
    $manifestPath = Join-Path $BackupDir "BACKUP_MANIFEST.sha256"
    if (!(Test-Path -LiteralPath $manifestPath)) { throw "Backup manifest missing: $manifestPath" }
    $entries = @()
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if (!$line.Trim()) { continue }
        $parts = $line -split "  ", 2
        $entries += [pscustomobject]@{ Hash = $parts[0]; Relative = $parts[1] }
    }
    if (!($entries | Where-Object { $_.Relative -eq "Controller.py" })) {
        throw "Backup has no Controller.py entry: $BackupDir"
    }
    return $entries
}

function Test-BackupIntegrity([string]$BackupDir) {
    foreach ($entry in Get-BackupManifest $BackupDir) {
        $path = Join-Path $BackupDir $entry.Relative
        if (!(Test-Path -LiteralPath $path)) { throw "Backup file missing: $path" }
        if ((Get-Sha256 $path) -ne $entry.Hash) { throw "Backup file corrupted: $path" }
    }
}

function Restore-FromBackup([string]$BackupDir, [string]$ControllerPath, [string]$PackageDir) {
    # Restores exactly the files listed in the backup manifest, then verifies.
    Test-BackupIntegrity $BackupDir
    $entries = Get-BackupManifest $BackupDir
    Copy-Item -LiteralPath (Join-Path $BackupDir "Controller.py") -Destination $ControllerPath -Force
    if (Test-Path -LiteralPath $PackageDir) { Remove-Item -LiteralPath $PackageDir -Recurse -Force }
    $packageEntries = @($entries | Where-Object { $_.Relative -ne "Controller.py" })
    if ($packageEntries.Count -gt 0) {
        New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
    }
    foreach ($entry in $packageEntries) {
        Copy-Item -LiteralPath (Join-Path $BackupDir $entry.Relative) -Destination (Join-Path $PackageDir (Split-Path $entry.Relative -Leaf)) -Force
    }
    foreach ($entry in $entries) {
        $destination = if ($entry.Relative -eq "Controller.py") { $ControllerPath } else { Join-Path $PackageDir (Split-Path $entry.Relative -Leaf) }
        if ((Get-Sha256 $destination) -ne $entry.Hash) { throw "Restore verification failed for $($entry.Relative)" }
    }
    $restored = Get-InstallFiles $ControllerPath $PackageDir
    if ($restored.Count -ne $entries.Count) {
        throw "Restored file count $($restored.Count) != backup manifest $($entries.Count)"
    }
}

function Add-History([string]$Root, [string]$Line) {
    # Append-only log; never rewritten or truncated.
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Add-Content -LiteralPath (Join-Path $Root "BACKUP_HISTORY.log") -Value "$(Get-Date -Format o) | $Line" -Encoding ASCII
}
