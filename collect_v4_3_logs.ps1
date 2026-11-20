$ErrorActionPreference = "Stop"

$logsRoot = Join-Path $env:LOCALAPPDATA "SimDynamX\STPro\Logs"
if (!(Test-Path -LiteralPath $logsRoot)) {
    throw "STPro Logs directory not found: $logsRoot"
}

$run = Get-ChildItem -LiteralPath $logsRoot -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (!$run) {
    throw "No simulation log folders found."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zip = Join-Path $env:USERPROFILE "Downloads\Phobos-V43-Logs-$stamp.zip"
Compress-Archive -LiteralPath $run.FullName -DestinationPath $zip -Force

Write-Host "Latest simulation log folder archived:" -ForegroundColor Green
Write-Host $run.FullName
Write-Host "ZIP:"
Write-Host $zip
