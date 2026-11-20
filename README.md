# Phobos Escape Controller V5.0-dev

Development tree based on the V4.3 500-debris validation candidate
(`../V43-BASELINE`, unmodified). Phases 0-1 are implemented: deployment
safety and radar/tracker integrity. See `CHANGELOG.md`.

**Not a competition release. Not deployed. No live STPro validation yet.**

## Offline tests

pytest is not installed; the STPro Python bundles numpy/scipy:

```powershell
$sp = "$env:LOCALAPPDATA\SimDynamX\STPro\Versions\v0.55.0\py\Lib\site-packages"
python -B tests\run_tests.py --site-packages $sp
```

`-B` prevents writing bytecode into the STPro environment.

Offline per-tick benchmark (Phase 1 flow vs current flow):

```powershell
python -B tests\benchmark_pipeline.py --site-packages $sp --seconds 20
```

Benchmark numbers are machine-dependent and exclude STPro API calls.

## Deployment tools (do not run without explicit approval)

- `verify_installed_v5.ps1` — read-only report of what STPro will load.
- `deploy_v5.ps1 [-WhatIf]` — verified backup, deploy, SHA256 verification.
- `rollback_v5.ps1 [-List] [-BackupPath <dir>] [-WhatIf]` — verified restore;
  backs up the current install first.

Backups and `BACKUP_HISTORY.log` live in
`%USERPROFILE%\Documents\STPro-Phobos-Backups`. Backups are never overwritten.
The V4.3 scripts (`deploy_v4_3.ps1`, `rollback_v4_3.ps1`) are preserved
unchanged; note that `rollback_v4_3.ps1` does not back up the current install.

The 500-radar validation simconf must already be installed in
`STU4-Sim\Sims`; these scripts verify it but do not install it.
