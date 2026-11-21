# PHOBOS V5.1 — deployment, final run, recording and log collection

Run every command from PowerShell in
`C:\Users\User\Desktop\TAUSIF-PHOBOS-V5\V5-DEVELOPMENT`. Nothing here modifies
STU4-Sim (actuator, radar, scoring, spawner, simconfs, thruster CSV).

## 0. Preconditions
- STPro closed (no simulation running).
- `git status` clean on `v5-development`; note the commit hash being deployed.
- `FINAL_SUBMISSION.simconf` present in
  `%LOCALAPPDATA%\SimDynamX\STPro\Assets\Repos\STU4-Sim\Sims\`
  (supplied by the organisers; this project does not create or edit it).

## 1. Deploy (requires explicit approval)
```powershell
powershell -ExecutionPolicy Bypass -File .\verify_installed_v5.ps1     # what is installed now (V50)
powershell -ExecutionPolicy Bypass -File .\deploy_v5.ps1 -WhatIf       # all checks, no changes
powershell -ExecutionPolicy Bypass -File .\deploy_v5.ps1               # timestamped backup + install + SHA256 verify
powershell -ExecutionPolicy Bypass -File .\verify_installed_v5.ps1     # must report tag PHOBOS_NAV_V53 and all modules MATCH
```
The deploy copies `Controller.py` and every `experimental\phobos_nav\*.py`
(including the new `ballistic.py`) and keeps a verified backup under
`%USERPROFILE%\Documents\STPro-Phobos-Backups\<timestamp>_pre-deploy-v5`.

## 2. Rollback (any time)
```powershell
powershell -ExecutionPolicy Bypass -File .\rollback_v5.ps1 -List                     # list verified backups
powershell -ExecutionPolicy Bypass -File .\rollback_v5.ps1 -BackupPath "<folder>"     # restore V50 (or any backup)
powershell -ExecutionPolicy Bypass -File .\verify_installed_v5.ps1
```
Rollback backs up the current install first. Fast in-place fallback without
redeploying: in `experimental\phobos_nav\config.py` set
`guidance_mode = "legacy_mpc"` (V5.0 planner path) — only if V5.3 misbehaves.

## 3. Validation run (validation simconf)
Open `COMPETITION_SIM_500_RADAR_VALIDATION.simconf` (after the STU4-Sim update its
debris equal FINAL_SUBMISSION's, seed 8193). In the on-screen/console log confirm:
- `PHOBOS_NAV_V53`, `radar_available=500`, `damage=0.000` before the latch.
- WAIT (first ~20 s): `cmd_accel_mps2` ~0.000-0.002 (gravity hold), `range_m`
  NOT increasing (an increasing range means the shuttle is sinking toward the pad),
  `fuel_kg` rising only ~0.05-0.07 kg/s; `v53_launch_state` GRAVITY_HOLD ->
  RADAR_ACQUISITION -> LAUNCH_COMMIT (SAFE_HOLD / EMERGENCY_LIFTOFF only with a
  logged pad or corridor conflict: check `v53_pad_clear_m`, `v53_pad_tc_s`, `v53_pad_cls`).
- `v51_phase` `WAIT -> BURN` at ~13-21 s, then `range_m` decreasing and `fuel_kg`
  rising ~0.5 kg/s during BURN, `safety=CLEAR`.
- Track health: `v53_unreliable` / `v53_suspect` small (a few), `v53_res_p90`
  < ~0.1 m/s, `v53_spd_p50` ~2-5 m/s (live debris speeds). `TRACK_DIAGNOSTIC`
  lines carry per-track `rr_res_mps` / `vsig_mps` for post-run analysis.
- Known open issue: target-ship contact after the latch (geometry unverified;
  not scored).
Abort and roll back if damage rises before the latch or range increases during WAIT.

## 4. Final competition run (recorded)
1. Start the screen recorder (full screen, include STPro window and clock), then
   open `FINAL_SUBMISSION.simconf` — verify the file name in the STPro title/sim list.
2. Start the simulation; do not touch controls. Expected (offline prediction,
   not a guarantee): gravity hold, launch ~20 s, burn ~70 s, coast, fly-by
   success ~250-260 s.
3. Keep recording until the Scoring alert "Successfully escaped Phobos with a
   score of ..." and its time/fuel/damage lines are visible, plus ~20 s.
4. Stop the simulation, then the recording.

## 5. Log collection (immediately after the run)
```powershell
powershell -ExecutionPolicy Bypass -File .\collect_v4_3_logs.ps1      # copies the newest run folder
```
Or manually: copy the newest folder in `%LOCALAPPDATA%\SimDynamX\STPro\Logs\`
(contains `*_PySys_Controller.log`, `*_PySys_Scoring.log`, the as-run simconf
and `run_uuid.txt`) to a dated folder outside STPro. Record the Scoring lines
(score, elapsed time, fuel, damage) and `run_uuid.txt` with the video.
