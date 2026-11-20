# V5.0-dev — Phases 0 and 1

Offline-tested only. Not deployed. No live STPro result exists for V5.

## Phase 0 — deployment safety
- `verify_installed_v5.ps1` (read-only): installed controller path, SHA256 and
  `PHOBOS_NAV_V*` tag; per-module comparison with this tree; official
  STU4-Sim file hashes and git cleanliness; validation-simconf hash.
- `deploy_v5.ps1` / `rollback_v5.ps1` / `phobos_deploy_common.ps1`:
  verified, never-overwritten timestamped backups with SHA256 manifests
  under `%USERPROFILE%\Documents\STPro-Phobos-Backups`; append-only
  `BACKUP_HISTORY.log`; official STU4-Sim systems checked against git
  upstream; SHA256 verification after deploy and rollback; automatic restore
  on failure; rollback backs up the current install first. Nothing is written
  into the V5 source tree. `-WhatIf` runs all checks without changes.
- V4.3 scripts preserved unchanged.
- `tests/test_500_debris_validation.py`: simconf path resolves the V5,
  baseline and package layouts (previously failed in V5).
- `tests/run_tests.py`: dependency-free runner (pytest is unavailable).

## Phase 1 — radar and tracker integrity (audit defects D1-D4)
- Cached radar records are identified by their exact raw bytes
  (`RadarObservation.raw_key`), independent of the clock-offset estimate.
  V4.3 keyed on offset-shifted time/position, so every offset change
  re-injected ~450 cached rows as "fresh".
- Radar clock offset is locked after 9 publication samples (overrun samples
  rejected), re-acquired only on a timebase reset or persistent >0.25 s jump.
  Both clocks derive from SimClock, so the true offset is constant.
- `DebrisTracker` is a constant-velocity Kalman filter (position +
  range-rate, per-track 6x6 covariance) replacing fixed-gain alpha-beta:
  - no update from measurements < 0.25 s apart (genuine revisit >= 1.67 s);
  - chi-square association gate (4 dof, p~1e-4) after a coarse prefilter;
  - replayed / out-of-order copies are dropped, never spawn tracks;
  - unassociated measurements re-acquire an overdue track instead of
    creating a duplicate (no ghosts);
  - confirmation counts only independent revisits;
  - expiry follows the measured revisit period (>= 6.0 s, capped 9 s) and
    runs after association (stalls no longer drop tracks);
  - capacity evictions prefer overdue/tentative tracks and are counted;
  - birth velocity uses only the observed radial component (tangential prior
    0), not the shuttle's tangential velocity;
  - debris speed bound 15 m/s (was 100 m/s).
- Controller log (logging only): tag `PHOBOS_NAV_V50`, `radar_clock_locked`,
  `radar_clock_resyncs`, `too_soon`, cumulative reacquired/evicted/expired/
  replays/velocity-clamped counts, `track_ids_issued`, `revisit_s`,
  `track_expiry_s`.
- `tests/radar_replay.py`: Radar.py-faithful replay harness;
  `tests/test_v5_tracker_integrity.py`: 15 regression tests.

# V4.3 500-debris validation candidate

This is a validation-only branch. It preserves the V4.3 navigation and MPC behavior while correcting the sensor/tracker capacity mismatch.

## Changes

1. Separate `COMPETITION_SIM_500_RADAR_VALIDATION.simconf` changes only the Radar system's `NumDebris` from 200 to 500. `SpawnDebris.NumDebris` remains 500. The original `COMPETITION_SIM.simconf` is untouched.
2. Controller tracking capacity increases from 240 to 550 so all 500 radar objects can be retained without routine eviction.
3. Track lifetime increases from 4.5 s to 6.0 s. The official Radar loop remains at 120 Hz; it updates one of 500 objects per iteration, so a complete round-robin refresh is about 4.17 s.
4. Radar clock stale threshold increases from 0.55 s to 0.75 s to tolerate the 500-object array's approximately 0.417 s publication interval (`sortFrequency = NumDebris / 10`).
5. Validation logs now include `track_age_max_s`, `track_age_p95_s`, and `fresh_age_max_s`.
6. The MPC planning subset remains 128 tracks, selected by lowest predicted CPA clearance from the full retained track set. This is retained to protect runtime; full-500 planner evaluation is not enabled.

## Unchanged

- Official `Radar.py`, `SpawnDebris.py`, `JetSelectionActuator.py`, scoring system, thruster CSV, and original `COMPETITION_SIM.simconf`.
- Fuel-related parameters and guidance gains. No fuel optimization is included.

## Validation status

Offline 500-track streaming regression and existing V4.3 tests must pass. A live STPro run with the separate 500-radar simconf is still required. A no-damage run with 500 radar records is the minimum acceptance test; one successful run is not a statistical guarantee.
