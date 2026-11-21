# V5.2-dev — live-run fixes (offline bench only; NOT deployed, no STPro run)

## Why: live V5.1 run bde2f2fb (STPro, validation simconf, 57 s) FAILED
- Unpowered WAIT: shuttle sank under gravity (-0.0118 m/s^2), touched the pad at
  t=16.2 s (-1.45 m), settled at -2.1 m; damage 42.6 + 2.6 + 0.4 = 45.6 J.
- Launch gate (95 % confirmed tracks) unreachable live (plateau 93-94 %, ~6.5
  reacquisitions/s) -> 30 s fallback.
- Emergency dodges against tracker artefacts: conflict 90 s ahead (no clearing
  candidate) and a track extrapolated with a corrupted velocity (passed 1.96 m
  from the shuttle, no damage, closing ~6 m/s); 3.6 of 3.9 m/s planned dv sideways.

## Fixes (`experimental/phobos_nav/ballistic.py`, `config.py`)
1. WAIT gravity hold: per-tick command = 0.2 x (0.01 m/s up - velocity), capped at
   0.0015. JetSelectionActuator delivers commands below ~dt x authority
   proportionally (on-time = LP solution < one tick), so per-tick dv == command;
   steady state 0.0002 = g/60.
2. Launch gate 0.90 confirmed, held for 1 s (fallback 30 s unchanged).
3. Phantom robustness: tracks faster than 1.5 + 0.015 t m/s (physical bound from
   SpawnDebris + gravity fall) are reset to their last measured position with zero
   velocity and treated as tentative; tentative tracks guard only 25 s ahead;
   conflicts must persist 2 checks unless <= 8 s away; no non-clearing (emergency)
   dodge for conflicts > 20 s away; a dodge stops once coasting is clear.
- Log tag PHOBOS_NAV_V52; new fields v52_implausible, v52_deferred_emerg, v52_aborts.
- Not changed: burn-phase re-aim policy (known cost issue, see below).

## Bench additions (test-only)
- `tests/missionbench/tracker_faults.py`: live-calibrated tracker faults
  (reacquisitions 6.5/s with ~1 m/s/axis velocity error, implausible-velocity
  tracks 0.35/s, optional phantoms aimed at the shuttle).
- Pad proxy `pad_max_drop_m` (max sink toward Phobos; live contact at ~1.4 m).
- SpawnDebris test copy follows the official seed line (100 -> 8193).

## Offline results, 95 missions (bench predictions, not competition results)
Bench seed 100 = FINAL debris (baked positions + official seed 8193).
| class | ver | runs | success, 0 collisions | score median / mean | worst / best | fuel median (max) | max sink |
|---|---|---|---|---|---|---|---|
| FINAL clean | V5.2 | 1 | 1 | 47.4 | - | 38.5 kg | 0.01 m |
| FINAL clean | V5.1 | 1 | 1 | 46.7 | - | 39.0 kg | 0.75 m |
| FINAL + faults (+phantoms) | V5.2 | 6 | 6 | 37.5 / 39.6 | 35.6 / 45.7 | 46.5 (47.9) kg | 0.01 m |
| FINAL + faults (+phantoms) | V5.1 | 6 | 5 | -98.1 / -101.7 | -143.7 / -59.8 | 130.6 (168.8) kg | >= 5.3 m |
| 20 seeds clean | V5.2 | 20 | 20 | 38.4 / 37.5 | 3.9 / 49.1 | 45.3 (72.6) kg | 0.01 m |
| 20 seeds clean | V5.1 | 20 | 20 | 38.6 / 38.9 | 20.5 / 49.7 | ~45 kg | ~0.75 m |
| 20 seeds + faults | V5.2 | 20 | 20 | 27.7 / 24.6 | -12.7 / 46.8 | 52.9 (85.5) kg | 0.01 m |
| 20 seeds + faults | V5.1 | 20 | 16 | -56.6 / -65.7 | -146.7 / -5.0 | 110 (177) kg | >= 5.3 m |
| 3 large threats + faults | V5.2 | 11 | 10 | 23.3 / 15.9 | -10.8 / 37.2 | 57.5 (83.4) kg | 0.01 m |
- Phantoms (3 per run): V5.2 output identical to the same runs without phantoms.
- Launch 11.8-12.5 s in the bench; live confirmation is slower (90 % at 19.3 s
  in bde2f2fb) -> expected live launch ~20 s (~ -2 points).
- Ascent: V5.2 hold keeps vertical speed +0.009 m/s; burn upward acceleration
  0.096-0.113 m/s^2 (bench) vs 0.092-0.102 m/s^2 measured live (V5.1).
- Remaining V5.2 collision: threats+faults seed 10 (t=175 s, injected large
  threat, 8 m/s): fault-corrupted velocity on that track at closest approach;
  passes without faults (30.5); V5.2 gates were not involved (dodges at 25/13/6 s).
- Known cost issue (not fixed): burn-phase re-aims are accepted without a dv
  comparison; usually ~free (median burn fuel 39.9 vs 39.4 kg) but chains late in
  the burn cost up to +32 kg (seed 17 clean: 71.9 kg burn, score 3.9).

## Tests
91 passed (V5.1 tests updated for the hold/persistence; 3 new V5.2 tests).

# V5.1-dev — fly-by guidance (offline bench only; NOT deployed, no STPro run)

## Guidance (`experimental/phobos_nav/ballistic.py`, wired in `Controller.py`)
- Fly-by interception: success latches at 50 m (official Scoring.py); aim point
  35 m from the target (8 alternatives), no braking before the latch.
- Gravity-aware targeting (Phobos point mass + rotating-frame/Mars-tide terms),
  launch at 7 m/s once >=95 % of tracks are confirmed (~11 s), burn-coast-trim:
  command is exactly zero while coasting (official actuator fires full thrust
  for any non-zero command); trims re-solved every 2 s, deadband 0.03 m/s,
  mandatory terminal accuracy checks inside 400 m.
- Predictive avoidance every 1 s: all tracks propagated under gravity for 90 s;
  margin 6 m + 0.07 m/s x (t + age/2) (+0.8 m/s for tentative tracks); 80
  impulse candidates; cheapest clearing one chosen (speed-up preferred,
  lateral cost includes the return). During the launch burn: zero-cost aim
  change first, then exact finite-burn "fold" evaluation, else interrupt the
  burn and dodge immediately; commitment 8 s; partial returns when blocked.
- Rollback switch: `guidance_mode = "legacy_mpc"` (V5.0 planner, unchanged).
- Log tag PHOBOS_NAV_V51; `v51_*` log fields (phase, dv, trims, avoids, ...).

## Offline mission bench (`tests/missionbench/`)
Official Radar.py, JetSelectionActuator.py, Scoring.py, SpawnDebris.py and the
real Controller.py run verbatim against a mock SpaceTeams API (STPro's Python
3.12.10 / NumPy 2.5.4 / SciPy 1.18.1). SpawnDebris seeds via a controlled copy
(seed line only); seed 100 reproduces the baked debris exactly. Modelled (no
source): rigid-body propagation, ApplyGravity (1 Hz), collisions (no damage).
Live V50 cross-check: fuel rate within ~7 %, NO_SAFE_HORIZON share 57 vs 58 %;
live tracker churn NOT reproduced (live debris less predictable than modelled).

## Offline results (bench predictions, not competition results)
| scenario | runs | collision-free success | score median / mean / worst | fuel median |
|---|---|---|---|---|
| V5.1 nominal seeds 100,1-32 | 33 | 33 | 44.5 / 39.9 / 6.1 | 40.6 kg |
| V5.1 live-like fast debris | 11 | 10 (pad hit at 1.6 s) | 47.5 / 46.6 / 42.6 | 38.4 kg |
| V5.1 3 injected path threats | 11 | 11 | 29.3 / 30.0 / 20.0 | 52.3 kg |
| V50 (ccfaf61) seeds 100, 1 | 2 | 2 | -225.4, -316.3 | 244, 309 kg |
Min debris clearance >= 7.0 m in all non-artefact runs. Guidance compute: loop
work p95 ~5 ms, worst tick ~70 ms (real timing, bench host).

## Tests
`tests/test_v51_guidance.py` (6). Suite: 88 passed.

# V5.0-dev — Phase A (tracker retuning from live run 0555d4ef, offline)

Offline-verified only. Not deployed, not run in STPro. Planner, guidance and
actuator behaviour unchanged.

## Live findings addressed (run 0555d4ef, commit ccfaf61)
- 550 cap hit at startup, 659 IDs for 500 objects, 615 re-acquisitions, 66
  evictions: ~77 % of tracks born from Radar.py's initial snapshot (raw
  timestamp 0.0) failed their first loop update and were duplicated.
- track_age_max 6.19 s (> 6.0 criterion): orphaned/duplicate tracks aging out
  at expiry = max(6.0, 1.3 x 4.71 s revisit) = 6.12 s.

## Changes (`tracking.py`, `config.py`)
- Provisional tracks (`track_provisional_init_snapshot`, default on): tracks
  born from raw-timestamp-0 records are replaced by their object's first loop
  measurement (counted `provisional_replaced`, not as re-acquisition); evicted
  first at capacity; removed once the first sweep is complete. Their
  snapshot-to-first-measurement interval is excluded from the revisit
  estimate.
- Expiry = 1.1 x p99 measured revisit + 0.60 s (`track_expiry_mode`
  "revisit_p99"; "legacy" = Phase 1 rule, tests only). Live revisit 4.71 s ->
  5.78 s. A valid track is at most sweep + sweep/10 + one poll (~5.2 s) old.
  6.0 s before revisits are measured; capped at 9.0 s.
- Round-robin association gate (`track_min_revisit_fraction` 0.75): official
  Radar.py refreshes `iteration % numDebris` of an unsorted list, so a track
  cannot be updated by a measurement sooner than 0.75 x measured revisit
  (~3.5 s live). Fixes fast-debris mis-association found in replay seed 100
  (an 8.3 m/s object's revisit stolen by a track updated 2.47 s earlier).
  Provisional tracks exempt.
- Unchanged after evaluation: tangential prior 5 m/s (10 m/s raised clamps,
  duplicates and expiries), speed bound 15 m/s (25 m/s only hid clamps),
  re-acquisition fraction 0.75.
- `Controller.py`: log field `provisional_replaced_total` (logging only).

## Replay harness (`tests/radar_replay.py`)
- Live debris speed mix (75 % 0.5-5 m/s, 25 % 5-10 m/s), radar loop overhead
  0.26 (~106 Hz, as live), initial-snapshot modes consistent / zero_velocity
  / stale, radar hitch injection.
- Evaluator matches each track to truth at the track's own last-update time
  (extrapolated matching mislabelled correct tracks near crossing fast debris).

## Offline results (12 SpawnDebris-style seeds: 100, 1-11; 30 s; worst [mean])
| init snapshot | config | 550 cap hit | IDs | evictions | expiries | age max s |
|---------------|--------|-------------|-----|-----------|----------|-----------|
| consistent    | ccfaf61 | 0/12 | 502 [500] | 0 | 2 [0.25] | 6.09 [5.25] |
| consistent    | Phase A | 0/12 | 500 [500] | 0 | 0 | 5.19 [5.18] |
| zero_velocity | ccfaf61 | 12/12 | 743 [723] | 184 [165] | 66 [58] | 6.12 [5.82] |
| zero_velocity | Phase A | 0/12 | 506 [503] | 0 | 6 [2.8] | 5.80 [5.28] |
| stale (10 s)  | ccfaf61 | 12/12 | 748 [731] | 191 [169] | 73 [62] | 6.12 [5.90] |
| stale (10 s)  | Phase A | 12/12 | 648 [635] | 94 [80] | 62 [55] | 5.75 [5.30] |
Stale: cap still reached during the first sweep; only placeholders evicted
(0 confirmed evictions), full 500-object coverage in every run.
`tracker.update` cost unchanged (p50 3.1 ms both). Pipeline benchmark: 0
ticks over 16.67 ms (sparse and dense).

## Tests
- `tests/test_v5_phase_a.py` (13 tests). Phase 1 equivalence test runs with
  Phase A disabled. Healthy-replay speed bound follows true debris speed.
  Suite: 82 passed.

# V5.0-dev — Phase 2b (60 Hz deadline engineering, offline)

Offline-benchmarked only. Not deployed. No live STPro timing exists for V5.
Navigation results are unchanged: every change is equivalence-tested against
frozen Phase 1 code (`tests/reference/`, commit 307cacb).

## Planner profile (Phase 2, 500 debris)
Rollout ~17.6 ms of ~19.4 ms per plan: swept-CPA geometry 9.2 ms, min/risk
reductions 2.0 ms, guidance propagation 3.3 ms, limiter callback 3.1 ms.

## Changes
- Exact bounded clearance evaluation (`planning.py` `_clearance_bounds`,
  `_resolve_clearance`, `_evaluate_chunks`): trajectories are propagated
  first; per-chunk bounding spheres give provable clearance lower bounds;
  every unit that can carry risk or hold the minimum is evaluated with the
  Phase 1 formula. Identical closest-track ids in all tests; values within
  rounding (<=1.7e-13 relative); result independent of chunk size.
- Two-stage planning API: `plan_stage_one` (shortlist, candidates,
  propagation, bounds) and `plan_stage_two` (exact clearance, selection,
  command); `plan()` = stage one + stage two (bit-identical outputs).
- `phobos_nav/scheduling.py` `DeadlinePlanner`: runs a stage only if it fits
  `control_tick_budget_s` (14 ms) given time already spent in the tick
  (90th-percentile recent stage duration); requests latch; at most
  `planner_max_stage_wait_ticks` (2) ticks of deferral, then forced; jobs
  older than 0.1 s (the planner's modelled control latency) are re-planned.
  `Controller.py` uses it instead of a synchronous `planner.plan()`.
- Guidance propagation: step-invariant constants hoisted; terminal-capture
  math skipped when no candidate is inside the capture radius (identical).
- `actuator_model.limit_acceleration_body_batch`: fast path when every axis
  has authority (as the controller requires); bit-identical to Phase 1
  (3000+ random cases); 48 -> 27 us per call.
- Tracker: batched Kalman update (`_kf_update_batch`), vectorized
  association bookkeeping and measurement covariances, O(log n)
  `StateHistory.sample` (bit-identical). Whole-tracker replay equivalence
  vs Phase 1: identical per-poll stats/ids/hits, states within 1.1e-13.
- Radar ingestion cap `radar_max_new_records_per_tick` (100, nearest first):
  spreads the one-time 500-record bootstrap over ~5 ticks; never binds in
  steady state (~50 new records per publication).
- Controller log: `plan_stage1_ms`, `plan_stage2_ms`, `plans_completed_total`,
  `plan_deferrals_total`, `plan_forced_total`, `plan_discarded_total`,
  `radar_deferred_new`.

## Offline results (`tests/benchmark_pipeline.py`, 500 debris, 25 s, 60 Hz)
| ms                       | Phase 1 sparse | current sparse | Phase 1 dense | current dense |
|--------------------------|----------------|----------------|---------------|---------------|
| all ticks p50 / p99      | 37.4 / 72.5    | 7.7 / 14.0     | 37.2 / 72.7   | 8.5 / 14.6    |
| planning ticks p99       | 84.6           | 13.6           | 84.5          | 15.0          |
| ticks over 16.67 ms      | 68.3 %         | 0.29 %         | 68.3 %        | 0.50 %        |
Planner call alone: sparse ~6.6 ms, dense ~11.5 ms (Phase 1 ~28 ms).
Residual over-budget ticks are isolated 3-5x stalls of a single operation;
the same desktop stalls a fixed 2 ms workload to p99 6.9 ms / max 50 ms.
Excludes STPro API calls; with 2 ms assumed API overhead: 0.64-0.78 %.

## Tests
- `tests/test_v5_deadline.py` (12 tests); frozen references
  `tests/reference/tracking_phase1.py`, `actuator_model_phase1.py`.

# V5.0-dev — Phase 2 (offline performance)

Offline-benchmarked only. Not deployed. No live STPro timing exists for V5.
Navigation mathematics unchanged: every optimization is equivalence-tested
against frozen Phase 1 copies (`tests/reference/`, commit 307cacb).

## Profiling (Phase 1, 500 debris, offline)
- Per tick: radar block ~20 ms (decode of all 500 records, ~450 of them
  cached copies; `StateHistory.sample` called 500x/tick), threat scans
  ~4.5 ms, planner ~36 ms per call (Python loop over all tracks in the
  shortlist, 40-step rollout). Mean ~33 ms/tick (~31 Hz compute ceiling).

## Changes
- Pre-decode filter: `DebrisTracker.select_new_records()` recognises cached
  records by raw bytes; `Controller.py` decodes only new records and passes
  `prefiltered_duplicates` to `tracker.update()`. Tracker state is
  bit-identical with the filter on or off (test).
- `phobos_nav/threats.py`: vectorized nearest-track distance and
  warning-envelope TTC (replace the per-track loops formerly at
  Controller.py:300-353 and 550-558). Tracks lacking `position_at` are
  excluded exactly as before (the old nearest-track loop would have raised).
- Planner `_prepare_track_arrays`: vectorized shortlist (same ordering).
- Planner `_rollout_candidates`: precomputed debris step positions, direct
  segment displacement, masked division, einsum norms; redundant collision
  mask removed. Max relative difference vs Phase 1: 4.3e-14 (rounding);
  identical decisions over a 25-step 500-track sequence (test).
- Radar clock `_max_timestamp`: vectorized with exact loop fallback.
- Controller log (logging only): `radar_decoded` keeps its V4.3 meaning
  (valid records decoded this poll + valid records decoded earlier and
  still present); new `radar_new_decoded`, `radar_ms`, `threat_ms`,
  `loop_interval_ms`, `prev_loop_work_ms`, `loop_overruns_total`.

## Offline benchmark (`tests/benchmark_pipeline.py`, planner every 2 ticks)
| ms (p50)            | Phase 1 | Phase 2 |
|---------------------|---------|---------|
| radar block         | 19.83   | 1.23    |
| threat scans        | 4.45    | 0.81    |
| planner call        | 35.81   | 18.33   |
| tick without plan   | 24.27   | 2.07    |
| tick with plan      | 60.12   | 20.66   |
| mean work per tick  | 32.75   | 7.00    |

Excludes STPro API calls (state reads, frame transforms, SetCommand).

## Tests
- `tests/test_v5_performance_equivalence.py` (9 tests) and frozen
  references `tests/reference/planning_phase1.py`,
  `tests/reference/controller_threats_phase1.py`.

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
