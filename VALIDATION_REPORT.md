# V4.3 baseline audit and 500-debris validation candidate

## 1. Baseline source audit

The supplied competition simconf has:

- `SpawnDebris.NumDebris = 500`
- `Radar.NumDebris = 200`
- `Radar.LoopFreqHz = 120`

The official `STU4-Sim/Systems/Radar.py` loads only `Debris0` through `Debris199`. Its `SortedDebrisData` array is therefore built from those 200 objects; it does not sort all 500 and then publish the nearest 200. `NumToDisplay = 50` is a separate UI display limit and is not the sensor-array count.

The saved simconf contains 24 ExplodedTank entities. Sixteen are named `Debris200` or higher and are not read by the original radar configuration.

The V4.3 controller requests up to 500 records (`radar_max_objects = 500`), but receives the 200-record array. The tracker cap is 240. The planner ranks tracks by predicted CPA clearance and evaluates the 128 lowest-clearance tracks per planning cycle.

## 2. November 18 live-log baseline

Unique controller samples parsed from the supplied run:

- 413 samples, simulation time 1.59–414.46 s.
- `radar_available` / `radar_read`: always 200.
- Maximum retained tracks: 240.
- Planner tracks: 128.
- Damage: 0.000 throughout the supplied controller log.
- Scoring log: successful Phobos escape, score -253.6, fuel at scoring event 250.12 kg.
- Avoidance active: 295/413 samples.
- `NO_SAFE_HORIZON`: 109/413 samples.
- `UNAVOIDABLE`: 3/413 samples; closest predicted times were 0–6 s.
- Planner compute time: p95 35.45 ms, maximum 106.38 ms.

This run demonstrates successful mission completion and no recorded damage for the 200 objects read by the radar. It does **not** validate avoidance against all 500 spawned objects.

## 3. Validation candidate modifications

- Separate simconf changes only Radar `NumDebris` from 200 to 500.
- Radar loop frequency remains 120 Hz. The radar updates one object per loop and republishes the sorted array every 50 updates; expected full sweep is approximately 4.17 s and publication interval approximately 0.417 s.
- Tracker capacity: 550.
- Track maximum age: 6.0 s.
- Radar newest-record stale threshold: 0.75 s.
- Planner shortlist remains 128, chosen from the complete retained set by predicted CPA clearance.
- Added maximum, p95, and fresh-measurement track-age log fields.
- Fuel/guidance/actuator parameters remain unchanged.

## 4. Offline validation

- Python compilation: passed.
- Existing V4.3 tests: 30 passed.
- New 500-debris tests: 3 passed.
- Total: **33 passed**.
- Synthetic 500-track round-robin test retained all 500 tracks across repeated 4.17-second sweeps.

A local synthetic tracker benchmark with 500 retained tracks and 50 new observations per radar publication measured approximately 6.7 ms median and 10.3 ms p95 per controller-poll tracker update in this environment. This is not a STPro runtime guarantee.

The planner remains capped at 128 because a local synthetic benchmark increased from about 39 ms median at 128 planning tracks to about 48 ms at 256 and 74 ms at 500. Full-500 MPC evaluation is therefore not enabled in this two-day validation branch.

## 5. Release gate

**Not yet live-validated.** The candidate must pass a short STPro integration check with 500 radar records, all 500 retained tracks, track age below 6 s, stable radar clock, acceptable planner runtime, and zero damage. Then a full mission must be completed and its threat intervals reviewed. Only after that gate passes should fuel optimization begin.
