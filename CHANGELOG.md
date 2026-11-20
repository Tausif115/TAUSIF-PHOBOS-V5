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
