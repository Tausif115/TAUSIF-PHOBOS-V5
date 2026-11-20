# TAUSIF - PHOBOS V5 DEVELOPMENT

## Authoritative baseline

The exclusive development starting point is the V4.3 500-debris
validation candidate copied into this V5-DEVELOPMENT directory.

The untouched original is in ../V43-BASELINE.

Never use older experimental candidates, candidate_v2 through
candidate_v17, verified_controllers, or the November 1 baseline
as the development starting point.

## Simulation reference

The read-only simulation source is:

_reference/STU4-Sim/

The required integration configuration is:

_reference/STU4-Sim/Sims/COMPETITION_SIM_500_RADAR_VALIDATION.simconf

Do not substitute the original COMPETITION_SIM.simconf.

Do not modify the simulation reference or physics definitions.

## Verified validation-candidate facts

- 500 debris spawned.
- Radar NumDebris changed from 200 to 500.
- Tracker capacity: 550.
- Track maximum age: 6 seconds.
- Radar stale threshold: 0.75 seconds.
- Planner shortlist: 128.
- Planner shortlist of 128 is intentional.
- Fuel and actuator parameters were intentionally unchanged.
- Offline tests: 33 passed.
- Live STPro validation: NOT YET PASSED.

## Mandatory validation sequence

A. Short 20-30 second STPro integration run.

Required:
- radar_available=500
- radar_read=500
- radar_decoded=500
- tracks grows toward and remains near 500
- confirmed tracks increase with repeat measurements
- radar_clock_ready=1
- radar_latest_age_s < 0.75
- track_age_max_s < 6.0
- track_age_p95_s comfortably below 6.0
- planning_tracks=128
- planner runtime not persistently worse than baseline
- damage=0.000

If radar counts remain 200, tracks cap near 240, track age
repeatedly reaches 6 seconds, or RADAR_HOLD repeats, stop.

Do not begin a full mission until the short check passes.

B. Full mission.

Inspect every NO_SAFE_HORIZON and UNAVOIDABLE interval,
selected maneuver, track age, radar coverage, closest predicted
clearance, planner runtime, damage, score and fuel consumption.

Zero damage alone does not prove 500-object avoidance.

## Development priorities

1. Complete live 500-debris validation.
2. Confirm full mission and inspect threat events.
3. Only after passing the debris gate, begin conservative
   fuel optimization.
4. Preserve collision avoidance and escape capability.

Targets:
- Average score: +40
- Propellant: 60-70 kg

These are targets, not proven results.

## Strict file safety

- Only modify files in this V5-DEVELOPMENT directory.
- Never modify ../V43-BASELINE.
- Never modify _reference/STU4-Sim.
- Never overwrite the official simulator systems.
- Never deploy without explicit user approval.
- Inspect deploy_v4_3.ps1 before proposing a deployment.
- Preserve rollback capability.

## First task

Perform a READ-ONLY engineering audit.

Trace the actual Controller.py execution entry point,
all imports, radar integration, tracking, planning,
guidance, actuator interface and deployment mechanism.

Inspect available historical logs.

Identify defects and propose a prioritized plan.

Do not modify any files until the user explicitly approves.

Never claim STPro validation or successful simulation results
unless they were actually executed and verified.
