"""Generate the Step 0 job list (python step0_jobs.py jobs.json <controller_dir>)."""
import json
import sys

ctl = sys.argv[2]
seeds100 = [100] + list(range(1, 100))
seeds30 = [100] + list(range(1, 30))
jobs = []
# D. current controller (df39ee0), full missions
for s in [100] + list(range(1, 12)):
    jobs.append({"label": f"ctl|{s}", "args": ["--pilot", "controller", "--controller-dir", ctl, "--seed", str(s),
                                                "--opt", "t_max_s=900"]})
# A. launch-speed sweep, no avoidance and corridor oracle
for v in (5.5, 6.0, 7.0, 8.0):
    for s in seeds100:
        for c in (False, True):
            jobs.append({"label": f"main|{v}|{s}|{int(c)}", "args": ["--seed", str(s), "--speed", str(v)] + (["--corridor"] if c else [])})
# B. dangerous scenarios
T_LATCH = {6.0: 300.0, 7.0: 243.0}
for v in (6.0, 7.0):
    spec = [{"fraction": f, "t": round(30 + f * (T_LATCH[v] - 30), 1), "cross_speed": 0.5} for f in (0.25, 0.55, 0.85)]
    for s in seeds30:
        jobs.append({"label": f"threat3|{v}|{s}", "args": ["--seed", str(s), "--speed", str(v), "--corridor", "--threats", json.dumps(spec)]})
for s in seeds30:
    for c in (False, True):
        jobs.append({"label": f"hunting|7.0|{s}|{int(c)}", "args": ["--seed", str(s), "--speed", "7", "--hunting"] + (["--corridor"] if c else [])})
        jobs.append({"label": f"livefast|7.0|{s}|{int(c)}", "args": ["--seed", str(s), "--speed", "7", "--opt",
                     'debris_extra_velocity={"profile":"live"}', "--opt", f"debris_extra_seed={s}"] + (["--corridor"] if c else [])})
# C. sensitivity at 7 m/s with corridor oracle (nominal = main|7.0|s|1)
SENS = {
    "rot_transpose": ["--opt", "rotation_transpose=true"],
    "jets_95_1deg": ["--opt", 'jet_perturb={"thrust_scale":0.95,"normal_deg":1.0}'],
    "jets_90_2deg": ["--opt", 'jet_perturb={"thrust_scale":0.90,"normal_deg":2.0}'],
    "miss30": ["--miss", "30"],
    "miss45": ["--miss", "45"],
    "debris_sigma0.1": ["--opt", 'debris_extra_velocity={"sigma":0.1}'],
    "no_frame_terms": ["--opt", "frame_terms=false"],
    "substep_60hz": ["--opt", "substep_s=0.016666666666666666"],
    "brake_on": ["--brake"],
}
for name, extra in SENS.items():
    for s in seeds30:
        jobs.append({"label": f"sens|{name}|{s}", "args": ["--seed", str(s), "--speed", "7", "--corridor"] + extra})
json.dump(jobs, open(sys.argv[1], "w"), indent=0)
print(len(jobs), "jobs")
