"""Run one offline mission in the Step 0 bench and print a JSON result line.

Usage (STPro's own Python, so NumPy/SciPy match the live sim):
  python run_mission.py --pilot ref --seed 100 --speed 7 [--corridor] [--brake]
  python run_mission.py --pilot controller --controller-dir <dir with Controller.py>
Options --opt key=value override bench options (see world.Bench.DEFAULTS).
Scores are OFFLINE PREDICTIONS from the bench, not STPro results.
"""
import argparse
import json
import pathlib
import re
import sys
import tempfile
import time as _time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import world as W  # noqa: E402

OFFICIAL = W.ASSETS / "Repos" / "STU4-Sim"
SIMCONF = OFFICIAL / "Sims" / "COMPETITION_SIM_500_RADAR_VALIDATION.simconf"


def parse_opts(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = json.loads(v) if v[:1] in "[{0123456789-.tfn" and v not in ("nan",) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", choices=("ref", "controller"), default="ref")
    ap.add_argument("--controller-dir")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--speed", type=float, default=7.0)
    ap.add_argument("--miss", type=float, default=40.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--corridor", action="store_true")
    ap.add_argument("--brake", action="store_true")
    ap.add_argument("--hunting", action="store_true", help="official hunting code path enabled (test copy)")
    ap.add_argument("--threats", default="", help="JSON list of injected crossing threats")
    ap.add_argument("--opt", action="append")
    ap.add_argument("--label", default="")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    opts = parse_opts(args.opt)
    wall0 = _time.time()
    work = pathlib.Path(tempfile.mkdtemp(prefix="missionbench_"))

    debris = None
    if args.seed != 100:
        debris = W.generate_debris(SIMCONF, args.seed, work)
    bench = W.Bench(SIMCONF, opts)
    bench.build_entities(debris_override=debris)
    if args.threats:
        import threats
        threats.inject(bench, json.loads(args.threats))
    sp = bench.system_params
    jet = opts.get("jet_perturb")
    if jet:     # controlled test copy of the official thruster CSV (official file untouched)
        import numpy as np
        src = OFFICIAL / "Data" / "SpaceBuggyJetConfigNoMains.csv"
        lines = open(src, encoding="utf-8").read().splitlines()
        rng = np.random.default_rng(int(jet.get("seed", 0)))
        out = [lines[0]]
        for line in lines[1:]:
            if not line.strip():
                continue
            v = [float(x) for x in line.split(",")]
            n = np.asarray(v[3:6]); axis = rng.normal(size=3); axis -= axis.dot(n) * n; axis /= np.linalg.norm(axis)
            ang = np.radians(float(jet.get("normal_deg", 0.0)))
            n = n * np.cos(ang) + np.cross(axis, n) * np.sin(ang)
            v[3:6] = list(n); v[6] *= float(jet.get("thrust_scale", 1.0))
            out.append(",".join(repr(float(x)) for x in v))
        csv = work / "jets_perturbed.csv"
        csv.write_text(chr(10).join(out) + chr(10), encoding="utf-8")
        sp["JetSelectionActuator"] = dict(sp["JetSelectionActuator"], JetConfigName=str(csv))
    bench.add_system("ApplyGravity", W.gravity_system_factory(bench), sp["ApplyGravity"])
    spawn_script = (OFFICIAL / "Systems" / "SpawnDebris.py" if args.seed == 100 and not args.hunting
                    else W.spawn_debris_copy(args.seed, work, hunting=args.hunting))
    bench.add_system("SpawnDebris", spawn_script, sp["SpawnDebris"], overhead=bench.o["spawn_overhead"], seed=1)
    bench.add_system("JetSelectionActuator", OFFICIAL / "Systems" / "JetSelectionActuator.py",
                     sp["JetSelectionActuator"], overhead=bench.o["actuator_overhead"], seed=2)
    bench.add_system("Scoring", OFFICIAL / "Systems" / "Scoring.py", sp["Scoring"], seed=3)
    if args.pilot == "controller":
        bench.add_system("Radar", OFFICIAL / "Systems" / "Radar.py", sp["Radar"],
                         start_delay_s=bench.o["radar_start_s"], overhead=bench.o["radar_overhead"], seed=4)
        script = pathlib.Path(args.controller_dir) / "Controller.py"
        bench.add_system("Controller", script, sp["Controller"], start_delay_s=bench.o.get("controller_start_s", 1.0))
    else:
        params = dict(sp["Controller"])
        params.update(LaunchSpeed=args.speed, MissDistance=args.miss, MissAzimuthDeg=args.azimuth,
                      Corridor=float(args.corridor), PostSuccessBrake=float(args.brake),
                      FrameTerms=float(bool(bench.o["frame_terms"])), ShuttleRadius=bench.o["shuttle_radius_m"])
        for k in ("TrimPeriod", "TrimDeadband", "TrimFirstAfterBurn", "LaunchTime"):
            if k in opts:
                params[k] = float(opts[k])
        bench.add_system("RefAutopilot", HERE / "reference_autopilot.py", params, start_delay_s=0.0)

    fuel_at = {}
    original_alert = bench.record_alert

    def alert_hook(message, tag):
        if "Successfully escaped" in message and "fuel" not in fuel_at:
            fuel_at["fuel"] = float(bench.entities["SpaceBuggy"].params["Stats"]["FuelUsed"])
            fuel_at["dv"] = bench.world.delta_v_shuttle
        original_alert(message, tag)
    bench.record_alert = alert_hook
    fault_stats = None
    if args.pilot == "controller" and opts.get("tracker_faults") is not None:
        # live-like tracker faults (test-only wrapper around the real tracker)
        sys.path.insert(0, str(pathlib.Path(args.controller_dir) / "experimental"))
        import phobos_nav.tracking as _tracking
        import tracker_faults
        fault_stats = tracker_faults.install(_tracking, opts["tracker_faults"])
    bench.run()

    w = bench.world
    alerts = {tag: msg for _, tag, msg in bench.alerts}
    score = None
    m = re.search(r"score of (-?[0-9.]+)", alerts.get("Winning", ""))
    if m:
        score = float(m.group(1))
    elapsed = re.search(r"= ([0-9.]+) s", alerts.get("SET", ""))
    success_t = bench.success_t
    contacts = w.contacts
    pre = [c for c in contacts if success_t is None or c[0] <= success_t]
    post = [c for c in contacts if success_t is not None and c[0] > success_t]
    finite = w.min_clearance[1:][w.min_clearance[1:] < float("inf")]
    result = dict(
        label=args.label, pilot=args.pilot, seed=args.seed, speed=args.speed, miss=args.miss,
        corridor=args.corridor, brake=args.brake, hunting=args.hunting, threats=bool(args.threats),
        options=opts, success=success_t is not None, score=score,
        scoring_elapsed_s=float(elapsed.group(1)) if elapsed else None, latch_world_t=success_t,
        fuel_at_latch_kg=fuel_at.get("fuel"), dv_at_latch=fuel_at.get("dv"),
        fuel_end_kg=float(bench.entities["SpaceBuggy"].params["Stats"]["FuelUsed"]),
        dv_total=w.delta_v_shuttle, t_end=w.t,
        collisions_pre_latch=len(pre), collisions_post_latch=len(post),
        collision_detail=[(round(c[0], 2), c[1], round(c[2], 2)) for c in contacts[:20]],
        min_debris_clearance_m=float(finite.min()) if finite.size else None,
        target_min_range_m=w.target_min_range, target_contact_t=w.target_contact_t,
        pad_max_drop_m=w.max_drop, tracker_faults=fault_stats,
        speed_at_latch=None, wall_s=round(_time.time() - wall0, 1),
        refap_log=[m for _, tag, m in bench.logs if tag == "RefAP"][:40],
    )
    if success_t is not None:
        k = min(range(len(w.trace)), key=lambda i: abs(w.trace[i][0] - success_t))
        result["speed_at_latch"] = float((w.trace[k][2] ** 2).sum() ** 0.5)
    if args.pilot == "controller":
        lines = [(t, msg) for t, tag, msg in bench.logs if msg.startswith("PHOBOS_NAV_V")]
        result["controller_log"] = [msg for _, msg in lines] if args.trace else []
        states = [re.search(r"safety=(\w+)", msg).group(1) for _, msg in lines if "safety=" in msg]
        result["safety_states"] = {s: states.count(s) for s in sorted(set(states))}
        result["controller_samples"] = [
            {k: v for k, v in (kv.split("=", 1) for kv in msg.split() if "=" in kv and not kv.startswith("cmd") and kv.split("=", 1)[0] in (
                "t", "range_m", "rel_speed_mps", "fuel_kg", "safety", "planner", "tracks", "track_ids_issued",
                "reacquired_total", "velocity_clamped_total", "track_age_max_s", "damage", "confirmed"))}
            for _, msg in lines]
    if args.trace:
        result["trace"] = [(round(t, 2), [round(x, 2) for x in p], [round(x, 4) for x in v]) for t, p, v in w.trace]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(1)
