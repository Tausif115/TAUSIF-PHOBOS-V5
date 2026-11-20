"""Offline full-tick benchmark at 60 Hz: Phase 1 flow vs current flow.

Emulates the controller's per-tick computation on the Radar.py-faithful
500-object replay feed:
  radar block  (clock sync, decode, tracker)
  threat scan  (nearest track, warning-envelope TTC)
  planner      (Controller cadence rules: critical / urgent / normal)
Phase 1 flow: decode every record, per-track Python threat loops and the
frozen Phase 1 planner, run synchronously when due.
Current flow: pre-decode filter, vectorized threats, two-stage planner run by
the real DeadlinePlanner (phobos_nav/scheduling.py) within the tick budget.

Both flows run in one process, interleaved tick by tick. Optional
--api-overhead-ms busy-waits at the start of every current-flow tick to
model STPro API time (state reads, frame transforms, radar array copy),
which this offline benchmark cannot measure. Numbers are machine-dependent
and are not live STPro timings.

Usage: python -B tests/benchmark_pipeline.py --site-packages DIR
           [--seconds 25] [--scenario sparse|dense|both] [--api-overhead-ms 0]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
PERIOD_S = 1.0 / 60.0


def busy_wait(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", action="append", default=[])
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--scenario", default="both", choices=("sparse", "dense", "both"))
    parser.add_argument("--api-overhead-ms", type=float, default=0.0)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--show-overruns", action="store_true")
    args = parser.parse_args()
    for path in args.site_packages:
        sys.path.append(path)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "experimental"))
    sys.path.insert(0, str(TESTS_DIR))

    from phobos_nav.config import CONFIG
    from phobos_nav import actuator_model, threats
    from phobos_nav.planning import PredictivePlanner
    from phobos_nav.radar import RadarClockSynchronizer, decode_radar
    from phobos_nav.scheduling import DeadlinePlanner
    from phobos_nav.tracking import DebrisTracker, StateHistory
    from radar_replay import ReplayScenario, ReplayWorld, generate_publications
    from reference import controller_threats_phase1 as reference_threats

    spec = importlib.util.spec_from_file_location(
        "phobos_nav._planning_phase1", TESTS_DIR / "reference" / "planning_phase1.py")
    reference_planning = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_planning
    spec.loader.exec_module(reference_planning)

    positive = np.array([0.193, 0.289, 0.386])
    negative = np.array([0.289, 0.289, 0.315])

    def limiter(command):
        command = np.asarray(command, dtype=float)
        if command.ndim == 2:
            return actuator_model.limit_acceleration_body_batch(
                command, positive, negative, safety_factor=CONFIG.authority_safety_factor)
        return actuator_model.limit_acceleration_body(
            command, positive, negative, safety_factor=CONFIG.authority_safety_factor)[0]

    scenarios = {
        "sparse": dict(),
        "dense": dict(field_half_extent_m=120.0, field_z_range_m=(-600.0, 100.0)),
    }
    names = ["sparse", "dense"] if args.scenario == "both" else [args.scenario]
    for scenario_name in names:
        scenario = ReplayScenario(duration_s=args.seconds, **scenarios[scenario_name])
        world = ReplayWorld(scenario)
        publications = generate_publications(world)
        publish_times = np.asarray([p[0] for p in publications])
        target = world.heading * 1300.0

        def make_flow(current):
            planner = (PredictivePlanner if current else reference_planning.PredictivePlanner)(CONFIG)
            return {
                "current": current, "planner": planner,
                "deadline": DeadlinePlanner(planner, CONFIG) if current else None,
                "tracker": DebrisTracker(CONFIG),
                "history": StateHistory(max_age_s=CONFIG.state_history_age_s),
                "clock": RadarClockSynchronizer(CONFIG),
                "metrics": {"clearance": float("inf"), "emergency": False},
                "bootstrap_ms": None,
                "ticks": [], "states": [],
            }

        flows = [make_flow(True)] if args.skip_phase1 else [make_flow(False), make_flow(True)]
        t, iteration = 0.05, 0
        while t < scenario.duration_s:
            loc, vel = world.shuttle_state(t)
            k = int(np.searchsorted(publish_times, t, side="right")) - 1
            raw = publications[k][1] if k >= 0 else []
            for flow in flows:
                tick_start = time.perf_counter()
                if flow["current"] and args.api_overhead_ms > 0.0:
                    busy_wait(args.api_overhead_ms / 1000.0)
                tracker = flow["tracker"]
                flow["history"].add(t, loc, vel)
                offset, ready, _ = flow["clock"].update(raw, t)
                if not ready:
                    continue
                bootstrap = len(tracker.tracks) < 0.9 * scenario.num_debris
                if flow["current"]:
                    records, cached = tracker.select_new_records(
                        raw, t, limit=CONFIG.radar_max_new_records_per_tick)
                    observations = decode_radar(records, loc, shuttle_velocity=vel, current_timestamp=t,
                                                state_history=flow["history"], radar_time_offset_s=offset, config=CONFIG)
                    tracks = tracker.update(observations, t, loc, vel, prefiltered_duplicates=cached)
                else:
                    observations = decode_radar(raw, loc, shuttle_velocity=vel, current_timestamp=t,
                                                state_history=flow["history"], radar_time_offset_s=offset, config=CONFIG)
                    tracks = tracker.update(observations, t, loc, vel)
                radar_done = time.perf_counter()
                if flow["current"]:
                    state = threats.track_state_arrays(tracks, t)
                    nearest = threats.nearest_track_distance(state, loc)
                    ttc = threats.threat_ttc_seconds(state, loc, vel, CONFIG)
                else:
                    nearest = reference_threats.nearest_track_distance(tracks, loc, t)
                    ttc = reference_threats.threat_ttc_seconds(tracks, loc, vel, t, CONFIG)
                threats_done = time.perf_counter()

                # Controller cadence rules (Controller.py, unchanged logic).
                cached_clearance = float(flow["metrics"].get("clearance", float("inf")))
                critical = bool(
                    flow["metrics"].get("emergency", False)
                    or cached_clearance <= CONFIG.minimum_safe_clearance_m + 2.0
                    or ttc <= CONFIG.critical_replan_ttc_s
                    or nearest <= CONFIG.critical_replan_distance_m
                )
                if critical:
                    due = iteration % CONFIG.critical_replan_every_n == 0
                elif (ttc <= CONFIG.urgent_replan_ttc_s or nearest <= CONFIG.urgent_replan_distance_m
                      or cached_clearance < CONFIG.replan_buffer_m + CONFIG.clearance_early_warning_m):
                    due = iteration % CONFIG.urgent_replan_every_n == 0
                else:
                    due = iteration % CONFIG.planner_update_every_n == 0

                planned = False
                if flow["current"]:
                    result = flow["deadline"].tick(
                        tick_start, due, critical, t,
                        lambda: dict(position=loc, velocity=vel, target_position=target,
                                     target_velocity=np.zeros(3), tracks=tracks, timestamp=t,
                                     max_accel=0.2, acceleration_limiter=limiter))
                    if result is not None:
                        flow["metrics"] = result[1]
                        planned = True
                elif due:
                    flow["metrics"] = flow["planner"].plan(
                        loc, vel, target, np.zeros(3), tracks, t,
                        max_accel=0.2, acceleration_limiter=limiter)[1]
                    planned = True
                end = time.perf_counter()
                total_ms = 1000.0 * (end - tick_start)
                if bootstrap:
                    flow.setdefault("bootstrap_ticks", []).append(total_ms)
                    flow["bootstrap_ms"] = max(flow["bootstrap_ticks"])
                    continue
                new_publication = k != flow.get("last_publication")
                flow["last_publication"] = k
                flow["ticks"].append((total_ms, 1000.0 * (radar_done - tick_start),
                                      1000.0 * (threats_done - radar_done), 1000.0 * (end - threats_done),
                                      planned, new_publication))
                flow["states"].append(flow["metrics"].get("safety_state", "STARTUP"))
            t += PERIOD_S
            iteration += 1

        print(f"\n=== {scenario_name.upper()} field, 500 debris, {args.seconds:.0f} s at 60 Hz"
              f"{f', assumed API overhead {args.api_overhead_ms:.1f} ms/tick (current flow)' if args.api_overhead_ms else ''}")
        for flow in flows:
            ticks = np.array([row[:4] for row in flow["ticks"]])
            planned = np.array([row[4] for row in flow["ticks"]])
            publication = np.array([bool(row[5]) for row in flow["ticks"]])
            total = ticks[:, 0]
            name = "CURRENT" if flow["current"] else "PHASE 1"

            def line(label, values):
                return (f"   {label:24s} n={values.size:5d}  p50 {np.median(values):6.2f}  "
                        f"p95 {np.percentile(values, 95):6.2f}  p99 {np.percentile(values, 99):6.2f}  "
                        f"max {values.max():6.2f} ms")

            print(f" {name}  tracks={len(flow['tracker'].tracks)}  bootstrap: {len(flow['bootstrap_ticks'])} tick(s), "
                  f"worst {flow['bootstrap_ms']:.1f} ms (startup only)")
            print(line("all ticks", total))
            print(line("planning ticks", total[planned]) if planned.any() else "   no planning ticks")
            print(line("radar-publication ticks", total[publication]))
            print(line("  radar part on those", ticks[publication, 1]))
            print(line("  planner part on those", ticks[publication, 3]))
            print(line("planner part (when run)", ticks[planned, 3]) if planned.any() else "")
            over = int(np.sum(total > 1000.0 * PERIOD_S))
            if args.show_overruns:
                for row in flow["ticks"]:
                    if row[0] > 1000.0 * PERIOD_S:
                        print(f"     overrun: total {row[0]:6.2f} radar {row[1]:6.2f} threats {row[2]:5.2f} "
                              f"planner {row[3]:6.2f} planned={row[4]} publication={row[5]}")
            print(f"   ticks over 16.67 ms: {over} of {total.size} ({100.0 * over / total.size:.2f} %)")
            if flow["current"]:
                stats = flow["deadline"].stats
                states = {s: flow["states"].count(s) for s in sorted(set(flow["states"]))}
                print(f"   planner: completed {stats['completed']}, deferred {stats['deferred']}, "
                      f"forced {stats['forced']}, discarded {stats['discarded']}; safety states {states}")


if __name__ == "__main__":
    main()
