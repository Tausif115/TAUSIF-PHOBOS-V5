"""Radar.py-faithful replay harness for offline tracker validation.

Reproduces the official STU4-Sim ``Systems/Radar.py`` behaviour:
  * init snapshot of all N objects with raw timestamp 0.0 (Radar.py 71-108);
  * one object refreshed per radar loop, ``idx = iteration % N`` (125-148);
  * radar timestamp = SimClock time since the radar loop start (117, 144-145);
  * the full array re-sorted by noisy range and republished whenever
    ``idx % (N // 10) == 0`` (113, 154-156): N/10 fresh rows plus cached rows
    that are bit-identical copies;
  * noise: position 2 m/axis, closest-surface range 2 m, range-rate 0.02 m/s,
    diameter 5 % (simconf Radar parameters; Radar.py 133-142).

The controller side runs the real phobos_nav pipeline in the same order as
Controller.py:431 and 476-502 (state history -> clock sync -> decode ->
tracker). Modules are injected so the same scenario can drive either the V5
package or the untouched V4.3 baseline for comparison.

This is a synthetic model. It is evidence about tracker logic, not proof of
live STPro behaviour.
"""

from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class ReplayScenario:
    num_debris: int = 500
    duration_s: float = 25.0
    radar_hz: float = 120.0
    # Extra radar loop time beyond the nominal sleep (Radar.py sleeps, then
    # works), as a fraction of the nominal period: uniform [0, value].
    radar_loop_overhead_fraction: float = 0.05
    controller_hz: float = 29.0
    radar_start_s: float = 0.25          # true clock offset (radar t0 in controller time)
    read_lag_max_s: float = 0.006        # radar read happens after t_now sampling
    transport_latency_s: float = 0.002
    overrun_probability: float = 0.05
    overrun_s: float = 0.12
    stall_windows: list = field(default_factory=list)  # [(start, end)] no controller polls
    shuffle_records: bool = False
    seed: int = 7
    position_sigma_m: float = 2.0
    closest_sigma_m: float = 2.0
    range_rate_sigma_mps: float = 0.02
    diameter_sigma_fraction: float = 0.05
    shuttle_accel_mps2: float = 0.15
    shuttle_max_speed_mps: float = 8.0
    shuttle_direction: tuple = (-0.15, 0.02, -0.99)
    debris_speed_range_mps: tuple = (0.5, 4.0)
    # Debris field around the initial path: lateral half-extent and z range.
    field_half_extent_m: float = 350.0
    field_z_range_m: tuple = (-700.0, 300.0)


class ReplayWorld:
    def __init__(self, scenario):
        self.s = scenario
        rng = np.random.default_rng(scenario.seed)
        n = scenario.num_debris
        # Debris field around the shuttle's initial path (similar extent to
        # SpawnDebris Spread_m = [300, 300, 500] around an offset point).
        half = scenario.field_half_extent_m
        self.p0 = np.column_stack((
            rng.uniform(-half, half, n),
            rng.uniform(-half, half, n),
            rng.uniform(*scenario.field_z_range_m, n),
        ))
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        speed = rng.uniform(*scenario.debris_speed_range_mps, n)
        self.v = direction * speed[:, None]
        self.diameter = rng.uniform(2.0, 20.0, n)
        self.noise = np.random.default_rng(100 + scenario.seed)  # Radar.py uses its own RNG
        heading = np.asarray(scenario.shuttle_direction, dtype=float)
        self.heading = heading / np.linalg.norm(heading)

    def debris_state(self, t):
        return self.p0 + self.v * t, self.v

    def shuttle_state(self, t):
        a = self.s.shuttle_accel_mps2
        t_burn = self.s.shuttle_max_speed_mps / a
        if t <= t_burn:
            distance = 0.5 * a * t * t
            speed = a * t
        else:
            distance = 0.5 * a * t_burn ** 2 + self.s.shuttle_max_speed_mps * (t - t_burn)
            speed = self.s.shuttle_max_speed_mps
        return self.heading * distance, self.heading * speed

    def measure(self, index, t, radar_time):
        positions, velocities = self.debris_state(t)
        shuttle_p, shuttle_v = self.shuttle_state(t)
        relative = positions[index] - shuttle_p
        relative_velocity = velocities[index] - shuttle_v
        distance = float(np.linalg.norm(relative))
        range_rate = float(np.dot(relative_velocity, relative / distance))
        closest = distance - 0.5 * self.diameter[index]
        s = self.s
        relative_noisy = relative + self.noise.normal(0.0, 1.0, 3) * s.position_sigma_m
        return np.array([
            [relative_noisy[0], relative_noisy[1], relative_noisy[2]],
            [
                closest + self.noise.normal(0.0, s.closest_sigma_m),
                range_rate + self.noise.normal(0.0, s.range_rate_sigma_mps),
                self.diameter[index] + self.noise.normal(
                    0.0, self.diameter[index] * s.diameter_sigma_fraction
                ),
            ],
            [radar_time, 0.0, 0.0],
        ])


def generate_publications(world):
    """Return [(publish_time, published_records, owner_index_per_record)]."""
    s = world.s
    n = s.num_debris
    sort_frequency = int(n / 10)
    rng = np.random.default_rng(1000 + s.seed)
    records = [world.measure(i, s.radar_start_s, 0.0) for i in range(n)]
    records = [r.copy() for r in records]
    for r in records:
        r[2, 0] = 0.0
    publications = []

    def publish(t):
        order = sorted(range(n), key=lambda i: np.linalg.norm(records[i][0, :]))
        if s.shuffle_records:
            order = list(rng.permutation(order))
        publications.append((t, [records[i].copy() for i in order], list(order)))

    publish(s.radar_start_s)
    t = s.radar_start_s
    iteration = 0
    period = 1.0 / s.radar_hz
    while t < s.duration_s + 1.0:
        t += period * (1.0 + rng.uniform(0.0, s.radar_loop_overhead_fraction))
        index = iteration % n
        records[index] = world.measure(index, t, t - s.radar_start_s)
        if index % sort_frequency == 0:
            publish(t)
        iteration += 1
    return publications


def run_replay(scenario, modules, evaluate_every_s=1.0, prefilter=True, new_record_limit=None):
    """Drive the injected phobos_nav modules with the replayed radar feed.

    prefilter=True mirrors the V5 Phase 2 Controller flow (decode only
    records not yet seen); False decodes every record (Phase 1 flow).
    Trackers without select_new_records (V4.3) always use the old flow.
    """
    world = ReplayWorld(scenario)
    publications = generate_publications(world)
    publish_times = np.asarray([p[0] for p in publications])
    cfg = modules["CONFIG"]
    tracker = modules["DebrisTracker"](cfg)
    history = modules["StateHistory"](max_age_s=cfg.state_history_age_s)
    clock = modules["RadarClockSynchronizer"](cfg)
    decode = modules["decode_radar"]
    rng = np.random.default_rng(2000 + scenario.seed)

    log = []
    evaluations = []
    t = 0.05
    next_eval = evaluate_every_s
    while t < scenario.duration_s:
        stalled = any(a <= t < b for a, b in scenario.stall_windows)
        if not stalled:
            read_time = t + rng.uniform(0.0, scenario.read_lag_max_s) - scenario.transport_latency_s
            k = int(np.searchsorted(publish_times, read_time, side="right")) - 1
            loc, vel = world.shuttle_state(t)
            history.add(t, loc, vel)
            raw = publications[k][1] if k >= 0 else []
            offset, ready, age = clock.update(raw, t)
            stats = {}
            decode_ms = tracker_ms = 0.0
            if ready:
                start = time.perf_counter()
                cached = 0
                records = raw
                if prefilter and hasattr(tracker, "select_new_records"):
                    records, cached = tracker.select_new_records(raw, t, limit=new_record_limit)
                observations = decode(
                    records, loc, shuttle_velocity=vel, current_timestamp=t,
                    state_history=history, radar_time_offset_s=offset, config=cfg,
                )
                middle = time.perf_counter()
                if cached:
                    tracker.update(observations, t, loc, vel, prefiltered_duplicates=cached)
                else:
                    tracker.update(observations, t, loc, vel)
                decode_ms = 1000.0 * (middle - start)
                tracker_ms = 1000.0 * (time.perf_counter() - middle)
                stats = dict(tracker.last_update_stats)
            ages = [t - float(x.timestamp) for x in tracker.tracks]
            log.append({
                "t": t, "ready": ready, "offset": offset, "latest_age": age,
                "publication": k, "tracks": len(tracker.tracks),
                "confirmed": sum(bool(x.confirmed) for x in tracker.tracks),
                "track_age_max": max(ages) if ages else 0.0,
                "decode_ms": decode_ms, "tracker_ms": tracker_ms,
                **{key: stats.get(key) for key in (
                    "fresh", "matched", "created", "duplicates", "too_soon",
                    "reacquired", "evicted", "expired",
                )},
            })
            if t >= next_eval and ready:
                evaluations.append(evaluate_tracks(world, tracker, t))
                next_eval += evaluate_every_s
        dt = 1.0 / scenario.controller_hz
        if rng.uniform() < scenario.overrun_probability:
            dt += scenario.overrun_s
        t += dt
    return {
        "world": world, "tracker": tracker, "clock": clock, "log": log,
        "evaluations": evaluations, "publications": publications,
    }


def evaluate_tracks(world, tracker, t, match_radius_m=20.0):
    """Associate tracks with truth by nearest true object at time t."""
    positions, velocities = world.debris_state(t)
    result = {"t": t, "tracks": len(tracker.tracks), "pairs": [], "ghosts": 0}
    owner_counts = {}
    for track in tracker.tracks:
        predicted = np.asarray(track.position_at(t), dtype=float)
        distances = np.linalg.norm(positions - predicted, axis=1)
        nearest = int(np.argmin(distances))
        # Allow for the track's age (tentative tracks extrapolate an unknown
        # tangential velocity); 20 m plus 5 m/s of age.
        allowance = match_radius_m + 5.0 * max(0.0, t - float(track.timestamp))
        if distances[nearest] > allowance:
            result["ghosts"] += 1
            continue
        owner_counts[nearest] = owner_counts.get(nearest, 0) + 1
        velocity_error_vector = np.asarray(track.velocity, dtype=float) - velocities[nearest]
        nees_velocity = float("nan")
        covariance = getattr(track, "covariance", None)
        if covariance is not None:
            p_vv = np.asarray(covariance, dtype=float)[3:, 3:]
            nees_velocity = float(velocity_error_vector @ np.linalg.solve(p_vv, velocity_error_vector))
        result["pairs"].append({
            "nees_velocity": nees_velocity,
            "id": track.identifier,
            "object": nearest,
            "confirmed": bool(track.confirmed),
            "hits": int(track.hits),
            "age": t - float(track.timestamp),
            "position_error": float(distances[nearest]),
            "velocity_error": float(np.linalg.norm(
                np.asarray(track.velocity, dtype=float) - velocities[nearest]
            )),
            "speed": float(np.linalg.norm(track.velocity)),
        })
    result["duplicate_owners"] = sum(1 for c in owner_counts.values() if c > 1)
    result["objects_covered"] = len(owner_counts)
    return result


def load_modules(package_parent):
    """Import phobos_nav from a specific 'experimental' directory."""
    import importlib
    import sys
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
    config = importlib.import_module("phobos_nav.config")
    radar = importlib.import_module("phobos_nav.radar")
    tracking = importlib.import_module("phobos_nav.tracking")
    return {
        "CONFIG": config.CONFIG,
        "decode_radar": radar.decode_radar,
        "RadarClockSynchronizer": radar.RadarClockSynchronizer,
        "DebrisTracker": tracking.DebrisTracker,
        "StateHistory": tracking.StateHistory,
    }
