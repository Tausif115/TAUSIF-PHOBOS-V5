"""V5 Phase 2b: planner/tracker optimizations and deadline-aware execution.

Correctness tests compare against frozen Phase 1 code (tests/reference/,
commit 307cacb). Deadline tests use the real DeadlinePlanner on the
Radar.py-faithful replay; timing assertions are offline, machine-dependent
and use percentiles (this desktop shows OS stalls of 3-5x on any workload).
"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experimental"))
sys.path.insert(0, str(TESTS_DIR))

from phobos_nav.config import CONFIG
from phobos_nav import actuator_model, threats
from phobos_nav.geometry import unit
from phobos_nav.planning import PredictivePlanner
from phobos_nav.scheduling import DeadlinePlanner, StageBudget
from phobos_nav.tracking import DebrisTrack, StateHistory
from radar_replay import ReplayScenario, ReplayWorld, generate_publications, run_replay, load_modules
from test_v5_performance_equivalence import REFERENCE_PLANNING, _limiter, _random_tracks


def _load_reference(name, filename):
    full = "phobos_nav." + name
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, TESTS_DIR / "reference" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


REFERENCE_TRACKING = _load_reference("_tracking_phase1", "tracking_phase1.py")
REFERENCE_ACTUATOR = _load_reference("_actuator_model_phase1", "actuator_model_phase1.py")
DENSE_FIELD = dict(field_half_extent_m=120.0, field_z_range_m=(-600.0, 100.0))


def _dense_tracks(rng, count, now):
    tracks = []
    for index in range(count):
        track = DebrisTrack(
            identifier=index + 1,
            position=rng.uniform(-150.0, 150.0, 3) + np.array([0.0, 0.0, -200.0]),
            velocity=rng.normal(0.0, 2.0, 3), diameter=float(rng.uniform(2.0, 25.0)),
            timestamp=now - float(rng.uniform(0.0, 5.0)), hits=int(rng.integers(1, 5)),
        )
        track.confirmed = track.hits >= 3
        tracks.append(track)
    return tracks


def _candidate_inputs(planner, tracks, position, velocity, target):
    arrays = planner._prepare_track_arrays(tracks, position, velocity, 0.0)
    specs = planner._candidate_specs(unit(target - position), arrays, position, velocity)
    return (arrays, np.array([s[1] for s in specs]), np.array([s[2] for s in specs]),
            np.array([s[3] for s in specs]))


# ---------------------------------------------------------------------------
# Exact bounded clearance evaluation
# ---------------------------------------------------------------------------

def test_bounded_clearance_matches_phase1_full_evaluation():
    rng = np.random.default_rng(101)
    target = np.array([-190.0, 25.0, -1290.0])
    for trial in range(45):
        tracks = (_dense_tracks if trial % 3 == 0 else _random_tracks)(rng, int(rng.integers(5, 520)), 0.0)
        position, velocity = rng.uniform(-30.0, 30.0, 3), rng.normal(0.0, 5.0, 3)
        new, reference = PredictivePlanner(CONFIG), REFERENCE_PLANNING.PredictivePlanner(CONFIG)
        arrays, biases, scales, holds = _candidate_inputs(new, tracks, position, velocity, target)
        got = new._rollout_candidates(position, velocity, target, np.zeros(3), 0.0, biases, scales, holds,
                                      0.2, track_arrays=arrays, acceleration_limiter=_limiter)
        expected = reference._rollout_candidates(position, velocity, target, np.zeros(3), 0.0, biases, scales,
                                                 holds, 0.2, track_arrays=arrays, acceleration_limiter=_limiter)
        assert np.array_equal(got["closest_track_id"], expected["closest_track_id"]), trial
        for key in ("risk", "clearance", "closest_time_s", "distance", "terminal_relative_speed"):
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-11, atol=1e-9, err_msg=key)
        np.testing.assert_array_equal(got["terminal_position"], expected["terminal_position"])


def test_bounded_clearance_result_is_independent_of_chunking():
    rng = np.random.default_rng(202)
    target = np.array([-190.0, 25.0, -1290.0])
    tracks = _dense_tracks(rng, 400, 0.0)
    position, velocity = np.zeros(3), np.array([0.0, 0.0, -4.0])
    results = []
    for chunk in (1, 3, 4, 8, 40):
        planner = PredictivePlanner(replace(CONFIG, planner_bound_chunk_steps=chunk))
        arrays, biases, scales, holds = _candidate_inputs(planner, tracks, position, velocity, target)
        results.append(planner._rollout_candidates(
            position, velocity, target, np.zeros(3), 0.0, biases, scales, holds, 0.2,
            track_arrays=arrays, acceleration_limiter=_limiter))
    for other in results[1:]:
        for key in ("risk", "clearance", "closest_time_s", "closest_track_id"):
            np.testing.assert_allclose(other[key], results[0][key], rtol=1e-12, atol=1e-10, err_msg=key)


# ---------------------------------------------------------------------------
# Two-stage planning == single-call planning
# ---------------------------------------------------------------------------

def test_two_stage_planning_equals_plan_over_a_sequence():
    rng = np.random.default_rng(303)
    tracks = _dense_tracks(rng, 500, 0.0)
    target = np.array([-190.0, 25.0, -1290.0])
    single, staged = PredictivePlanner(CONFIG), PredictivePlanner(CONFIG)
    position, velocity = np.zeros(3), np.zeros(3)
    for step in range(20):
        now = 0.5 * step
        a1, m1 = single.plan(position, velocity, target, np.zeros(3), tracks, now,
                             max_accel=0.2, acceleration_limiter=_limiter)
        job = staged.plan_stage_one(position, velocity, target, np.zeros(3), tracks, now,
                                    max_accel=0.2, acceleration_limiter=_limiter)
        a2, m2 = staged.plan_stage_two(job)
        np.testing.assert_array_equal(a1, a2)
        for key in ("safety_state", "selected_candidate", "closest_track_id", "clearance", "risk"):
            assert m1[key] == m2[key], key
        np.testing.assert_array_equal(single.last_bias, staged.last_bias)
        velocity = velocity + a1 * 0.5
        position = position + velocity * 0.5


# ---------------------------------------------------------------------------
# Bit-identical helpers vs Phase 1
# ---------------------------------------------------------------------------

def test_batch_limiter_is_bit_identical_to_phase1():
    rng = np.random.default_rng(404)
    for trial in range(1500):
        commands = rng.normal(0.0, rng.choice([1e-12, 1e-3, 0.2, 5.0]), (int(rng.integers(1, 40)), 3))
        commands[rng.random(len(commands)) < 0.15] = 0.0
        positive, negative = rng.uniform(0.01, 0.5, 3), rng.uniform(0.01, 0.5, 3)
        if trial % 50 == 0:
            positive[int(rng.integers(0, 3))] = 0.0          # general (no-authority) path
        factor = float(rng.uniform(0.1, 1.0))
        np.testing.assert_array_equal(
            actuator_model.limit_acceleration_body_batch(commands, positive, negative, safety_factor=factor),
            REFERENCE_ACTUATOR.limit_acceleration_body_batch(commands, positive, negative, safety_factor=factor))


def test_state_history_bisect_lookup_is_bit_identical_to_phase1():
    rng = np.random.default_rng(505)
    for trial in range(60):
        new, reference = StateHistory(max_age_s=8.0), REFERENCE_TRACKING.StateHistory(max_age_s=8.0)
        t = float(rng.uniform(0.0, 5.0))
        for _ in range(int(rng.integers(1, 300))):
            t += float(rng.choice([0.0, 1e-10, rng.uniform(0.005, 0.12)]))
            p, v = rng.normal(size=3), rng.normal(size=3)
            new.add(t, p, v)
            reference.add(t, p, v)
        times = [sample[0] for sample in reference.samples]
        for query in list(rng.uniform(times[0] - 1.0, times[-1] + 1.0, 20)) + times[::7]:
            a, b = new.sample(query), reference.sample(query)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])


def test_tracker_matches_phase1_reference_on_replays():
    # Phase A behaviour (provisional snapshot tracks, revisit-p99 expiry,
    # round-robin revisit gate) is switched off here: this test proves the
    # Phase 2 performance rewrite is equivalent to Phase 1. Phase A is tested
    # in test_v5_phase_a.py.
    modules = load_modules(ROOT / "experimental")
    modules = dict(modules, CONFIG=replace(
        modules["CONFIG"], track_provisional_init_snapshot=False, track_expiry_mode="legacy",
        track_min_revisit_fraction=0.0,
    ))
    reference = dict(modules, DebrisTracker=REFERENCE_TRACKING.DebrisTracker,
                     StateHistory=REFERENCE_TRACKING.StateHistory)
    keys = ("fresh", "matched", "created", "duplicates", "too_soon", "reacquired", "evicted", "expired")
    for overrides in (dict(duration_s=10.0, seed=31), dict(duration_s=10.0, seed=32, **DENSE_FIELD)):
        got = run_replay(ReplayScenario(**overrides), modules)
        expected = run_replay(ReplayScenario(**overrides), reference)
        for a, b in zip(got["log"], expected["log"]):
            assert a["tracks"] == b["tracks"] and a["confirmed"] == b["confirmed"]
            for key in keys:
                assert (a[key] or 0) == (b[key] or 0), key
        for a, b in zip(got["tracker"].tracks, expected["tracker"].tracks):
            assert a.identifier == b.identifier and a.hits == b.hits
            np.testing.assert_allclose(a.position, b.position, atol=1e-9)
            np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-9)
            np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-9)


def test_ingestion_cap_spreads_bootstrap_without_losing_objects():
    modules = load_modules(ROOT / "experimental")
    result = run_replay(ReplayScenario(duration_s=10.0, seed=33), modules, new_record_limit=100)
    log = [entry for entry in result["log"] if entry["ready"]]
    assert max(entry["fresh"] or 0 for entry in log) <= 100
    assert log[-1]["tracks"] == 500
    assert result["tracker"].next_identifier - 1 == 500
    assert result["evaluations"][-1]["ghosts"] == 0


# ---------------------------------------------------------------------------
# DeadlinePlanner logic (fake clock, fake planner)
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _FakePlanner:
    def __init__(self, clock, stage1_s, stage2_s):
        self.clock, self.stage1_s, self.stage2_s = clock, stage1_s, stage2_s
        self.started, self.finished = 0, 0

    def plan_stage_one(self, timestamp, **_):
        self.clock.now += self.stage1_s
        self.started += 1
        return {"timestamp": timestamp}

    def plan_stage_two(self, job):
        self.clock.now += self.stage2_s
        self.finished += 1
        return np.ones(3), {"safety_state": "CLEAR"}


def _executor(stage1_s=0.004, stage2_s=0.004, **config_overrides):
    clock = _FakeClock()
    planner = _FakePlanner(clock, stage1_s, stage2_s)
    cfg = replace(CONFIG, **config_overrides)
    return DeadlinePlanner(planner, cfg, clock=clock), planner, clock


def _tick(executor, clock, elapsed_before_s, due, now_s):
    tick_start = clock.now
    clock.now += elapsed_before_s
    return executor.tick(tick_start, due, False, now_s, lambda: {"timestamp": now_s})


def test_deadline_planner_runs_both_stages_when_they_fit():
    executor, planner, clock = _executor()
    assert _tick(executor, clock, 0.002, True, 0.0) is not None
    assert planner.started == planner.finished == 1


def test_deadline_planner_defers_then_forces_after_max_wait():
    executor, planner, clock = _executor(stage1_s=0.006, stage2_s=0.006)
    # 9 ms already used: stage one (6 ms) would exceed the 14 ms budget.
    assert _tick(executor, clock, 0.009, True, 0.0) is None
    assert planner.started == 0 and executor.stats["deferred"] == 1
    assert _tick(executor, clock, 0.009, False, 1 / 60) is None       # still waiting, request latched
    assert planner.started == 0
    # Third tick: waited max_wait (2) ticks -> forced start; stage two then deferred.
    assert _tick(executor, clock, 0.009, False, 2 / 60) is None
    assert planner.started == 1 and executor.stats["forced"] == 1
    assert _tick(executor, clock, 0.002, False, 3 / 60) is not None    # stage two fits now
    assert planner.finished == 1


def test_deadline_planner_never_drops_requests_and_discards_stale_jobs():
    executor, planner, clock = _executor(stage1_s=0.004, stage2_s=0.012)
    for _ in range(StageBudget.MIN_SAMPLES):                           # learned stage-2 cost
        executor.budget.record(2, 0.012)
    assert _tick(executor, clock, 0.002, True, 0.0) is None            # stage one only
    assert planner.started == 1 and planner.finished == 0
    # Job becomes older than planner_max_job_age_s (0.1 s): discarded, re-requested.
    result = _tick(executor, clock, 0.001, False, 0.2)
    assert executor.stats["discarded"] == 1
    assert planner.started == 2 or result is not None


def test_stage_estimate_is_robust_to_one_outlier():
    budget = StageBudget((0.006, 0.006), history=20)
    for _ in range(19):
        budget.record(1, 0.005)
    budget.record(1, 0.050)                                            # one OS stall
    assert budget.estimate(1) < 0.010


# ---------------------------------------------------------------------------
# Deadline performance on the 500-debris replay (offline)
# ---------------------------------------------------------------------------

def test_full_tick_meets_60hz_budget_dense_500_debris():
    from phobos_nav.radar import RadarClockSynchronizer, decode_radar
    from phobos_nav.tracking import DebrisTracker
    scenario = ReplayScenario(duration_s=8.0, seed=34, **DENSE_FIELD)
    world = ReplayWorld(scenario)
    publications = generate_publications(world)
    publish_times = np.asarray([p[0] for p in publications])
    tracker, history = DebrisTracker(CONFIG), StateHistory(max_age_s=CONFIG.state_history_age_s)
    clock_sync = RadarClockSynchronizer(CONFIG)
    executor = DeadlinePlanner(PredictivePlanner(CONFIG), CONFIG)
    target = world.heading * 1300.0
    ticks, t, iteration = [], 0.05, 0
    while t < scenario.duration_s:
        start = time.perf_counter()
        loc, vel = world.shuttle_state(t)
        history.add(t, loc, vel)
        k = int(np.searchsorted(publish_times, t, side="right")) - 1
        raw = publications[k][1] if k >= 0 else []
        offset, ready, _ = clock_sync.update(raw, t)
        if ready:
            records, cached = tracker.select_new_records(raw, t, limit=CONFIG.radar_max_new_records_per_tick)
            observations = decode_radar(records, loc, shuttle_velocity=vel, current_timestamp=t,
                                        state_history=history, radar_time_offset_s=offset, config=CONFIG)
            tracks = tracker.update(observations, t, loc, vel, prefiltered_duplicates=cached)
            state = threats.track_state_arrays(tracks, t)
            threats.nearest_track_distance(state, loc)
            threats.threat_ttc_seconds(state, loc, vel, CONFIG)
            executor.tick(start, iteration % 2 == 0, True, t, lambda: dict(
                position=loc, velocity=vel, target_position=target, target_velocity=np.zeros(3),
                tracks=tracks, timestamp=t, max_accel=0.2, acceleration_limiter=_limiter))
            if len(tracker.tracks) >= 450:                             # past bootstrap
                ticks.append(1000.0 * (time.perf_counter() - start))
        t += 1.0 / 60.0
        iteration += 1
    ticks = np.asarray(ticks)
    assert executor.stats["completed"] > 100
    assert np.percentile(ticks, 95) < 16.67, np.percentile(ticks, 95)
    assert np.median(ticks) < 12.0
