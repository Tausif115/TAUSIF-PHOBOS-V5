"""V5 Phase 2: optimizations must not change navigation results.

Every optimized function is compared with a frozen copy of the committed
Phase 1 implementation (tests/reference/, commit 307cacb):
  * pre-decode filtering of cached radar records -> identical tracker state;
  * vectorized nearest-track and warning-envelope TTC -> identical values;
  * vectorized planner shortlist -> identical selection and arrays;
  * full planner decisions -> identical commands and metrics;
  * vectorized radar newest-timestamp scan -> identical result.
Timing assertions use generous, machine-tolerant bounds and speed ratios.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experimental"))
sys.path.insert(0, str(TESTS_DIR))

from phobos_nav.config import CONFIG
from phobos_nav import actuator_model
from phobos_nav.planning import PredictivePlanner
from phobos_nav.radar import RadarClockSynchronizer
from phobos_nav.threats import (
    track_state_arrays, nearest_track_distance, threat_ttc_seconds,
)
from phobos_nav.tracking import DebrisTrack
from radar_replay import ReplayScenario, run_replay, load_modules
from reference import controller_threats_phase1 as reference_threats


def _load_reference_planning():
    name = "phobos_nav._planning_phase1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TESTS_DIR / "reference" / "planning_phase1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE_PLANNING = _load_reference_planning()
POSITIVE_AUTHORITY = np.array([0.193, 0.289, 0.386])
NEGATIVE_AUTHORITY = np.array([0.289, 0.289, 0.315])


def _limiter(command):
    command = np.asarray(command, dtype=float)
    if command.ndim == 2:
        return actuator_model.limit_acceleration_body_batch(
            command, POSITIVE_AUTHORITY, NEGATIVE_AUTHORITY,
            safety_factor=CONFIG.authority_safety_factor,
        )
    return actuator_model.limit_acceleration_body(
        command, POSITIVE_AUTHORITY, NEGATIVE_AUTHORITY,
        safety_factor=CONFIG.authority_safety_factor,
    )[0]


def _random_tracks(rng, count, now, include_duck=False):
    tracks = []
    for index in range(count):
        position = rng.uniform(-600.0, 600.0, 3)
        velocity = rng.normal(0.0, 2.0, 3)
        if index % 17 == 0:
            velocity = np.zeros(3)                       # co-moving / static
        track = DebrisTrack(
            identifier=index + 1, position=position, velocity=velocity,
            diameter=float(rng.uniform(1.0, 25.0)),
            timestamp=now - float(rng.uniform(0.0, 5.5)),
            hits=int(rng.integers(1, 6)),
        )
        track.confirmed = track.hits >= 3
        tracks.append(track)
    if include_duck:
        p = rng.uniform(-100.0, 100.0, 3)
        v = rng.normal(0.0, 3.0, 3)
        tracks.append(SimpleNamespace(
            identifier=9001, position=p, velocity=v, diameter=12.0,
            timestamp=now - 1.0, hits=4, confirmed=True,
            position_at=lambda t, p=p, v=v, t0=now - 1.0: p + v * (t - t0),
        ))
        tracks.append(SimpleNamespace(        # no 'confirmed' attribute
            identifier=9002, position=p + 40.0, velocity=-v, diameter=6.0,
            timestamp=now - 0.5, hits=1,
        ))
    return tracks


# ---------------------------------------------------------------------------
# Radar pipeline: pre-decode filtering is a pure optimization
# ---------------------------------------------------------------------------

def test_predecode_filter_gives_identical_tracker_state():
    modules = load_modules(ROOT / "experimental")
    scenario = dict(duration_s=12.0, seed=21, overrun_probability=0.1)
    filtered = run_replay(ReplayScenario(**scenario), modules, prefilter=True)
    full = run_replay(ReplayScenario(**scenario), modules, prefilter=False)

    keys = ("fresh", "matched", "created", "duplicates", "too_soon", "reacquired", "evicted", "expired")
    for a, b in zip(filtered["log"], full["log"]):
        assert a["t"] == b["t"] and a["tracks"] == b["tracks"] and a["confirmed"] == b["confirmed"]
        for key in keys:
            assert (a[key] or 0) == (b[key] or 0), key
    tracks_a, tracks_b = filtered["tracker"].tracks, full["tracker"].tracks
    assert [t.identifier for t in tracks_a] == [t.identifier for t in tracks_b]
    for a, b in zip(tracks_a, tracks_b):
        assert np.array_equal(a.position, b.position)
        assert np.array_equal(a.velocity, b.velocity)
        assert np.array_equal(a.covariance, b.covariance)
        assert a.timestamp == b.timestamp and a.hits == b.hits

    # And the filter really skips the cached copies.
    steady = [e for e in filtered["log"] if e["ready"] and e["t"] > 3.0]
    assert np.median([e["decode_ms"] for e in steady]) < 0.25 * np.median(
        [e["decode_ms"] for e in full["log"] if e["ready"] and e["t"] > 3.0]
    )


def test_select_new_records_counts_cached_copies():
    modules = load_modules(ROOT / "experimental")
    result = run_replay(ReplayScenario(duration_s=6.0, seed=4), modules, prefilter=True)
    tracker = result["tracker"]
    last_poll = [entry for entry in result["log"] if entry["ready"]][-1]
    publication = result["publications"][last_poll["publication"]][1]
    new_records, cached = tracker.select_new_records(publication, last_poll["t"])
    assert cached + len(new_records) == len(publication) == 500
    assert len(new_records) <= 50


# ---------------------------------------------------------------------------
# Vectorized threat geometry == Phase 1 per-track loops
# ---------------------------------------------------------------------------

def test_vectorized_nearest_and_ttc_match_phase1_reference():
    rng = np.random.default_rng(12)
    for trial in range(150):
        now = 50.0 + trial
        tracks = _random_tracks(rng, int(rng.integers(0, 120)), now, include_duck=(trial % 5 == 0))
        if trial % 7 == 0:
            tracks = tracks[:0]
        shuttle_p = rng.uniform(-50.0, 50.0, 3)
        shuttle_v = rng.normal(0.0, 4.0, 3)
        state = track_state_arrays(tracks, now)

        # Phase 1 nearest-track code raised on tracks without position_at
        # (never produced by DebrisTracker); compare on its valid domain.
        # The vectorized version excludes such tracks instead of crashing.
        expected_nearest = reference_threats.nearest_track_distance(
            [t for t in tracks if hasattr(t, "position_at")], shuttle_p, now)
        got_nearest = nearest_track_distance(state, shuttle_p)
        assert np.isclose(got_nearest, expected_nearest, rtol=1e-12, atol=1e-9) or (
            np.isinf(got_nearest) and np.isinf(expected_nearest))

        expected_ttc = reference_threats.threat_ttc_seconds(tracks, shuttle_p, shuttle_v, now, CONFIG)
        got_ttc = threat_ttc_seconds(state, shuttle_p, shuttle_v, CONFIG)
        assert (np.isinf(got_ttc) and np.isinf(expected_ttc)) or np.isclose(
            got_ttc, expected_ttc, rtol=1e-12, atol=1e-9), (trial, got_ttc, expected_ttc)


def test_vectorized_ttc_detects_real_close_approach():
    now = 10.0
    track = DebrisTrack(identifier=1, position=np.array([200.0, 5.0, 0.0]),
                        velocity=np.array([-4.0, 0.0, 0.0]), diameter=10.0,
                        timestamp=now, hits=3, confirmed=True)
    state = track_state_arrays([track], now)
    ttc = threat_ttc_seconds(state, np.zeros(3), np.array([4.0, 0.0, 0.0]), CONFIG)
    assert abs(ttc - 25.0) < 1e-9          # closing at 8 m/s from 200 m
    assert nearest_track_distance(state, np.zeros(3)) == np.hypot(200.0, 5.0)


# ---------------------------------------------------------------------------
# Planner: vectorized shortlist and full decisions == Phase 1
# ---------------------------------------------------------------------------

def test_vectorized_planner_shortlist_matches_phase1_reference():
    rng = np.random.default_rng(34)
    new_planner = PredictivePlanner(CONFIG)
    reference_planner = REFERENCE_PLANNING.PredictivePlanner(CONFIG)
    for trial in range(60):
        now = 100.0 + trial
        tracks = _random_tracks(rng, int(rng.integers(1, 520)), now, include_duck=(trial % 4 == 0))
        position = rng.uniform(-20.0, 20.0, 3)
        velocity = rng.normal(0.0, 5.0, 3)
        got = new_planner._prepare_track_arrays(tracks, position, velocity, now)
        expected = reference_planner._prepare_track_arrays(tracks, position, velocity, now)
        assert got.keys() == expected.keys()
        assert np.array_equal(got["identifier"], expected["identifier"]), trial
        for key in expected:
            if expected[key].dtype == bool:
                assert np.array_equal(got[key], expected[key]), key
            else:
                np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, atol=1e-9, err_msg=key)


def test_planner_decisions_identical_to_phase1_over_a_sequence():
    rng = np.random.default_rng(56)
    new_planner = PredictivePlanner(CONFIG)
    reference_planner = REFERENCE_PLANNING.PredictivePlanner(CONFIG)
    target = np.array([-190.0, 25.0, -1290.0])
    position = np.zeros(3)
    velocity = np.zeros(3)
    tracks = _random_tracks(rng, 500, 0.0)
    for step in range(25):
        now = 0.5 * step
        got_accel, got = new_planner.plan(position, velocity, target, np.zeros(3), tracks, now,
                                          max_accel=0.2, acceleration_limiter=_limiter)
        exp_accel, exp = reference_planner.plan(position, velocity, target, np.zeros(3), tracks, now,
                                                max_accel=0.2, acceleration_limiter=_limiter)
        np.testing.assert_allclose(got_accel, exp_accel, rtol=1e-10, atol=1e-12)
        for key in ("safety_state", "selected_candidate", "closest_track_id", "candidate_count",
                    "safe_candidate_count", "preferred_candidate_count", "planning_track_count",
                    "avoidance_active", "emergency"):
            assert got[key] == exp[key], (step, key)
        for key in ("clearance", "nominal_clearance_m", "risk", "closest_time_s"):
            np.testing.assert_allclose(got[key], exp[key], rtol=1e-9, atol=1e-9, err_msg=key)
        velocity = velocity + got_accel * 0.5
        position = position + velocity * 0.5


# ---------------------------------------------------------------------------
# Radar clock: vectorized newest-timestamp scan == per-record loop
# ---------------------------------------------------------------------------

def test_vectorized_max_timestamp_matches_loop():
    rng = np.random.default_rng(78)

    def loop_reference(raw):
        values = []
        for item in raw:
            matrix = np.asarray(item, dtype=float)
            if matrix.shape == (3, 3):
                value = float(matrix[2, 0])
                if np.isfinite(value) and value >= 0.0:
                    values.append(value)
        return max(values) if values else None

    for trial in range(100):
        raw = [rng.normal(size=(3, 3)) for _ in range(int(rng.integers(0, 40)))]
        for item in raw:
            item[2, 0] = rng.choice([rng.uniform(0, 30), -1.0, np.nan, np.inf, 0.0])
        assert RadarClockSynchronizer._max_timestamp(raw) == loop_reference(raw)
    malformed = [np.zeros((3, 3)), np.zeros((2, 3))]
    malformed[0][2, 0] = 4.0
    assert RadarClockSynchronizer._max_timestamp(malformed) == 4.0


# ---------------------------------------------------------------------------
# Controller wiring (Controller.py needs spaceteams; checked statically)
# ---------------------------------------------------------------------------

def test_controller_uses_the_emulated_phase2_flow():
    import ast
    source = (ROOT / "Controller.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "threat_ttc_seconds" not in defined, "per-track loop must not return"
    for fragment in (
        "tracker.select_new_records(",
        "prefiltered_duplicates=cached_record_count",
        "decode_radar(\n                new_records,",
        "track_state_arrays(tracks, t_from_start)",
        "nearest_track_distance_from_state(",
        "threat_ttc_seconds_from_state(",
        "radar_decoded={radar_decoded_count}",
        "loop_overruns_total=",
        "limit=CONFIG.radar_max_new_records_per_tick",
        "deadline_planner = DeadlinePlanner(planner, CONFIG)",
        "completed_plan = deadline_planner.tick(",
        "deadline_planner.reset()",
        "plan_deferrals_total=",
    ):
        assert fragment in source, fragment


# ---------------------------------------------------------------------------
# Performance (offline, this machine; ratios are the robust measure)
# ---------------------------------------------------------------------------

def _median_time(function, repeats=15):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def test_threat_scan_and_shortlist_speedups():
    rng = np.random.default_rng(90)
    now = 30.0
    tracks = _random_tracks(rng, 500, now)
    shuttle_p, shuttle_v = np.zeros(3), np.array([0.0, 0.0, -5.0])
    new_planner = PredictivePlanner(CONFIG)
    reference_planner = REFERENCE_PLANNING.PredictivePlanner(CONFIG)

    def vectorized_scan():
        state = track_state_arrays(tracks, now)
        nearest_track_distance(state, shuttle_p)
        threat_ttc_seconds(state, shuttle_p, shuttle_v, CONFIG)

    def reference_scan():
        reference_threats.nearest_track_distance(tracks, shuttle_p, now)
        reference_threats.threat_ttc_seconds(tracks, shuttle_p, shuttle_v, now, CONFIG)

    scan_new, scan_ref = _median_time(vectorized_scan), _median_time(reference_scan)
    prep_new = _median_time(lambda: new_planner._prepare_track_arrays(tracks, shuttle_p, shuttle_v, now))
    prep_ref = _median_time(lambda: reference_planner._prepare_track_arrays(tracks, shuttle_p, shuttle_v, now))
    assert scan_new * 4.0 < scan_ref, (scan_new, scan_ref)
    assert prep_new * 4.0 < prep_ref, (prep_new, prep_ref)
    assert scan_new < 0.005 and prep_new < 0.010
