"""V5 Phase 1 regression tests: radar identity, clock lock and tracker integrity.

The replay scenarios reproduce official Radar.py publication behaviour
(tests/radar_replay.py). They prove tracker logic offline only; they are not
evidence of live STPro performance.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experimental"))
sys.path.insert(0, str(TESTS_DIR))

from phobos_nav.config import CONFIG
from phobos_nav.radar import (
    RadarClockSynchronizer, RadarObservation, decode_radar, raw_record_key,
)
from phobos_nav.tracking import DebrisTrack, DebrisTracker, StateHistory
from radar_replay import ReplayScenario, run_replay, load_modules

MODULES = load_modules(ROOT / "experimental")
_CACHE = {}


def _replay(name, **overrides):
    if name not in _CACHE:
        _CACHE[name] = run_replay(ReplayScenario(**overrides), MODULES)
    return _CACHE[name]


def _assert_healthy_replay(result, scenario_n=500):
    log = [entry for entry in result["log"] if entry["ready"]]
    tracker = result["tracker"]
    evaluations = result["evaluations"]
    final = evaluations[-1]

    # All objects retained, no ghost or duplicate tracks, no replacement.
    assert max(entry["tracks"] for entry in log) <= scenario_n
    assert log[-1]["tracks"] == scenario_n
    assert final["objects_covered"] == scenario_n
    assert final["ghosts"] == 0
    assert final["duplicate_owners"] == 0
    assert max(e["ghosts"] for e in evaluations) == 0
    assert tracker.next_identifier - 1 == scenario_n, "tracks were replaced"
    assert sum(entry["evicted"] or 0 for entry in log) == 0
    assert tracker.cumulative["velocity_clamped"] == 0

    # Track ages follow the measured revisit period (no 6 s expiries).
    late = [entry for entry in log if entry["t"] > 6.0]
    assert max(entry["track_age_max"] for entry in late) < 5.5
    assert sum(entry["expired"] or 0 for entry in log) == 0

    # Physically plausible velocities: no track faster than the fastest true
    # object (live speed mix 0.5-10 m/s) plus 1 m/s of estimation error.
    speeds = [pair["speed"] for pair in final["pairs"]]
    true_max = float(np.max(np.linalg.norm(result["world"].v, axis=1)))
    assert max(speeds) < true_max + 1.0
    # Velocity errors must be consistent with the filter's own covariance
    # (NEES, 3 dof): mean ~3; 95th percentile below the chi-square 0.99
    # quantile (11.34). Absolute error depends on the number of revisits.
    confirmed = [p for e in evaluations[-5:] for p in e["pairs"] if p["confirmed"]]
    nees = np.asarray([p["nees_velocity"] for p in confirmed])
    assert np.mean(nees) < 6.0
    assert np.percentile(nees, 95) < 11.34
    return log


def test_replay_500_objects_retained_without_ghosts_or_false_confirmation():
    result = _replay("nominal")
    log = _assert_healthy_replay(result)
    scenario = ReplayScenario()
    # After ~6 sweeps, confirmed velocity error is sub-m/s in absolute terms.
    errors = [
        p["velocity_error"] for e in result["evaluations"][-5:]
        for p in e["pairs"] if p["confirmed"]
    ]
    assert np.percentile(errors, 95) < 0.6
    assert max(errors) < 1.5

    # Confirmation needs two genuine loop measurements of an object; the
    # earliest possible time is about one full sweep after radar start.
    earliest_possible = scenario.radar_start_s + 0.95 * scenario.num_debris / scenario.radar_hz
    assert all(entry["confirmed"] == 0 for entry in log if entry["t"] < earliest_possible)
    assert log[-1]["confirmed"] == scenario.num_debris

    # Cached rows never count as fresh: only the first ready poll may see
    # more than one publication's worth (N/10) of new rows.
    fresh = [entry["fresh"] or 0 for entry in log]
    assert fresh[0] == scenario.num_debris
    assert max(fresh[1:]) <= scenario.num_debris // 10
    # Every genuine record is processed at most once.
    genuine_upper_bound = scenario.num_debris + int(
        (scenario.duration_s - scenario.radar_start_s) * scenario.radar_hz
    )
    assert sum(fresh) <= genuine_upper_bound


def test_clock_offset_locks_and_stays_constant_under_jitter_and_overruns():
    result = _replay(
        "jitter", duration_s=15.0, overrun_probability=0.2, overrun_s=0.15,
        read_lag_max_s=0.030, seed=11,
    )
    log = [entry for entry in result["log"] if entry["ready"]]
    clock = result["clock"]
    assert clock.locked
    assert clock.resync_count == 0
    locked_offsets = {entry["offset"] for entry in log if entry["t"] > 6.0}
    assert len(locked_offsets) == 1, "offset changed after lock"
    # True offset 0.25 s; estimate within a few controller periods.
    assert abs(clock.offset_s - 0.25) < 0.05
    assert all(entry["latest_age"] < CONFIG.radar_stale_limit_s for entry in log)
    _assert_healthy_replay(result)


def test_missing_publications_do_not_create_ghosts_or_replace_tracks():
    result = _replay(
        "stall", duration_s=15.0, stall_windows=[(7.0, 8.6), (11.0, 11.7)], seed=5,
    )
    _assert_healthy_replay(result)


def test_reordered_records_do_not_change_identity():
    result = _replay("shuffled", duration_s=12.0, shuffle_records=True, seed=3)
    log = [entry for entry in result["log"] if entry["ready"]]
    assert result["tracker"].next_identifier - 1 == 500
    assert log[-1]["tracks"] == 500
    assert result["evaluations"][-1]["ghosts"] == 0


def test_original_200_object_radar_configuration():
    result = _replay("n200", num_debris=200, duration_s=10.0, seed=9)
    _assert_healthy_replay(result, scenario_n=200)


def test_tracker_update_cost_is_bounded():
    log = [entry for entry in _replay("nominal")["log"] if entry["ready"]]
    steady = [entry["tracker_ms"] for entry in log if entry["t"] > 3.0]
    # Offline budget on this machine; recorded for Phase 2 comparison.
    assert np.median(steady) < 5.0
    assert np.percentile(steady, 99) < 40.0


# ---------------------------------------------------------------------------
# Focused unit tests
# ---------------------------------------------------------------------------

def _record(relative, closest, range_rate, diameter, raw_time):
    return np.array([
        list(relative),
        [closest, range_rate, diameter],
        [raw_time, 0.0, 0.0],
    ], dtype=float)


def _decode(records, offset, now, shuttle_p=np.zeros(3), shuttle_v=np.zeros(3)):
    history = StateHistory()
    history.add(now - 6.0, shuttle_p - shuttle_v * 6.0, shuttle_v)
    history.add(now, shuttle_p, shuttle_v)
    return decode_radar(
        records, shuttle_p, shuttle_velocity=shuttle_v, current_timestamp=now,
        state_history=history, radar_time_offset_s=offset, config=CONFIG,
    )


def test_raw_key_is_independent_of_clock_offset():
    record = _record([120.0, -40.0, 15.0], 118.0, -1.2, 9.0, 3.217)
    a = _decode([record], 0.250, 5.0)[0]
    b = _decode([record], 0.262, 5.0)[0]
    assert a.raw_key == b.raw_key == raw_record_key(record)
    # The legacy key changed with the offset (the V4.3 defect).
    assert DebrisTracker._measurement_key(a.position, a.diameter, a.timestamp) != \
        DebrisTracker._measurement_key(b.position, b.diameter, b.timestamp)


def test_offset_jitter_replay_never_adds_hits_or_moves_tracks():
    rng = np.random.default_rng(1)
    records = [
        _record(rng.uniform(-300, 300, 3), 200.0, rng.uniform(-2, 2), 8.0, 1.0 + 0.01 * i)
        for i in range(50)
    ]
    tracker = DebrisTracker(CONFIG)
    tracker.update(_decode(records, 0.25, 2.0), 2.0, np.zeros(3), np.zeros(3))
    snapshot = {t.identifier: (t.position.copy(), t.velocity.copy(), t.hits) for t in tracker.tracks}
    for step in range(1, 120):
        now = 2.0 + step / 29.0
        offset = 0.25 + rng.uniform(-0.02, 0.02)   # fluctuating estimate
        tracker.update(_decode(records, offset, now), now, np.zeros(3), np.zeros(3))
        assert tracker.last_update_stats["fresh"] == 0
    assert len(tracker.tracks) == 50
    for track in tracker.tracks:
        position, velocity, hits = snapshot[track.identifier]
        assert track.hits == hits == 1 and not track.confirmed
        np.testing.assert_allclose(track.position, position)
        np.testing.assert_allclose(track.velocity, velocity)


def test_lost_seen_keys_replay_is_rejected_not_duplicated():
    rng = np.random.default_rng(2)
    records = [
        _record(rng.uniform(-300, 300, 3), 200.0, rng.uniform(-2, 2), 8.0, 1.0 + 0.01 * i)
        for i in range(40)
    ]
    tracker = DebrisTracker(CONFIG)
    tracker.update(_decode(records, 0.25, 2.0), 2.0, np.zeros(3), np.zeros(3))
    tracker._seen_measurements.clear()          # e.g. long RADAR_HOLD
    tracker.update(_decode(records, 0.25, 30.0 / 29.0 + 2.0), 3.03, np.zeros(3), np.zeros(3))
    stats = tracker.last_update_stats
    assert stats["created"] == 0 and stats["matched"] == 0
    assert stats["too_soon"] == 40
    assert len(tracker.tracks) == 40
    assert all(track.hits == 1 for track in tracker.tracks)


def test_tiny_dt_measurement_cannot_update_velocity():
    tracker = DebrisTracker(CONFIG)
    base = _record([200.0, 0.0, 0.0], 195.0, 0.5, 10.0, 4.000)
    tracker.update(_decode([base], 0.25, 4.5), 4.5, np.zeros(3), np.zeros(3))
    velocity = tracker.tracks[0].velocity.copy()
    # A distinct record (new raw values) only 20 ms later, offset by 1.5 m.
    near = _record([201.5, 0.4, 0.0], 196.5, 0.5, 10.0, 4.020)
    tracker.update(_decode([base, near], 0.25, 4.6), 4.6, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 0
    assert tracker.last_update_stats["created"] == 0
    assert tracker.last_update_stats["too_soon"] == 1
    np.testing.assert_allclose(tracker.tracks[0].velocity, velocity)
    assert tracker.tracks[0].hits == 1


def test_confirmation_requires_independent_revisits():
    tracker = DebrisTracker(CONFIG)
    true_p, true_v = np.array([300.0, 50.0, -20.0]), np.array([0.5, -1.0, 0.2])
    for k, t in enumerate([1.0, 5.17, 9.34]):
        rel = true_p + true_v * t
        radial = rel / np.linalg.norm(rel)
        record = _record(rel, np.linalg.norm(rel) - 5.0, float(true_v @ radial), 10.0, t - 0.25)
        tracker.update(_decode([record], 0.25, t + 0.1), t + 0.1, np.zeros(3), np.zeros(3))
        assert tracker.tracks[0].hits == k + 1
        assert tracker.tracks[0].confirmed == (k == 2)
    assert len(tracker.tracks) == 1
    assert np.linalg.norm(tracker.tracks[0].velocity - true_v) < 0.5


def test_corrupted_track_is_reacquired_not_duplicated():
    tracker = DebrisTracker(CONFIG)
    tracker.tracks.append(DebrisTrack(
        identifier=7, position=np.array([250.0, 0.0, 0.0]),
        velocity=np.array([0.0, 12.0, 0.0]),   # badly wrong tangential velocity
        diameter=10.0, timestamp=1.0, hits=3, confirmed=True,
        # A confirmed Kalman track is (over)confident: 1 m, 0.2 m/s sigma.
        covariance=np.diag([1.0, 1.0, 1.0, 0.04, 0.04, 0.04]),
    ))
    tracker.next_identifier = 8
    record = _record([250.5, 1.0, 0.0], 245.0, 0.0, 10.0, 4.92)   # true object barely moved
    tracker.update(_decode([record], 0.25, 5.3), 5.3, np.zeros(3), np.zeros(3))
    stats = tracker.last_update_stats
    assert stats["reacquired"] == 1 and stats["created"] == 0
    assert len(tracker.tracks) == 1 and tracker.tracks[0].identifier == 7
    assert tracker.tracks[0].hits == 1 and not tracker.tracks[0].confirmed


def test_capacity_eviction_is_counted_and_prefers_overdue_tentative():
    cfg = replace(CONFIG, max_tracks=3)
    tracker = DebrisTracker(cfg)
    confirmed = [
        DebrisTrack(identifier=i, position=np.array([100.0 * i, 500.0, 0.0]),
                    velocity=np.zeros(3), diameter=10.0, timestamp=4.0,
                    hits=3, confirmed=True)
        for i in (1, 2)
    ]
    overdue = DebrisTrack(identifier=3, position=np.array([0.0, -500.0, 0.0]),
                          velocity=np.zeros(3), diameter=10.0, timestamp=0.5)
    tracker.tracks = confirmed + [overdue]
    tracker.next_identifier = 4
    record = _record([-400.0, 0.0, 300.0], 495.0, 0.0, 10.0, 4.75)
    tracker.update(_decode([record], 0.25, 5.1), 5.1, np.zeros(3), np.zeros(3))
    stats = tracker.last_update_stats
    assert stats["evicted"] == 1 and stats["evicted_confirmed"] == 0
    assert {t.identifier for t in tracker.tracks} == {1, 2, 4}


def test_birth_velocity_excludes_shuttle_tangential_velocity():
    shuttle_v = np.array([0.0, 8.0, 0.0])          # moving across the line of sight
    record = _record([300.0, 0.0, 0.0], 295.0, -0.4, 10.0, 2.0)
    tracker = DebrisTracker(CONFIG)
    tracker.update(_decode([record], 0.25, 2.4, shuttle_v=shuttle_v), 2.4, np.zeros(3), shuttle_v)
    velocity = tracker.tracks[0].velocity
    # Radial absolute speed = shuttle radial (0) + range-rate (-0.4); no 8 m/s copy.
    np.testing.assert_allclose(velocity, [-0.4, 0.0, 0.0], atol=1e-9)


def test_clock_resynchronises_after_radar_restart_and_persistent_jump():
    sync = RadarClockSynchronizer(CONFIG)
    for k in range(1, 15):
        sync.update([_record([1, 0, 0], 1, 0, 1, 0.417 * k)], 0.25 + 0.417 * k + 0.01)
    assert sync.locked and abs(sync.offset_s - 0.25) < 0.02
    offset = sync.offset_s
    # Large single overrun: ignored.
    sync.update([_record([1, 0, 0], 1, 0, 1, 0.417 * 15)], 0.25 + 0.417 * 15 + 0.4)
    assert sync.offset_s == offset
    # Persistent +1 s jump (timebase change): re-acquired.
    for k in range(16, 30):
        sync.update([_record([1, 0, 0], 1, 0, 1, 0.417 * k)], 1.25 + 0.417 * k + 0.01)
    assert sync.resync_count == 1 and abs(sync.offset_s - 1.25) < 0.02
