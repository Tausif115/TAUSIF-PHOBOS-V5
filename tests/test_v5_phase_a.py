"""V5 Phase A regression tests: tracker retuning from live run 0555d4ef.

Covers: provisional initial-snapshot tracks, revisit-p99 expiry (6 s track-age
criterion), the round-robin revisit association gate (fast-debris
mis-association), and multi-seed replays with the live debris speed mix
(0.5-10 m/s). Replays prove tracker logic offline only; they are not evidence
of live STPro performance.
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
from phobos_nav.tracking import DebrisTrack, DebrisTracker
from radar_replay import ReplayScenario, run_replay, load_modules
from test_v5_tracker_integrity import _assert_healthy_replay, _decode, _record

MODULES = load_modules(ROOT / "experimental")
PHASE1_BEHAVIOUR = replace(
    CONFIG, track_provisional_init_snapshot=False, track_expiry_mode="legacy",
    track_min_revisit_fraction=0.0,
)
LIVE_REVISIT_S = 4.71          # live run 0555d4ef, 500 objects


def _with_revisit(tracker, revisit_s=LIVE_REVISIT_S, samples=100):
    tracker._revisit_samples.clear()
    tracker._revisit_samples.extend([revisit_s] * samples)
    return tracker


def _snapshot_track(tracker, position=(300.0, 0.0, 0.0)):
    record = _record(position, float(np.linalg.norm(position)) - 5.0, 0.0, 10.0, 0.0)
    tracker.update(_decode([record], 0.25, 0.3), 0.3, np.zeros(3), np.zeros(3))
    return tracker.tracks[-1]


# --- provisional initial-snapshot tracks ------------------------------------

def test_snapshot_record_creates_provisional_track_loop_record_does_not():
    tracker = DebrisTracker(CONFIG)
    assert _snapshot_track(tracker).provisional
    record = _record([-200.0, 50.0, 0.0], 200.0, 0.0, 10.0, 1.7)
    tracker.update(_decode([record], 0.25, 2.0), 2.0, np.zeros(3), np.zeros(3))
    assert not tracker.tracks[-1].provisional


def test_inconsistent_snapshot_track_is_replaced_by_first_loop_measurement():
    # Live: the initial snapshot is inconsistent with loop measurements, so
    # the first loop measurement falls outside the chi-square gate. Phase 1
    # created a duplicate (the snapshot track then aged out as a ghost).
    for cfg, provisional in ((CONFIG, True), (PHASE1_BEHAVIOUR, False)):
        tracker = DebrisTracker(cfg)
        track = _snapshot_track(tracker)
        assert track.provisional is provisional
        record = _record([300.0, 33.0, 0.0], 296.8, 0.0, 10.0, 1.0)   # 33 m off, dt 1 s
        observations = _decode([record], 0.25, 1.3)
        tracker.update(observations, 1.3, np.zeros(3), np.zeros(3))
        stats = tracker.last_update_stats
        if provisional:
            assert stats["provisional_replaced"] == 1
            assert stats["created"] == 0 and stats["reacquired"] == 0
            assert len(tracker.tracks) == 1 and tracker.tracks[0].identifier == 1
            assert not tracker.tracks[0].provisional
            np.testing.assert_allclose(tracker.tracks[0].position, observations[0].position)
            assert tracker.cumulative["provisional_replaced"] == 1
        else:
            assert stats["created"] == 1 and len(tracker.tracks) == 2   # the Phase 1 defect


def test_snapshot_to_first_measurement_interval_is_not_a_revisit_sample():
    tracker = DebrisTracker(CONFIG)
    _snapshot_track(tracker)
    record = _record([300.0, 0.5, 0.0], 295.0, 0.0, 10.0, 1.0)      # consistent: KF update
    tracker.update(_decode([record], 0.25, 1.3), 1.3, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 1
    assert len(tracker._revisit_samples) == 0
    record = _record([300.0, 0.4, 0.0], 295.0, 0.0, 10.0, 5.71)     # genuine revisit
    tracker.update(_decode([record], 0.25, 6.0), 6.0, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 1
    assert list(tracker._revisit_samples) == [4.71]


def test_unreplaced_provisional_track_is_removed_once_first_sweep_is_complete():
    tracker = DebrisTracker(CONFIG)
    _snapshot_track(tracker)
    _with_revisit(tracker)                      # first sweep complete
    record = _record([-200.0, 50.0, 0.0], 200.0, 0.0, 10.0, 2.0)
    tracker.update(_decode([record], 0.25, 2.3), 2.3, np.zeros(3), np.zeros(3))
    assert [t.provisional for t in tracker.tracks] == [False]
    assert tracker.last_update_stats["expired"] == 1


def test_capacity_eviction_prefers_provisional_over_overdue_tentative():
    tracker = DebrisTracker(replace(CONFIG, max_tracks=3))
    tracker.tracks = [
        DebrisTrack(identifier=1, position=np.array([100.0, 500.0, 0.0]), velocity=np.zeros(3),
                    diameter=10.0, timestamp=4.0, hits=3, confirmed=True),
        DebrisTrack(identifier=2, position=np.array([0.0, -500.0, 0.0]), velocity=np.zeros(3),
                    diameter=10.0, timestamp=0.2),                       # overdue tentative
        DebrisTrack(identifier=3, position=np.array([500.0, 0.0, 300.0]), velocity=np.zeros(3),
                    diameter=10.0, timestamp=4.9, provisional=True),
    ]
    tracker.next_identifier = 4
    record = _record([-400.0, 0.0, 300.0], 495.0, 0.0, 10.0, 4.85)
    tracker.update(_decode([record], 0.25, 5.1), 5.1, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["evicted"] == 1
    assert {t.identifier for t in tracker.tracks} == {1, 2, 4}


# --- expiry / 6 s track-age criterion ---------------------------------------

def test_expiry_follows_measured_revisit_distribution():
    tracker = DebrisTracker(CONFIG)
    assert tracker.expiry_age_s() == CONFIG.track_max_age_s           # startup: 6.0 s
    _with_revisit(tracker, samples=10)
    assert tracker.expiry_age_s() == CONFIG.track_max_age_s           # too few samples
    _with_revisit(tracker)
    # Latest valid refresh: sweep + one publication interval (sweep/10) + poll.
    assert abs(tracker.expiry_age_s() - (1.1 * 4.71 + 0.60)) < 1e-9   # 5.781 s
    assert 1.1 * LIVE_REVISIT_S < tracker.expiry_age_s() < 6.0
    _with_revisit(tracker, revisit_s=8.0)                             # slow radar
    assert tracker.expiry_age_s() == CONFIG.track_max_age_cap_s       # coverage first, capped
    legacy = _with_revisit(DebrisTracker(PHASE1_BEHAVIOUR))
    assert abs(legacy.expiry_age_s() - 1.3 * 4.71) < 1e-9             # Phase 1: 6.123 s


def test_radar_hitches_do_not_expire_valid_tracks():
    result = run_replay(ReplayScenario(
        duration_s=30.0, seed=3, radar_pauses=[(14.0, 1.0), (22.0, 1.0)],
    ), MODULES)
    log = [entry for entry in result["log"] if entry["ready"]]
    assert sum(entry["expired"] or 0 for entry in log) == 0
    assert result["tracker"].next_identifier - 1 == 500
    assert result["evaluations"][-1]["objects_covered"] == 500


# --- round-robin revisit association gate -----------------------------------

def _gate_tracker(provisional=False):
    tracker = _with_revisit(DebrisTracker(CONFIG))
    tracker.tracks = [DebrisTrack(
        identifier=1, position=np.array([300.0, 0.0, 0.0]), velocity=np.zeros(3),
        diameter=10.0, timestamp=10.0, hits=3, confirmed=not provisional,
        covariance=np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0]), provisional=provisional,
    )]
    tracker.next_identifier = 2
    tracker.last_timestamp = 10.0
    return tracker


def test_track_is_not_updated_sooner_than_round_robin_revisit_allows():
    record = _record([300.0, 3.0, 0.0], 295.0, 0.0, 10.0, 11.75)    # 2.0 s after update
    tracker = _gate_tracker()
    tracker.update(_decode([record], 0.25, 12.1), 12.1, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 0
    original = [t for t in tracker.tracks if t.identifier == 1][0]
    assert original.timestamp == 10.0 and original.hits == 3
    np.testing.assert_allclose(original.position, [300.0, 0.0, 0.0])
    # The same measurement one sweep after the update is a genuine revisit.
    record = _record([300.0, 3.0, 0.0], 295.0, 0.0, 10.0, 14.46)
    tracker = _gate_tracker()
    tracker.update(_decode([record], 0.25, 14.8), 14.8, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 1 and tracker.tracks[0].hits == 4


def test_round_robin_gate_exempts_provisional_and_can_be_disabled():
    record = _record([300.0, 3.0, 0.0], 295.0, 0.0, 10.0, 11.75)
    tracker = _gate_tracker(provisional=True)
    tracker.update(_decode([record], 0.25, 12.1), 12.1, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 1
    tracker = _gate_tracker()
    tracker.config = replace(CONFIG, track_min_revisit_fraction=0.0)
    tracker.update(_decode([record], 0.25, 12.1), 12.1, np.zeros(3), np.zeros(3))
    assert tracker.last_update_stats["matched"] == 1


def test_fast_debris_crossing_does_not_steal_measurements_seed_100():
    # Phase 1 on this replay: an 8.3 m/s object's revisit was assigned to a
    # different object's track updated 2.47 s earlier (track corrupted, own
    # track orphaned, expired, re-created). Seed 100 is the official default.
    for seed in (100, 9):
        result = run_replay(ReplayScenario(duration_s=30.0, seed=seed), MODULES)
        _assert_healthy_replay(result)
        late = [entry for entry in result["log"] if entry["ready"] and entry["t"] > 6.0]
        assert max(entry["track_age_max"] for entry in late) < 6.0
        errors = [p["velocity_error"] for p in result["evaluations"][-1]["pairs"]]
        assert max(errors) < 1.5


# --- startup snapshot conditions (live run 0555d4ef) --------------------------

def test_zero_velocity_snapshot_keeps_500_tracks_without_cap_or_eviction():
    scenario = ReplayScenario(duration_s=20.0, seed=41, init_snapshot_mode="zero_velocity")
    result = run_replay(scenario, MODULES)
    log = [entry for entry in result["log"] if entry["ready"]]
    tracker = result["tracker"]
    assert max(entry["tracks"] for entry in log) <= 505
    assert sum(entry["evicted"] or 0 for entry in log) == 0
    assert tracker.next_identifier - 1 <= 515
    # Slow objects' snapshot tracks pass the gate and are KF-updated; the
    # rest (seed 41: 232 of 500) are replaced instead of duplicated.
    assert tracker.cumulative["provisional_replaced"] >= 150
    assert result["evaluations"][-1]["objects_covered"] == 500
    assert not any(t.provisional for t in tracker.tracks)
    # Phase 1 behaviour reproduces the live failure: the 550 cap is reached.
    legacy = run_replay(scenario, dict(MODULES, CONFIG=PHASE1_BEHAVIOUR))
    assert max(entry["tracks"] for entry in legacy["log"] if entry["ready"]) == CONFIG.max_tracks


def test_stale_snapshot_evicts_only_placeholders_and_recovers_full_coverage():
    result = run_replay(ReplayScenario(duration_s=20.0, seed=42, init_snapshot_mode="stale"), MODULES)
    log = [entry for entry in result["log"] if entry["ready"]]
    tracker = result["tracker"]
    assert tracker.cumulative["evicted_confirmed"] == 0
    assert not any(t.provisional for t in tracker.tracks)
    final = result["evaluations"][-1]
    assert final["objects_covered"] == 500 and final["ghosts"] == 0
    assert final["duplicate_owners"] == 0 and log[-1]["tracks"] <= 505


# --- multi-seed generalisation (final SpawnDebris seed is unknown) -------------

def test_multi_seed_live_speed_replays_are_healthy():
    for seed in (1, 7, 11):
        result = run_replay(ReplayScenario(duration_s=20.0, seed=seed), MODULES)
        _assert_healthy_replay(result)
        assert result["tracker"].cumulative["reacquired"] <= 2
