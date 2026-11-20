"""Regression tests for the separate 500-debris validation candidate."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "experimental"
sys.path.insert(0, str(PACKAGE_ROOT))

from phobos_nav.config import CONFIG
from phobos_nav.planning import PredictivePlanner
from phobos_nav.radar import RadarObservation
from phobos_nav.tracking import DebrisTrack, DebrisTracker


def _make_observation(position, velocity, timestamp, diameter):
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    radial = position / max(float(np.linalg.norm(position)), 1.0e-9)
    return RadarObservation(
        position=position.copy(),
        diameter=float(diameter),
        range_rate=float(np.dot(velocity, radial)),
        timestamp=float(timestamp),
        relative_position=position.copy(),
        shuttle_velocity=np.zeros(3, dtype=float),
    )


SIMCONF_RELATIVE = Path("STU4-Sim") / "Sims" / "COMPETITION_SIM_500_RADAR_VALIDATION.simconf"


def find_validation_simconf():
    """Locate the read-only validation simconf in any supported layout.

    V5 layout:        V5-DEVELOPMENT/_reference/STU4-Sim/Sims/...
    Baseline layout:  V43-BASELINE/STU4-Sim/Sims/... (sibling of V5-DEVELOPMENT)
    V4.3 package:     <package parent>/STU4-Sim/Sims/...
    """
    candidates = [
        ROOT / "_reference" / SIMCONF_RELATIVE,
        ROOT.parent / SIMCONF_RELATIVE,
        ROOT.parent / "V43-BASELINE" / SIMCONF_RELATIVE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "COMPETITION_SIM_500_RADAR_VALIDATION.simconf not found; searched: "
        + ", ".join(str(item) for item in candidates)
    )


def test_validation_simconf_radar_and_spawner_both_use_500():
    simconf = find_validation_simconf()
    data = json.loads(simconf.read_text(encoding="utf-8-sig"))
    systems = {
        item["Nametag"]: item["Inst_Parameters"]
        for item in data["SystemManager"]["Systems"]
    }
    assert systems["Radar"]["NumDebris"] == ["int32", 500]
    assert systems["SpawnDebris"]["NumDebris"] == ["int32", 500]
    assert systems["Radar"]["LoopFreqHz"] == 120.0


def test_tracker_retains_all_500_round_robin_debris_tracks():
    cfg = CONFIG
    assert cfg.radar_max_objects == 500
    assert cfg.max_tracks >= 500
    assert cfg.track_max_age_s > (500 / 120.0)

    count = 500
    index = np.arange(count)
    positions = np.column_stack((
        (index % 10) * 120.0 - 540.0,
        ((index // 10) % 10) * 120.0 - 540.0,
        (index // 100) * 180.0 - 360.0,
    ))
    rng = np.random.default_rng(500)
    velocities = rng.uniform(-0.08, 0.08, size=(count, 3))
    diameters = 12.0 + (index % 7) * 2.0

    observations = [
        _make_observation(positions[i], velocities[i], 1.0, diameters[i])
        for i in range(count)
    ]
    tracker = DebrisTracker(cfg)
    tracker.update(observations, 1.0, np.zeros(3), np.zeros(3))
    assert len(tracker.tracks) == 500

    # Radar.py updates one debris per loop and republishes after each 50 updates.
    # At 120 Hz this gives 50 fresh observations about every 0.417 seconds and
    # a complete per-object sweep about every 4.17 seconds.
    for batch in range(20):
        timestamp = 1.0 + (batch + 1) * 0.417
        start = (batch % 10) * 50
        ids = range(start, start + 50)
        for i in ids:
            updated_position = positions[i] + velocities[i] * timestamp
            observations[i] = _make_observation(
                updated_position, velocities[i], timestamp, diameters[i]
            )

        tracker.update(observations, timestamp, np.zeros(3), np.zeros(3))
        assert len(tracker.tracks) == 500
        assert tracker.last_update_stats["created"] == 0
        assert tracker.last_update_stats["matched"] == 50


def test_planner_selects_highest_risk_128_from_500_retained_tracks():
    assert CONFIG.max_planning_tracks == 128
    tracks = [
        DebrisTrack(
            identifier=i + 1,
            position=np.array([500.0 + i, 0.0, 0.0]),
            velocity=np.array([-1.0, 0.0, 0.0]),
            diameter=20.0,
            timestamp=0.0,
            hits=3,
            confirmed=True,
        )
        for i in range(500)
    ]
    planner = PredictivePlanner(CONFIG)
    arrays = planner._prepare_track_arrays(
        tracks,
        np.zeros(3),
        np.zeros(3),
        0.0,
    )
    assert arrays is not None
    assert len(arrays["position"]) == 128
    assert np.all(np.diff(arrays["cpa_clearance"]) >= -1.0e-9)
    assert set(arrays["identifier"].tolist()) == set(range(1, 129))
