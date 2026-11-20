"""V5.1 fly-by guidance unit tests (phobos_nav/ballistic.py). Offline only."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.ballistic import Dynamics, FlybyGuidance, closest_approach, solve_velocity
from phobos_nav.threats import TrackStateArrays

SHUTTLE = np.array([-651.0228484243066, 95.19902620579852, -8239.668001620705])
TARGET = np.array([-850.0, 120.0, -9530.0])
MU = 7.075e5


def _tracks(positions=(), velocities=None, diameters=None, confirmed=True):
    p = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(p)
    v = np.zeros((n, 3)) if velocities is None else np.asarray(velocities, dtype=float).reshape(-1, 3)
    d = np.full(n, 10.0) if diameters is None else np.asarray(diameters, dtype=float)
    return TrackStateArrays(
        positions=p, velocities=v, ages=np.zeros(n), diameters=d,
        confirmed=np.full(n, confirmed), confirmed_or_hits=np.full(n, confirmed),
        identifiers=np.arange(n), timestamp=0.0, has_position_at=np.ones(n, dtype=bool),
    )


def _clock():
    return 0.0


def test_defaults_select_flyby_and_legacy_is_available():
    assert CONFIG.guidance_mode == "flyby_v51"
    assert replace(CONFIG, guidance_mode="legacy_mpc").guidance_mode == "legacy_mpc"


def test_gravity_and_frame_terms_match_bench_physics():
    dyn = Dynamics(MU, CONFIG.v51_phobos_rate, True)
    a = np.array(dyn.accel(*SHUTTLE, 0.0, 0.0, 0.0))
    g = -MU * SHUTTLE / np.linalg.norm(SHUTTLE) ** 3
    assert abs(np.linalg.norm(g) - 0.01035) < 1e-4
    n = CONFIG.v51_phobos_rate
    expected = g + np.array([3 * n * n * SHUTTLE[0], 0.0, -n * n * SHUTTLE[2]])
    np.testing.assert_allclose(a, expected, rtol=1e-12)
    np.testing.assert_allclose(dyn.accel_array(SHUTTLE[None], np.zeros((1, 3)))[0], a, rtol=1e-12)


def test_gravity_aware_solution_passes_through_aim_point():
    dyn = Dynamics(MU, CONFIG.v51_phobos_rate, True)
    aim = TARGET + np.array([35.0, 0.0, 0.0])
    v, miss = solve_velocity(dyn, SHUTTLE, aim - SHUTTLE, aim, fixed_speed=7.0)
    assert miss < 0.5 and abs(np.linalg.norm(v) - 7.0) < 1e-9
    d, _, tca = closest_approach(dyn, SHUTTLE, v, aim)
    assert d < 0.5 and 150.0 < tca < 260.0
    # gravity matters: aiming straight at the point misses it by ~17 m (inside the
    # 50 m latch, but the solver removes the error instead of spending trim fuel)
    straight = (aim - SHUTTLE) / np.linalg.norm(aim - SHUTTLE) * 7.0
    assert closest_approach(dyn, SHUTTLE, straight, aim)[0] > 10.0


def test_waits_for_confirmed_tracks_then_burns_then_coasts_with_zero_command():
    guide = FlybyGuidance(CONFIG, MU)
    tracks = _tracks([[2000.0, 2000.0, -8000.0]])
    accel, metrics = guide.update(1.0, SHUTTLE, np.zeros(3), TARGET, tracks, True, True, _clock)
    assert guide.phase == "WAIT" and np.all(accel == 0.0) and metrics["safety_state"] == "LAUNCH_WAIT"
    accel, _ = guide.update(12.0, SHUTTLE, np.zeros(3), TARGET, tracks, True, True, _clock)
    assert guide.phase == "BURN" and np.linalg.norm(accel) > 0.0
    burn_dir = accel / np.linalg.norm(accel)
    assert burn_dir @ ((TARGET - SHUTTLE) / np.linalg.norm(TARGET - SHUTTLE)) > 0.99
    # velocity target reached -> coast with EXACTLY zero command (no thrust chatter)
    accel, metrics = guide.update(80.0, SHUTTLE, guide.v_target.copy(), TARGET, tracks, True, True, _clock)
    assert guide.phase == "COAST" and np.all(accel == 0.0) and metrics["safety_state"] == "CLEAR"


def test_no_launch_on_stale_radar():
    guide = FlybyGuidance(CONFIG, MU)
    tracks = _tracks([[2000.0, 2000.0, -8000.0]])
    accel, _ = guide.update(40.0, SHUTTLE, np.zeros(3), TARGET, tracks, False, False, _clock)
    assert guide.phase == "WAIT" and np.all(accel == 0.0)


def test_debris_on_coast_path_triggers_smallest_clearing_avoidance():
    guide = FlybyGuidance(CONFIG, MU)
    far = _tracks([[2000.0, 2000.0, -8000.0]])
    guide.update(12.0, SHUTTLE, np.zeros(3), TARGET, far, True, True, _clock)
    v = guide.v_target.copy()
    guide.update(80.0, SHUTTLE, v, TARGET, far, True, True, _clock)        # coasting, clear
    assert guide.phase == "COAST"
    # a 20 m object parked on the predicted path ~40 s ahead
    dyn = guide.dyn
    p, vel = SHUTTLE.copy(), v.copy()
    for _ in range(400):
        vel = vel + np.array(dyn.accel(*p, *vel)) * 0.1
        p = p + vel * 0.1
    tracks = _tracks([p], diameters=[20.0])
    guide.next_check = 0.0
    accel, metrics = guide.update(81.0, SHUTTLE, v, TARGET, tracks, True, True, _clock)
    assert guide.phase == "AVOID" and metrics["avoidance_active"]
    assert np.linalg.norm(accel) > 0.0
    dv = guide.v_target - v
    assert 0.0 < np.linalg.norm(dv) <= max(CONFIG.v51_avoid_magnitudes)
    assert guide.counts["emergencies"] == 0
