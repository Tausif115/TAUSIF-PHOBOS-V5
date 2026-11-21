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


def _tracks(positions=(), velocities=None, diameters=None, confirmed=True, vel_sigma=0.1,
            radial_meas=None, los=None):
    """Mature, radar-consistent tracks by default (filter velocity sigma 0.1 m/s,
    the bench steady state)."""
    p = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(p)
    v = np.zeros((n, 3)) if velocities is None else np.asarray(velocities, dtype=float).reshape(-1, 3)
    d = np.full(n, 10.0) if diameters is None else np.asarray(diameters, dtype=float)
    return TrackStateArrays(
        positions=p, velocities=v, ages=np.zeros(n), diameters=d,
        confirmed=np.full(n, confirmed), confirmed_or_hits=np.full(n, confirmed),
        identifiers=np.arange(n), timestamp=0.0, has_position_at=np.ones(n, dtype=bool),
        radial_meas=None if radial_meas is None else np.asarray(radial_meas, dtype=float),
        los=None if los is None else np.asarray(los, dtype=float).reshape(-1, 3),
        vel_sigma=np.full(n, vel_sigma), clamped=np.zeros(n, dtype=bool),
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
    assert guide.phase == "WAIT" and metrics["safety_state"] == "LAUNCH_WAIT"
    assert np.linalg.norm(accel) <= CONFIG.v51_wait_max_cmd + 1e-12      # gravity hold only
    guide.update(12.0, SHUTTLE, np.zeros(3), TARGET, tracks, True, True, _clock)
    assert guide.phase == "WAIT"                                          # must stay ready for 1 s
    accel, _ = guide.update(13.1, SHUTTLE, np.zeros(3), TARGET, tracks, True, True, _clock)
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
    assert guide.phase == "WAIT" and np.linalg.norm(accel) <= CONFIG.v51_wait_max_cmd + 1e-12


def test_debris_on_coast_path_triggers_smallest_clearing_avoidance():
    guide = FlybyGuidance(CONFIG, MU)
    far = _tracks([[2000.0, 2000.0, -8000.0]])
    guide.update(12.0, SHUTTLE, np.zeros(3), TARGET, far, True, True, _clock)
    guide.update(13.1, SHUTTLE, np.zeros(3), TARGET, far, True, True, _clock)
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
    guide.update(81.0, SHUTTLE, v, TARGET, tracks, True, True, _clock)   # first sighting
    assert guide.phase == "COAST"                                        # persistence required
    guide.next_check = 0.0
    accel, metrics = guide.update(82.0, SHUTTLE, v, TARGET, tracks, True, True, _clock)
    assert guide.phase == "AVOID" and metrics["avoidance_active"]
    assert np.linalg.norm(accel) > 0.0
    dv = guide.v_target - v
    assert 0.0 < np.linalg.norm(dv) <= max(CONFIG.v51_avoid_magnitudes)
    assert guide.counts["emergencies"] == 0


def test_wait_hold_cancels_fall_with_proportional_commands():
    guide = FlybyGuidance(CONFIG, MU)
    up = (TARGET - SHUTTLE) / np.linalg.norm(TARGET - SHUTTLE)
    falling = -0.05 * up                                   # sinking toward the pad
    accel, _ = guide.update(5.0, SHUTTLE, falling, TARGET, _tracks(), True, False, _clock)
    assert guide.phase == "WAIT"
    assert accel @ up > 0.0                                # pushes back up
    assert np.linalg.norm(accel) <= CONFIG.v51_wait_max_cmd + 1e-12   # proportional regime
    accel, _ = guide.update(5.1, SHUTTLE, CONFIG.v51_wait_hold_speed_mps * up, TARGET, _tracks(), True, False, _clock)
    assert np.linalg.norm(accel) < 1e-12                   # at the hold velocity: no thrust


def test_fast_track_consistent_with_radar_range_rate_is_trusted():
    # live run 2e9a2483: debris genuinely move 1-5 m/s; V5.2's speed cap was wrong
    guide = FlybyGuidance(CONFIG, MU)
    p = SHUTTLE + np.array([60.0, 0.0, 0.0])
    los = np.array([[1.0, 0.0, 0.0]])
    v = [[-4.0, 1.5, 0.0]]                                  # radial -4 m/s, matches the radar
    r = guide._reliable(_tracks([p], velocities=v, radial_meas=[-4.0], los=los), 40.0)
    assert guide.implausible == 0 and r.confirmed[0] and r.cls[0] == 0
    np.testing.assert_allclose(r.velocities[0], v[0])


def test_velocity_inconsistent_with_range_rate_falls_back_to_radar_radial_velocity():
    guide = FlybyGuidance(CONFIG, MU)
    p = SHUTTLE + np.array([60.0, 0.0, 0.0])
    los = np.array([[1.0, 0.0, 0.0]])
    ts = _tracks([p], velocities=[[-8.0, 3.0, 0.0]], radial_meas=[-1.0], los=los)   # radial says -1
    ts.ages[:] = 3.0
    r = guide._reliable(ts, 40.0)
    assert guide.implausible == 1 and not r.confirmed[0] and r.lowconf[0] and r.cls[0] == 2
    np.testing.assert_allclose(r.velocities[0], [-1.0, 0.0, 0.0])           # measured radial only
    # position re-extrapolated from the last measurement with the fallback velocity
    np.testing.assert_allclose(r.positions[0], p - (np.array([-8.0, 3.0, 0.0]) - [-1.0, 0.0, 0.0]) * 3.0)


def test_low_confidence_track_guards_only_the_near_term_and_margins_follow_sigma():
    guide = FlybyGuidance(CONFIG, MU)
    p = SHUTTLE + np.array([60.0, 0.0, 0.0])
    fresh = guide._reliable(_tracks([p], vel_sigma=4.0), 40.0)           # 5 m/s tangential prior
    mature = guide._reliable(_tracks([p], vel_sigma=0.1), 40.0)
    assert fresh.lowconf[0] and not mature.lowconf[0]
    assert fresh.growth[0] == CONFIG.v53_growth_cap_mps
    assert mature.growth[0] == max(CONFIG.v51_margin_growth_mps, CONFIG.v53_sigma_k * 0.1)
    dt, steps = 1.0, 40
    path = np.repeat(p[None, None, :], steps + 1, axis=1)                 # sitting on the track
    grid = np.repeat(p[None, None, :], steps + 1, axis=0)
    early = np.repeat(path, 1, axis=0).copy()
    early[0, : int(CONFIG.v53_lowconf_horizon_s) + 1] += 500.0           # clear while the track is guarded
    clear, _, _ = guide._clearances(early, grid, fresh, dt)
    assert clear[0] > 0.0                                                 # ignored beyond the horizon
    clear, _, _ = guide._clearances(early, grid, mature, dt)
    assert clear[0] < 0.0                                                 # confident track: full horizon


def test_distant_unclearable_conflict_does_not_trigger_emergency_dodge():
    guide = FlybyGuidance(CONFIG, MU)
    far = _tracks([[2000.0, 2000.0, -8000.0]])
    guide.update(12.0, SHUTTLE, np.zeros(3), TARGET, far, True, True, _clock)
    guide.update(13.1, SHUTTLE, np.zeros(3), TARGET, far, True, True, _clock)
    v = guide.v_target.copy()
    guide.update(80.0, SHUTTLE, v, TARGET, far, True, True, _clock)
    dyn = guide.dyn
    p, vel = SHUTTLE.copy(), v.copy()
    for _ in range(700):                                   # ~70 s ahead
        vel = vel + np.array(dyn.accel(*p, *vel)) * 0.1
        p = p + vel * 0.1
    wall = _tracks([p], diameters=[400.0])                 # cannot be cleared by any candidate
    for t in (81.0, 82.0, 83.0):
        guide.next_check = 0.0
        accel, _ = guide.update(t, SHUTTLE, v, TARGET, wall, True, True, _clock)
    assert guide.phase == "COAST" and np.all(accel == 0.0)
    assert guide.counts["emergencies"] == 0 and guide.counts["deferred_emergencies"] >= 1


def _hold_point_after(guide, seconds):
    up = (TARGET - SHUTTLE) / np.linalg.norm(TARGET - SHUTTLE)
    return SHUTTLE + CONFIG.v51_wait_hold_speed_mps * up * seconds


def test_wait_pad_threat_triggers_emergency_liftoff_when_launch_clears_more():
    guide = FlybyGuidance(CONFIG, MU)
    up = (TARGET - SHUTTLE) / np.linalg.norm(TARGET - SHUTTLE)
    side = np.cross(up, [0.0, 0.0, 1.0]); side /= np.linalg.norm(side)
    # a confirmed 12 m rock crossing the hold point 25 s from now at 2 m/s
    p0 = _hold_point_after(guide, 30.0) - side * 2.0 * 25.0
    rock = _tracks([p0], velocities=[side * 2.0], diameters=[12.0])
    for t in (5.0, 6.0, 7.0):                       # radar up, tracks not ready yet
        guide.next_wait_check = 0.0
        guide.update(t, SHUTTLE, np.zeros(3), TARGET, rock, True, False, _clock)
    assert guide.phase == "BURN" and guide.launch_state == "EMERGENCY_LIFTOFF"
    assert guide.counts.get("liftoffs", 0) == 1


def test_blocked_launch_corridor_with_safe_pad_holds_then_launches_when_clear():
    guide = FlybyGuidance(CONFIG, MU)
    up = (TARGET - SHUTTLE) / np.linalg.norm(TARGET - SHUTTLE)
    blocker = _tracks([SHUTTLE + up * 45.0], diameters=[24.0])    # parked in the corridor, 45 m up
    for t in (12.0, 13.1, 14.2, 15.3):
        guide.update(t, SHUTTLE, np.zeros(3), TARGET, blocker, True, True, _clock)
    assert guide.phase == "WAIT" and guide.launch_state == "SAFE_HOLD"
    _, metrics = guide.update(16.4, SHUTTLE, np.zeros(3), TARGET, blocker, True, True, _clock)
    assert metrics["safety_state"] == "SAFE_HOLD"
    gone = _tracks([[2000.0, 2000.0, -8000.0]])
    guide.update(17.5, SHUTTLE, np.zeros(3), TARGET, gone, True, True, _clock)
    assert guide.phase == "BURN" and guide.launch_state == "LAUNCH_COMMIT"


def test_suspect_track_is_low_confidence():
    guide = FlybyGuidance(CONFIG, MU)
    ts = _tracks([SHUTTLE + np.array([80.0, 0.0, 0.0])])
    ts.suspect = np.array([True])
    r = guide._reliable(ts, 40.0)
    assert r.lowconf[0] and r.cls[0] == 2
