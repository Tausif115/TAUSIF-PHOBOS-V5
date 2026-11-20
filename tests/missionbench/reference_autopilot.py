"""Bench-only reference autopilot (Step 0). NOT the navigation controller.

Runs as a STPro system through the same API as Controller.py (shuttle/target
truth state via getLocVelAcc; commands via JetSelection.CommandedAccel). It is
used only to compare launch strategies in the official plant:

  WAIT -> BURN (gravity-aware aimed launch at LaunchSpeed) -> COAST (command
  exactly zero) -> scheduled TRIMs (re-aim if error > TrimDeadband) ->
  after success latch: optional BRAKE (unscored) -> zero.

Corridor=1 selects the fly-by aim point with PERFECT debris foreknowledge (truth
debris states, oracle) - an upper bound for near-free avoidance, not a real
avoidance capability. Debris are otherwise ignored (no avoidance).
Dynamics knowledge: Phobos point mass (+ frame terms when FrameTerms=1), the
same physics the bench integrates.
"""
import math
import sys
import time

import numpy as np
import spaceteams as st

st.connect_to_sim(sys.argv)
this = st.GetThisSystem()


def P(key, default):
    return this.GetParam(st.VarType.double, key) if this.HasParam(key) else default


phobos = this.GetParam(st.VarType.entityRef, "Phobos")
shuttle = this.GetParam(st.VarType.entityRef, "Shuttle")
target = this.GetParam(st.VarType.entityRef, "Target")
phobosFrame = phobos.GetBodyFixedFrame()
shuttleFrame = shuttle.GetBodyFixedFrame()
MU = phobos.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"]) * 1e9
RATE = 2.0 * math.pi / (0.31891023 * 86400.0)
FRAME_TERMS = bool(P("FrameTerms", 1.0))
V0 = float(P("LaunchSpeed", 7.0))
MISS = float(P("MissDistance", 40.0))
AZ = math.radians(float(P("MissAzimuthDeg", 0.0)))
CORRIDOR = bool(P("Corridor", 0.0))
BRAKE = bool(P("PostSuccessBrake", 0.0))
TRIM_PERIOD = float(P("TrimPeriod", 60.0))
TRIM_FIRST = float(P("TrimFirstAfterBurn", 20.0))
DEADBAND = float(P("TrimDeadband", 0.03))
LAUNCH_T = float(P("LaunchTime", 2.0))
SHUTTLE_R = float(P("ShuttleRadius", 3.0))
CMD = 0.5 * np.ones(1)          # any non-zero command saturates (official actuator)
DONE_TOL = 0.004                # ~2 ticks of impulse
LOOP = 1.0 / this.GetParam(st.VarType.double, "LoopFreqHz")
t0 = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"]).as_datetime()


def accel(p, v):
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    a = -MU * p / r ** 3
    if FRAME_TERMS:
        n2 = RATE * RATE
        tide = n2 * np.stack([2.0 * p[..., 0], -p[..., 1], -p[..., 2]], axis=-1)
        fict = np.stack([n2 * p[..., 0] + 2 * RATE * v[..., 1], n2 * p[..., 1] - 2 * RATE * v[..., 0],
                         np.zeros(p.shape[:-1])], axis=-1)
        a = a + tide + fict
    return a


def closest_approach(p, v, aim, t_max=900.0, h=0.5):
    best = (np.inf, None, 0.0)
    t = 0.0
    while t < t_max:
        d = float(np.linalg.norm(aim - p))
        if d < best[0]:
            best = (d, aim - p, t)
        elif d > best[0] + 50.0:
            break
        a = accel(p, v)
        v = v + a * h
        p = p + v * h
        t += h
    return best


def solve_velocity(p, v_guess, aim, fixed_speed=None, iterations=40):
    v = np.array(v_guess, dtype=float)
    if fixed_speed is not None:
        v = v / np.linalg.norm(v) * fixed_speed
    for _ in range(iterations):
        d, vec, tca = closest_approach(p, v, aim)
        if d < 0.5 or vec is None:
            break
        v = v + 0.8 * vec / max(tca, 5.0)
        if fixed_speed is not None:
            v = v / np.linalg.norm(v) * fixed_speed
    return v, d


def state():
    s = shuttle.getLocVelAcc()
    return s.loc_WRT_ExprIn(phobosFrame), s.vel_WRT_ExprIn(phobosFrame)


def command(direction):
    vec = np.zeros(3) if direction is None else 0.5 * direction / np.linalg.norm(direction)
    body = st.frames.FramedVector(vec, phobosFrame).ExprIn(shuttleFrame)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], body)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], np.zeros(3))


def aim_point(p_start, miss, az):
    tgt = target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame)
    los = (tgt - p_start) / np.linalg.norm(tgt - p_start)
    e1 = np.cross(los, np.array([0.0, 0.0, 1.0])); e1 /= np.linalg.norm(e1)
    e2 = np.cross(los, e1)
    return tgt + miss * (math.cos(az) * e1 + math.sin(az) * e2)


def corridor_select(p, v):
    """Oracle: choose the aim point with the largest predicted debris clearance."""
    debris = list(st.SimGlobals.GetSimEntity().GetParam(st.VarType.entityRef, "Debris").values())
    dp = np.array([d.getLocVelAcc().loc_WRT_ExprIn(phobosFrame) for d in debris])
    dv = np.array([d.getLocVelAcc().vel_WRT_ExprIn(phobosFrame) for d in debris])
    dr = np.array([0.5 * d.GetParam(st.VarType.double, "RadarDiameter_m") for d in debris])
    h, steps = 0.5, 1800
    traj = np.empty((steps, len(debris), 3))
    for k in range(steps):
        traj[k] = dp
        dv = dv + accel(dp, dv) * h
        dp = dp + dv * h
    best = None
    for miss in (MISS, MISS - 8.0):
        for k_az in range(8):
            az = AZ + k_az * math.pi / 4.0
            aim = aim_point(p, miss, az)
            vel, _ = solve_velocity(p, aim - p, aim, fixed_speed=V0)
            sp, sv, clearance = p.copy(), vel.copy(), np.inf
            for k in range(steps):
                c = float(np.min(np.linalg.norm(traj[k] - sp, axis=1) - dr - SHUTTLE_R))
                clearance = min(clearance, c)
                if np.linalg.norm(aim - sp) < 10.0:
                    break
                sv = sv + accel(sp, sv) * h
                sp = sp + sv * h
            key = (min(clearance, 25.0), -k_az, miss)
            if best is None or key > best[0]:
                best = (key, aim, clearance, miss, az)
    st.OnScreenLogMessage(f"REFAP corridor miss={best[3]:.0f} az={math.degrees(best[4]):.0f} "
                          f"predicted_clearance={best[2]:.1f}", "RefAP", st.Severity.Info)
    return best[1]


mode, v_req, aim, next_solve, next_trim, latched = "WAIT", None, None, 0.0, None, False
while True:
    t = (st.SimGlobals.SimClock.GetTimeNow().as_datetime() - t0).total_seconds()
    p, v = state()
    tgt = target.getLocVelAcc().loc_WRT_ExprIn(phobosFrame)
    rng = float(np.linalg.norm(tgt - p))
    if not latched and rng <= 50.0:
        latched = True
        mode = "BRAKE" if BRAKE else "DONE"
        st.OnScreenLogMessage(f"REFAP latch t={t:.2f} speed={np.linalg.norm(v):.2f}", "RefAP", st.Severity.Info)
    direction = None
    if mode == "WAIT" and t >= LAUNCH_T:
        aim = corridor_select(p, v) if CORRIDOR else aim_point(p, MISS, AZ)
        v_req, _ = solve_velocity(p, aim - p, aim, fixed_speed=V0)
        mode, next_solve = "BURN", t + 2.0
    if mode == "BURN":
        if t >= next_solve:
            v_req, _ = solve_velocity(p, v_req, aim, fixed_speed=V0)
            next_solve = t + 2.0
        err = v_req - v
        if np.linalg.norm(err) < DONE_TOL:
            mode, next_trim = "COAST", t + TRIM_FIRST
            st.OnScreenLogMessage(f"REFAP burn_end t={t:.2f}", "RefAP", st.Severity.Info)
        else:
            direction = err
    elif mode == "COAST":
        if next_trim is not None and t >= next_trim and rng > 150.0:
            v_new, _ = solve_velocity(p, v, aim)
            if np.linalg.norm(v_new - v) > DEADBAND:
                v_req, mode = v_new, "TRIM"
                st.OnScreenLogMessage(f"REFAP trim t={t:.2f} dv={np.linalg.norm(v_new - v):.3f}", "RefAP", st.Severity.Info)
            next_trim = t + TRIM_PERIOD
    if mode == "TRIM":
        err = v_req - v
        if np.linalg.norm(err) < DONE_TOL:
            mode = "COAST"
        else:
            direction = err
    elif mode == "BRAKE":
        if np.linalg.norm(v) < 0.02:
            mode = "DONE"
        else:
            direction = -v
    command(direction)
    time.sleep(LOOP)
