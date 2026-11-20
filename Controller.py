#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt


this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(
        st.VarType.doubleM3x3,
        ["Radar", "SortedDebrisData"]
    )
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(
        st.math.Euler321_to_DCM([
            np.deg2rad(yaw_deg),
            np.deg2rad(pitch_deg),
            0.0
        ]),
        shuttleFrame
    )


def SetCommand(
    accelCommand: npt.NDArray,
    angAccelCommand: npt.NDArray
) -> None:
    shuttle.SetParam(
        st.VarType.doubleV3,
        ["JetSelection", "CommandedAccel"],
        accelCommand
    )
    shuttle.SetParam(
        st.VarType.doubleV3,
        ["JetSelection", "CommandedAngAccel"],
        angAccelCommand
    )


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################


# Competitor code begins here. The official actuator and competition simconf
# are not modified by this build.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTAL_DIR = os.path.join(REPO_DIR, "experimental")
if EXPERIMENTAL_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTAL_DIR)

from phobos_nav.config import CONFIG
from phobos_nav.geometry import norm
from phobos_nav.radar import RadarClockSynchronizer, decode_radar
from phobos_nav.radar_diagnostics import format_radar_diagnostic
from phobos_nav.tracking import DebrisTracker, StateHistory
from phobos_nav.planning import PredictivePlanner
from phobos_nav.scheduling import DeadlinePlanner
from phobos_nav.threats import (
    track_state_arrays,
    nearest_track_distance as nearest_track_distance_from_state,
    threat_ttc_seconds as threat_ttc_seconds_from_state,
)
from phobos_nav.actuator_model import (
    read_thruster_csv,
    build_activity_matrix,
    compute_axis_authorities,
    limit_acceleration_body,
    limit_acceleration_body_batch,
)


# Shared controller time epoch: simulation StartDateTAI.
t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(
    st.VarType.timestamp,
    ["Time", "StartDateTAI"]
)
t0: datetime.datetime = t0_sim.as_datetime()

loop_frequency_hz = float(this.GetParam(st.VarType.double, "LoopFreqHz"))
if not np.isfinite(loop_frequency_hz) or loop_frequency_hz <= 0.0:
    raise RuntimeError("Controller LoopFreqHz must be positive")

loop_period_s = 1.0 / loop_frequency_hz
max_linear_accel = float(shuttle.GetParam(
    st.VarType.double,
    ["Thrusters", "MaxAccel_m_s2"]
))
if not np.isfinite(max_linear_accel) or max_linear_accel <= 0.0:
    raise RuntimeError("Invalid Thrusters.MaxAccel_m_s2")

tracker = DebrisTracker(CONFIG)
state_history = StateHistory(
    max_age_s=CONFIG.state_history_age_s
)
radar_clock = RadarClockSynchronizer(CONFIG)
planner = PredictivePlanner(CONFIG)
deadline_planner = DeadlinePlanner(planner, CONFIG)

# Read-only estimate of official thruster authority from the asset path
# configured by JetConfigName in the original competition simconf.
authority_positive = np.zeros(3, dtype=float)
authority_negative = np.zeros(3, dtype=float)
authority_valid = False
authority_error = "THRUSTER_CSV_NOT_READY"
authority_last_refresh_s = -1.0e9
authority_last_mass = float("nan")
authority_last_inertia = None
authority_residual = float("inf")
authority_last_attempt_s = -1.0e9
rotation_phobos_to_body = np.eye(3, dtype=float)
thruster_records_cache = None


def load_thruster_records():
    """Read the CSV referenced by the original simconf; never modify it."""
    asset_path = st.path_utils.AssetPathToReal(
        st.AssetType.Generic,
        CONFIG.jet_config_asset_path
    )
    return read_thruster_csv(asset_path)


def refresh_thruster_authority(sim_time, force=False):
    global authority_positive, authority_negative
    global authority_valid, authority_error
    global authority_last_refresh_s, authority_last_mass
    global authority_last_inertia, authority_residual
    global authority_last_attempt_s, thruster_records_cache

    retry_interval = 1.0 if not authority_valid else CONFIG.authority_refresh_interval_s
    if (
        not force
        and authority_last_attempt_s > -1.0e8
        and sim_time - authority_last_attempt_s < retry_interval
        and not authority_valid
    ):
        return

    try:
        authority_last_attempt_s = float(sim_time)
        mass = float(shuttle.getMass())
        inertia = np.asarray(
            shuttle.getInertiaTensor(shuttleFrame),
            dtype=float
        ).reshape(3, 3)
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("Invalid live shuttle mass")
        if not np.all(np.isfinite(inertia)):
            raise RuntimeError("Invalid live shuttle inertia")

        mass_changed = (
            not np.isfinite(authority_last_mass)
            or abs(mass - authority_last_mass)
            > CONFIG.authority_mass_change_fraction
            * max(1.0, abs(authority_last_mass))
        )
        inertia_changed = (
            authority_last_inertia is None
            or np.linalg.norm(inertia - authority_last_inertia)
            > CONFIG.authority_inertia_change_fraction
            * max(1.0, np.linalg.norm(inertia))
        )
        due = (
            force
            or authority_last_refresh_s < 0.0
            or sim_time - authority_last_refresh_s
            >= CONFIG.authority_refresh_interval_s
            or mass_changed
            or inertia_changed
        )
        if not due:
            return

        if thruster_records_cache is None:
            thruster_records_cache = load_thruster_records()
        activity_matrix = build_activity_matrix(
            thruster_records_cache, mass, inertia
        )
        positive, negative, valid, residual = compute_axis_authorities(
            activity_matrix
        )
        positive = np.minimum(positive, max_linear_accel)
        negative = np.minimum(negative, max_linear_accel)

        if not valid:
            raise RuntimeError(
                "Thruster LP could not establish all six translation axes"
            )
        if (
            not np.all(np.isfinite(positive))
            or not np.all(np.isfinite(negative))
            or np.any(positive <= 1.0e-6)
            or np.any(negative <= 1.0e-6)
        ):
            raise RuntimeError("Invalid or zero directional thrust authority")

        authority_positive = positive
        authority_negative = negative
        authority_valid = True
        authority_error = ""
        authority_residual = float(residual)
        authority_last_refresh_s = float(sim_time)
        authority_last_mass = mass
        authority_last_inertia = inertia.copy()

    except Exception as error:
        authority_valid = False
        authority_error = str(error)


def limit_accel_phobos(command_phobos):
    """Convert Phobos-frame acceleration to body axes and limit it to read-only
    zero-torque thrust authority. Supports either one vector or an (N,3) batch.
    """
    command = np.asarray(command_phobos, dtype=float)
    if not authority_valid:
        return np.zeros_like(command)

    rotation = rotation_phobos_to_body
    if (
        not np.all(np.isfinite(rotation))
        or np.linalg.norm(
            rotation.T @ rotation - np.eye(3), ord=np.inf
        ) > 1.0e-3
    ):
        return np.zeros_like(command)

    if command.ndim == 2:
        if command.shape[1] != 3:
            raise ValueError("Batch acceleration commands must have shape (N,3)")
        command_body = command @ rotation.T
        limited_body = limit_acceleration_body_batch(
            command_body,
            authority_positive,
            authority_negative,
            safety_factor=CONFIG.authority_safety_factor,
        )
        return limited_body @ rotation

    command_body = rotation @ command.reshape(3)
    limited_body, _ = limit_acceleration_body(
        command_body,
        authority_positive,
        authority_negative,
        safety_factor=CONFIG.authority_safety_factor,
    )
    return rotation.T @ limited_body


def get_frame_rotation():
    """Return Phobos-to-body DCM and validate it before any command transform."""
    basis_vectors_body = [
        np.asarray(
            st.frames.FramedVector(axis, phobosFrame).ExprIn(shuttleFrame),
            dtype=float
        )
        for axis in np.eye(3, dtype=float)
    ]
    rotation = np.column_stack(basis_vectors_body)
    if (
        not np.all(np.isfinite(rotation))
        or np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf) > 1.0e-3
    ):
        raise RuntimeError("Invalid Phobos-to-shuttle frame rotation")
    return rotation


# V5 Phase 2: the per-track threat loops (V4.3 threat_ttc_seconds and the
# nearest-track generator) are vectorized in phobos_nav/threats.py.


iteration = 0
cached_accel_phobos = np.zeros(3, dtype=float)
cached_planner_metrics = {
    "risk": 0.0,
    "distance": float("inf"),
    "clearance": float("inf"),
    "emergency": False,
    "safety_state": "STARTUP",
    "candidate_count": 0,
    "safe_candidate_count": 0,
    "preferred_candidate_count": 0,
    "closest_track_id": -1,
    "planning_track_count": 0,
}
previous_velocity_for_diagnostics = None
previous_velocity_time = None
measured_accel_vector_phobos = np.full(3, np.nan, dtype=float)
measured_accel_norm_mps2 = float("nan")
# V5 Phase 2 loop-deadline telemetry (logging only).
previous_loop_start_wall = None
loop_interval_ms = float("nan")
previous_loop_work_ms = float("nan")
loop_overrun_count = 0


def fail_closed_tick(safety_state, detail=""):
    """Send zero body-frame thrust and keep the controller alive for retry."""
    global iteration
    SetCommand(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
    if iteration % CONFIG.log_every_n == 0:
        print(
            f"PHOBOS_NAV_V50 t={t_from_start:.2f} "
            f"safety={safety_state} detail={str(detail)!r} "
            "cmd_accel_mps2=0.000",
            flush=True
        )
    iteration += 1
    remaining_sleep = loop_period_s - (
        time.perf_counter() - loop_start_wall
    )
    if remaining_sleep > 0.0:
        time.sleep(remaining_sleep)


while True:
    loop_start_wall = time.perf_counter()
    if previous_loop_start_wall is not None:
        loop_interval_ms = 1000.0 * (loop_start_wall - previous_loop_start_wall)
    previous_loop_start_wall = loop_start_wall
    t_now = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    t_from_start = (t_now - t0).total_seconds()

    # Current shuttle state in the Phobos body-fixed frame.
    try:
        shuttle_state = shuttle.getLocVelAcc()
        loc = np.asarray(
            shuttle_state.loc_WRT_ExprIn(phobosFrame),
            dtype=float
        ).reshape(3)
        vel = np.asarray(
            shuttle_state.vel_WRT_ExprIn(phobosFrame),
            dtype=float
        ).reshape(3)
    except Exception as error:
        fail_closed_tick("SHUTTLE_STATE_READ_ERROR", error)
        continue
    if not np.all(np.isfinite(loc)) or not np.all(np.isfinite(vel)):
        fail_closed_tick("SHUTTLE_STATE_INVALID", "nonfinite location/velocity")
        continue

    if previous_velocity_for_diagnostics is not None:
        diagnostic_dt = t_from_start - previous_velocity_time
        if diagnostic_dt > 1.0e-6:
            measured_accel_vector_phobos = (
                vel - previous_velocity_for_diagnostics
            ) / diagnostic_dt
            measured_accel_norm_mps2 = float(np.linalg.norm(
                measured_accel_vector_phobos
            ))
    previous_velocity_for_diagnostics = vel.copy()
    previous_velocity_time = float(t_from_start)

    # Record state before reconstructing delayed radar measurements.
    state_history.add(t_from_start, loc, vel)

    try:
        target_state = target.getLocVelAcc()
        target_loc = np.asarray(
            target_state.loc_WRT_ExprIn(phobosFrame),
            dtype=float
        ).reshape(3)
        target_vel = np.asarray(
            target_state.vel_WRT_ExprIn(phobosFrame),
            dtype=float
        ).reshape(3)
    except Exception as error:
        fail_closed_tick("TARGET_STATE_READ_ERROR", error)
        continue
    if not np.all(np.isfinite(target_loc)) or not np.all(np.isfinite(target_vel)):
        fail_closed_tick("TARGET_STATE_INVALID", "nonfinite location/velocity")
        continue
    target_vector = target_loc - loc
    target_distance = float(np.linalg.norm(target_vector))
    relative_velocity = vel - target_vel
    relative_speed = float(np.linalg.norm(relative_velocity))

    # Refresh the body/Phobos transform once per loop. The planner reuses this
    # validated DCM during its vectorized rollout instead of querying frames
    # inside every prediction step.
    try:
        rotation_phobos_to_body = get_frame_rotation()
    except Exception as error:
        # A bad frame transform must never be converted into a thrust command.
        fail_closed_tick("FRAME_TRANSFORM_INVALID", error)
        continue

    # Refresh the read-only authority model from the CSV asset named in the
    # original simconf. The controller never writes the file or jet data.
    refresh_thruster_authority(t_from_start)

    radar_error = None
    raw_debris = []
    observations = []
    cached_record_count = 0
    radar_available_count = 0
    radar_clock_offset_s = radar_clock.offset_s
    radar_latest_age_s = float("inf")
    radar_clock_ready = False
    radar_start_wall = time.perf_counter()

    try:
        # Use the documented competitor helper; it reads only the on-board
        # radar array and caps requests at the API's 500-record maximum.
        raw_debris = GetNearestDebrisLocations(CONFIG.radar_max_objects)
        radar_available_count = len(raw_debris)

        radar_clock_offset_s, radar_clock_ready, radar_latest_age_s = (
            radar_clock.update(raw_debris, t_from_start)
        )
        if not radar_clock_ready:
            radar_error = "RADAR_CLOCK_SYNC_OR_STALE"
        else:
            # V5 Phase 2: decode only records not already processed. Cached
            # copies (about 9/10 of every publication) are recognised by
            # their raw bytes and skipped before decoding.
            new_records, cached_record_count = tracker.select_new_records(
                raw_debris, t_from_start,
                limit=CONFIG.radar_max_new_records_per_tick,
            )
            observations = decode_radar(
                new_records,
                loc,
                shuttle_velocity=vel,
                current_timestamp=t_from_start,
                state_history=state_history,
                radar_time_offset_s=radar_clock_offset_s,
                config=CONFIG,
            )
            tracks = tracker.update(
                observations,
                t_from_start,
                loc,
                vel,
                prefiltered_duplicates=cached_record_count,
            )
    except Exception as error:
        radar_error = f"RADAR_READ_ERROR:{error}"
        tracks = tracker.tracks
    # Records decoded validly this poll plus valid records decoded earlier
    # and still present (same meaning as the V4.3 radar_decoded field).
    radar_decoded_count = len(observations) + cached_record_count
    radar_ms = 1000.0 * (time.perf_counter() - radar_start_wall)
    threat_ms = 0.0

    # Authority is required before issuing a translation command. If the
    # official actuator has not published its read-only jet records yet,
    # remain at zero acceleration and report the exact reason.
    if not authority_valid:
        accel_phobos = np.zeros(3, dtype=float)
        planner_metrics = {
            "risk": float("inf"),
            "distance": target_distance,
            "clearance": float("-inf"),
            "emergency": True,
            "safety_state": "AUTHORITY_WAIT",
            "candidate_count": 0,
            "safe_candidate_count": 0,
            "preferred_candidate_count": 0,
            "closest_track_id": -1,
            "planning_track_count": len(tracks) if "tracks" in locals() else 0,
        }
        min_ttc_s = float("inf")
        desired_approach_speed = 0.0
        nearest_track_distance = float("inf")

    elif radar_error is not None:
        # Never continue target pursuit on stale/un-aligned radar. Remove
        # target-relative velocity using only the validated thrust model.
        accel_phobos = limit_accel_phobos(target_vel - vel)
        cached_accel_phobos = accel_phobos.copy()
        planner_metrics = {
            "risk": float("inf"),
            "distance": target_distance,
            "clearance": float("-inf"),
            "emergency": True,
            "safety_state": "RADAR_HOLD",
            "candidate_count": 0,
            "safe_candidate_count": 0,
            "preferred_candidate_count": 0,
            "closest_track_id": -1,
            "planning_track_count": len(tracks) if "tracks" in locals() else 0,
        }
        min_ttc_s = float("inf")
        desired_approach_speed = 0.0
        nearest_track_distance = float("inf")

    else:
        # V5 Phase 2: one vectorized pass over all retained tracks replaces
        # two per-track Python loops (same mathematics; equivalence-tested).
        threat_start_wall = time.perf_counter()
        track_state = track_state_arrays(tracks, t_from_start)
        nearest_track_distance = nearest_track_distance_from_state(
            track_state, loc
        )
        min_ttc_s = threat_ttc_seconds_from_state(
            track_state, loc, vel, CONFIG
        )
        threat_ms = 1000.0 * (time.perf_counter() - threat_start_wall)

        # The speed profile must not assume braking authority equals forward
        # authority. The installed jet geometry is asymmetric, so use the
        # smaller of the measured forward and reverse line-of-sight limits.
        # The MPC still applies the full directional limiter to every rollout.
        goal_direction = target_vector / max(target_distance, 1.0e-9)
        forward_authority = float(np.linalg.norm(
            limit_accel_phobos(goal_direction * max_linear_accel)
        ))
        reverse_authority = float(np.linalg.norm(
            limit_accel_phobos(-goal_direction * max_linear_accel)
        ))
        guidance_accel = max(
            1.0e-4,
            min(forward_authority, reverse_authority),
        )
        desired_approach_speed = min(
            CONFIG.goal_speed,
            np.sqrt(
                CONFIG.terminal_speed_mps ** 2
                + 2.0 * guidance_accel
                * max(target_distance - CONFIG.capture_radius_m, 0.0)
                * CONFIG.braking_accel_fraction
            )
        )

        cached_clearance = float(
            cached_planner_metrics.get("clearance", float("inf"))
        )
        cached_emergency = bool(
            cached_planner_metrics.get("emergency", False)
        )

        planning_critical = bool(
            cached_emergency
            or cached_clearance <= CONFIG.minimum_safe_clearance_m + 2.0
            or min_ttc_s <= CONFIG.critical_replan_ttc_s
            or nearest_track_distance <= CONFIG.critical_replan_distance_m
        )
        if planning_critical:
            planner_due = (
                iteration % CONFIG.critical_replan_every_n == 0
            )
        elif (
            min_ttc_s <= CONFIG.urgent_replan_ttc_s
            or nearest_track_distance <= CONFIG.urgent_replan_distance_m
            or cached_clearance
            < CONFIG.replan_buffer_m + CONFIG.clearance_early_warning_m
        ):
            planner_due = (
                iteration % CONFIG.urgent_replan_every_n == 0
            )
        else:
            planner_due = (
                iteration % CONFIG.planner_update_every_n == 0
            )

        # V5 Phase 2b: deadline-aware two-stage planning (phobos_nav/
        # scheduling.py). A due plan is latched and its stages run when they
        # fit this tick's budget (at most one tick of deferral; immediately
        # under critical conditions). The cached command is held meanwhile.
        planner_start = time.perf_counter()
        try:
            completed_plan = deadline_planner.tick(
                loop_start_wall,
                planner_due,
                planning_critical,
                t_from_start,
                lambda: dict(
                    position=loc,
                    velocity=vel,
                    target_position=target_loc,
                    target_velocity=target_vel,
                    tracks=tracks,
                    timestamp=t_from_start,
                    max_accel=guidance_accel,
                    acceleration_limiter=limit_accel_phobos,
                ),
            )
            if completed_plan is not None:
                cached_accel_phobos, cached_planner_metrics = completed_plan
        except Exception as error:
            # Do not continue blind target pursuit if the planner fails.
            deadline_planner.reset()
            cached_accel_phobos = limit_accel_phobos(target_vel - vel)
            cached_planner_metrics = {
                "risk": float("inf"),
                "distance": target_distance,
                "clearance": float("-inf"),
                "emergency": True,
                "safety_state": "PLANNER_ERROR",
                "planner_error": str(error),
                "candidate_count": 0,
                "safe_candidate_count": 0,
                "preferred_candidate_count": 0,
                "closest_track_id": -1,
                "planning_track_count": len(tracks),
                "planner_compute_ms": (
                    1000.0 * (time.perf_counter() - planner_start)
                ),
            }

        accel_phobos = cached_accel_phobos.copy()
        planner_metrics = cached_planner_metrics

    # Final validation and body-frame conversion. Angular command remains zero
    # so the planner's zero-torque translation authority model remains valid.
    accel_phobos = np.asarray(accel_phobos, dtype=float).reshape(3)
    if not np.all(np.isfinite(accel_phobos)):
        accel_phobos = np.zeros(3, dtype=float)

    linear_accel_body = rotation_phobos_to_body @ accel_phobos
    angular_accel_body = np.zeros(3, dtype=float)

    SetCommand(
        linear_accel_body,
        angular_accel_body
    )

    if iteration % CONFIG.log_every_n == 0:
        try:
            fuel_used = float(shuttle.GetParam(
                st.VarType.double,
                ["Stats", "FuelUsed"]
            ))
        except Exception:
            fuel_used = float("nan")
        try:
            damage = float(shuttle.GetParam(
                st.VarType.double,
                ["Stats", "Damage"]
            ))
        except Exception:
            damage = float("nan")

        clearance = float(
            planner_metrics.get("clearance", float("inf"))
        )
        if np.isposinf(clearance):
            clearance_text = "inf"
        elif np.isneginf(clearance):
            clearance_text = "-inf"
        elif np.isnan(clearance):
            clearance_text = "nan"
        else:
            clearance_text = f"{clearance:.2f}"

        # Validation-only observability: expose the age of the oldest and
        # 95th-percentile retained debris tracks in the live controller log.
        track_ages_s = np.asarray([
            max(0.0, t_from_start - float(track.timestamp))
            for track in tracker.tracks
        ], dtype=float)
        track_age_max_s = (
            float(np.max(track_ages_s)) if track_ages_s.size else 0.0
        )
        track_age_p95_s = (
            float(np.percentile(track_ages_s, 95.0))
            if track_ages_s.size else 0.0
        )
        fresh_age_max_s = float(
            tracker.last_update_stats.get("fresh_age_max", float("nan"))
        )

        log_line = (
            f"PHOBOS_NAV_V50 "
            f"t={t_from_start:.2f} "
            f"range_m={target_distance:.2f} "
            f"rel_speed_mps={relative_speed:.3f} "
            f"desired_speed_mps={desired_approach_speed:.3f} "
            f"forward_authority_mps2={forward_authority if authority_valid and radar_error is None else 0.0:.3f} "
            f"reverse_authority_mps2={reverse_authority if authority_valid and radar_error is None else 0.0:.3f} "
            f"guidance_accel_mps2={guidance_accel if authority_valid and radar_error is None else 0.0:.3f} "
            f"cmd_accel_mps2={np.linalg.norm(accel_phobos):.3f} "
            f"cmd_accel_phobos={np.round(accel_phobos, 4).tolist()} "
            f"measured_accel_norm_mps2={measured_accel_norm_mps2:.3f} "
            f"measured_accel_phobos={np.round(measured_accel_vector_phobos, 4).tolist()} "
            f"target_direction_phobos={np.round(target_vector / max(target_distance, 1.0e-9), 4).tolist()} "
            f"relative_velocity_phobos={np.round(relative_velocity, 4).tolist()} "
            f"authority_valid={int(authority_valid)} "
            f"authority_source=THRUSTER_CSV_READ_ONLY "
            f"authority_error={authority_error!r} "
            f"authority_pos={np.round(authority_positive, 3).tolist()} "
            f"authority_neg={np.round(authority_negative, 3).tolist()} "
            f"authority_residual={authority_residual:.3g} "
            f"radar_clock_ready={int(radar_clock_ready)} "
            f"radar_clock_offset_s={radar_clock_offset_s:.3f} "
            f"radar_clock_locked={int(radar_clock.locked)} "
            f"radar_clock_resyncs={radar_clock.resync_count} "
            f"radar_latest_age_s={radar_latest_age_s:.3f} "
            f"radar_available={radar_available_count} "
            f"radar_read={len(raw_debris)} "
            f"radar_decoded={radar_decoded_count} "
            f"radar_new_decoded={len(observations)} "
            f"radar_deferred_new={tracker.deferred_new_records} "
            f"fresh={tracker.last_update_stats.get('fresh', 0)} "
            f"tracks={len(tracker.tracks)} "
            f"track_age_max_s={track_age_max_s:.3f} "
            f"track_age_p95_s={track_age_p95_s:.3f} "
            f"fresh_age_max_s={fresh_age_max_s:.3f} "
            f"confirmed={sum(x.confirmed for x in tracker.tracks)} "
            f"matched={tracker.last_update_stats.get('matched', 0)} "
            f"created={tracker.last_update_stats.get('created', 0)} "
            f"duplicates={tracker.last_update_stats.get('duplicates', 0)} "
            f"stale={tracker.last_update_stats.get('stale', 0)} "
            f"invalid={tracker.last_update_stats.get('invalid', 0)} "
            f"too_soon={tracker.last_update_stats.get('too_soon', 0)} "
            f"reacquired_total={tracker.cumulative['reacquired']} "
            f"evicted_total={tracker.cumulative['evicted']} "
            f"evicted_confirmed_total={tracker.cumulative['evicted_confirmed']} "
            f"expired_total={tracker.cumulative['expired']} "
            f"replays_total={tracker.cumulative['replays']} "
            f"velocity_clamped_total={tracker.cumulative['velocity_clamped']} "
            f"track_ids_issued={tracker.next_identifier - 1} "
            f"revisit_s={tracker.last_update_stats.get('revisit_s', float('nan')):.3f} "
            f"track_expiry_s={tracker.last_update_stats.get('expiry_s', float('nan')):.3f} "
            f"min_ttc_s={min_ttc_s:.2f} "
            f"nearest_track_m={nearest_track_distance:.2f} "
            f"planner={planner.last_choice} "
            f"safety={planner_metrics.get('safety_state', 'UNKNOWN')} "
            f"planner_error={str(planner_metrics.get('planner_error', ''))[:100]!r} "
            f"emergency={int(bool(planner_metrics.get('emergency', False)))} "
            f"clearance_m={clearance_text} "
            f"nominal_clearance_m={planner_metrics.get('nominal_clearance_m', float('nan')):.2f} "
            f"closest_time_s={planner_metrics.get('closest_time_s', float('nan')):.2f} "
            f"closest_track_id={planner_metrics.get('closest_track_id', -1)} "
            f"planning_tracks={planner_metrics.get('planning_track_count', 0)} "
            f"candidates={planner_metrics.get('candidate_count', 0)} "
            f"safe_candidates={planner_metrics.get('safe_candidate_count', 0)} "
            f"preferred_candidates={planner_metrics.get('preferred_candidate_count', 0)} "
            f"selected_bias={planner_metrics.get('selected_bias', [0, 0, 0])} "
            f"selected_hold={int(bool(planner_metrics.get('selected_hold', False)))} "
            f"emergency_clearance_margin_m={planner_metrics.get('emergency_clearance_margin_m', float('nan')):.2f} "
            f"avoidance_active={int(bool(planner_metrics.get('avoidance_active', False)))} "
            f"planner_compute_ms={planner_metrics.get('planner_compute_ms', 0.0):.2f} "
            f"radar_ms={radar_ms:.2f} "
            f"threat_ms={threat_ms:.2f} "
            f"loop_interval_ms={loop_interval_ms:.1f} "
            f"prev_loop_work_ms={previous_loop_work_ms:.1f} "
            f"loop_overruns_total={loop_overrun_count} "
            f"plan_stage1_ms={deadline_planner.stats['stage1_ms']:.2f} "
            f"plan_stage2_ms={deadline_planner.stats['stage2_ms']:.2f} "
            f"plans_completed_total={deadline_planner.stats['completed']} "
            f"plan_deferrals_total={deadline_planner.stats['deferred']} "
            f"plan_forced_total={deadline_planner.stats['forced']} "
            f"plan_discarded_total={deadline_planner.stats['discarded']} "
            f"fuel_kg={fuel_used:.3f} "
            f"damage={damage:.3f}"
        )
        if radar_error is not None:
            log_line += f" radar_error={radar_error!r}"

        print(log_line, flush=True)
        severity = (
            st.Severity.Warning
            if radar_error is not None or bool(planner_metrics.get("emergency", False))
            else st.Severity.Info
        )
        st.OnScreenLogMessage(
            log_line,
            "PhobosNavigation",
            severity
        )

    if (
        iteration % CONFIG.radar_diagnostic_every_n == 0
        and radar_error is None
    ):
        track_summary = [
            {
                "id": track.identifier,
                "range_m": round(float(np.linalg.norm(
                    track.position_at(t_from_start) - loc
                )), 1),
                "age_s": round(max(
                    0.0, t_from_start - track.timestamp
                ), 2),
                "relative_speed_mps": round(float(np.linalg.norm(
                    track.velocity - vel
                )), 2),
                "debris_speed_mps": round(float(np.linalg.norm(
                    track.velocity
                )), 2),
                "hits": track.hits,
                "confirmed": track.confirmed,
            }
            for track in sorted(
                tracker.tracks,
                key=lambda item: np.linalg.norm(
                    item.position_at(t_from_start) - loc
                )
            )[:8]
        ]
        print(
            f"TRACK_DIAGNOSTIC t={t_from_start:.2f} "
            f"nearest_tracks={track_summary}",
            flush=True
        )
        print(
            "RADAR_DIAGNOSTIC "
            + format_radar_diagnostic(
                t_from_start,
                loc,
                vel,
                raw_debris,
                max_objects=5,
                radar_time_offset_s=radar_clock_offset_s,
            ),
            flush=True
        )

    iteration += 1
    loop_work_s = time.perf_counter() - loop_start_wall
    previous_loop_work_ms = 1000.0 * loop_work_s
    if loop_work_s > loop_period_s:
        loop_overrun_count += 1
    remaining_sleep = loop_period_s - loop_work_s
    if remaining_sleep > 0.0:
        time.sleep(remaining_sleep)

st.leave_sim()
