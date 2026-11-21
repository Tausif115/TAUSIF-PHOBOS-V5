from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Mission guidance.
    goal_speed: float = 12.0
    velocity_gain: float = 0.90
    capture_radius_m: float = 50.0
    capture_standoff_m: float = 25.0
    braking_accel_fraction: float = 0.70
    terminal_speed_mps: float = 0.30
    capture_gain_per_s: float = 0.012
    capture_max_speed_mps: float = 0.30
    bias_fade_distance_m: float = 120.0

    # Radar/tracking. The 500-debris validation simconf publishes all 500 records.
    # The radar still refreshes records round-robin; the tracker must retain the
    # full set between updates. The controller reads but never changes radar data.
    radar_max_objects: int = 500
    radar_controller_poll_hz: float = 60.0
    radar_sync_min_samples: int = 3
    radar_sync_window: int = 21
    radar_stale_limit_s: float = 0.75
    radar_position_sigma_m: float = 2.0
    radar_closest_range_sigma_m: float = 2.0
    radar_diameter_sigma_fraction: float = 0.05
    # 500 records at the current 120 Hz radar loop are refreshed round-robin
    # (one object per loop, full cycle about 4.17 s); allow timing margin.
    track_max_age_s: float = 6.0
    # V5: the offset is locked after this many publication samples. Both
    # clocks derive from SimClock, so the true offset is a per-run constant.
    radar_sync_lock_samples: int = 9
    # Acquisition samples more than this above the minimum are loop overruns
    # (legitimate read-delay spread is about one controller period).
    radar_sync_outlier_window_s: float = 0.05
    radar_resync_threshold_s: float = 0.25
    radar_resync_consecutive: int = 3
    # Coarse (Euclidean) association prefilter; the statistical gate below
    # makes the final decision.
    association_gate_m: float = 25.0
    association_speed_gate_mps: float = 10.0
    diameter_gate_min_m: float = 2.0
    diameter_gate_fraction: float = 0.50
    # V5 constant-velocity Kalman tracker. Gate: chi-square, 4 dof
    # (3 position + 1 range-rate), p ~= 1e-4.
    association_chi2_gate: float = 23.5
    # Radar.py range-rate sigma is 0.02 m/s; 0.05 adds line-of-sight
    # direction error. A further geometric term is added per update.
    radar_range_rate_sigma_mps: float = 0.05
    radar_measurement_extra_sigma_m: float = 0.5
    # Unmodelled debris acceleration in the rotating Phobos frame (gravity
    # ~5e-3, Coriolis/centrifugal ~2e-3 m/s^2) as white-noise PSD (m^2/s^3).
    track_process_accel_psd: float = 1.0e-3
    # Prior on the unobserved tangential absolute debris velocity at track
    # birth (observed Phobos-frame debris speeds are ~0.7-5 m/s).
    track_initial_tangential_velocity_sigma_mps: float = 5.0
    # Radar.py refreshes each object once per NumDebris loop iterations
    # (1.67 s at 200, 4.17 s at 500, 120 Hz). Two genuine measurements of one
    # object can never be closer than this; anything closer is a cached copy
    # or a mis-association and must not update a track.
    track_min_update_interval_s: float = 0.25
    # Phase A: round-robin radar -> once the per-object revisit is measured,
    # a track may only be updated by a measurement at least this fraction of
    # a revisit after its previous update (0 disables; live ~3.5 s).
    track_min_revisit_fraction: float = 0.75
    # Track expiry follows the measured per-object revisit period:
    # max(track_max_age_s, factor * median revisit), capped.
    track_revisit_expiry_factor: float = 1.3  # legacy (Phase 1); superseded below
    # Phase A expiry: publication_factor x p99 measured revisit + margin.
    # Sweep + one publication interval (sweep/10) is the latest a valid
    # refresh can arrive; the margin covers radar jitter and controller
    # overruns and radar-process hitches (live: revisit 4.71 s -> expiry ~5.8 s).
    track_expiry_mode: str = "revisit_p99"   # "legacy" = Phase 1 rule (tests only)
    track_expiry_publication_factor: float = 1.1
    track_expiry_margin_s: float = 0.60
    track_revisit_min_samples: int = 50
    track_max_age_cap_s: float = 9.0
    # An unassociated observation may re-acquire an overdue track (one that
    # missed its expected revisit) instead of spawning a duplicate.
    track_reacquire_min_fraction: float = 0.75
    # Phase A: tracks born from Radar.py's initial snapshot (raw time 0.0)
    # are provisional and replaced by their first loop measurement.
    track_provisional_init_snapshot: bool = True
    radar_seen_key_retention_s: float = 15.0
    # V5 Phase 2b: at most this many unseen radar records are decoded per
    # tick (nearest first); the rest wait one tick. Steady state brings
    # ~50 per publication, so this only spreads the 500-record bootstrap.
    radar_max_new_records_per_tick: int = 100
    alpha: float = 0.70  # legacy (V4.3 alpha-beta); unused by the V5 tracker
    beta: float = 0.18  # legacy (V4.3 alpha-beta); unused by the V5 tracker
    confirmation_hits: int = 3
    # Physical bound on absolute debris speed in the Phobos frame (observed
    # <= ~5 m/s); V4.3 used 100 m/s, which permitted corrupted velocities.
    max_track_speed_mps: float = 15.0
    future_timestamp_tolerance_s: float = 0.25
    min_measurement_dt_s: float = 0.005
    # Must be >= spawned debris count so the tracker does not evict valid objects.
    max_tracks: int = 550
    state_history_age_s: float = 8.0

    # Receding-horizon prediction.
    horizon_s: float = 40.0
    step_s: float = 1.0
    control_latency_s: float = 0.10
    predicted_max_accel: float = 0.25

    # Conservative collision envelope.
    shuttle_radius_m: float = 3.0
    clearance_margin_m: float = 7.0
    initial_uncertainty_m: float = 4.0
    uncertainty_growth_mps: float = 0.70
    uncertainty_accel_mps2: float = 0.020
    confirmed_velocity_uncertainty_mps: float = 0.25
    tentative_velocity_uncertainty_mps: float = 1.00
    new_track_uncertainty_m: float = 8.0
    replan_buffer_m: float = 28.0
    minimum_safe_clearance_m: float = 8.0
    clearance_tradeoff_m: float = 5.0
    recovery_hysteresis_m: float = 12.0
    recovery_release_cycles: int = 3
    threat_candidate_threshold_m: float = 800.0
    threat_candidate_count: int = 3
    # Keep the real-time MPC workload bounded; planner selects the 128 lowest
    # predicted CPA-clearance tracks from the full retained track set.
    max_planning_tracks: int = 128
    # V5 Phase 2b exact bounded clearance evaluation (performance only; the
    # result is identical for any values): steps per bounding chunk and
    # seed units per candidate used to bound its minimum clearance.
    planner_bound_chunk_steps: int = 4
    planner_bound_seed_units: int = 8

    # V5 Phase 2b deadline-aware planner execution (scheduling.py). A
    # planner stage starts only if (time already spent in this tick + its
    # recent worst duration) fits control_tick_budget_s; the rest of the
    # 16.67 ms period is margin for SetCommand, logging and OS jitter.
    control_tick_budget_s: float = 0.014
    planner_max_stage_wait_ticks: int = 2     # <= 33 ms extra latency
    planner_max_job_age_s: float = 0.10       # = control_latency_s modelled by the planner
    planner_initial_stage_estimates_s: tuple = (0.006, 0.006)
    planner_stage_history: int = 20

    # Replanning cadence.
    planner_update_every_n: int = 3
    urgent_replan_every_n: int = 2
    critical_replan_every_n: int = 2
    urgent_replan_distance_m: float = 120.0
    critical_replan_distance_m: float = 45.0
    urgent_replan_ttc_s: float = 12.0
    critical_replan_ttc_s: float = 6.0
    ttc_early_warning_margin_m: float = 18.0
    clearance_early_warning_m: float = 12.0

    # Read-only thruster-authority estimation from RCSConfig.
    authority_safety_factor: float = 0.82
    jet_config_asset_path: str = "Local/Repos/STU4-Sim/Data/SpaceBuggyJetConfigNoMains.csv"
    authority_refresh_interval_s: float = 3.0
    authority_mass_change_fraction: float = 0.01
    authority_inertia_change_fraction: float = 0.01

    # Diagnostics.
    log_every_n: int = 30
    radar_diagnostic_every_n: int = 300

    # ------------------------------------------------------------------
    # V5.1 fly-by guidance (phobos_nav/ballistic.py). guidance_mode
    # "legacy_mpc" restores the V5.0 PredictivePlanner path unchanged.
    guidance_mode: str = "flyby_v51"
    v51_phobos_mu_m3_s2: float = 7.075e5          # fallback; live value read from Phobos
    v51_phobos_rate: float = 2.2803e-4            # synchronous rotation (rad/s)
    v51_frame_terms: bool = True                  # rotating-frame + Mars-tide terms
    v51_launch_speed_mps: float = 7.0
    v51_miss_m: float = 35.0                      # fly-by offset; latch at 50 m
    v51_aim_candidates: int = 8
    v51_aim_clearance_cap_m: float = 30.0
    v51_launch_horizon_s: float = 300.0
    v51_min_launch_time_s: float = 8.0
    v51_thrust_accel_mps2: float = 0.10           # torque-balanced authority (~0.11 live)
    v51_command_accel_mps2: float = 0.5           # any non-zero command saturates
    v51_done_tol_mps: float = 0.004               # ~2 ticks of impulse
    v51_first_trim_s: float = 20.0
    v51_trim_period_s: float = 40.0
    v51_trim_deadband_mps: float = 0.03
    v51_trim_min_range_m: float = 150.0
    v51_final_trim_range_m: float = 350.0
    v51_ca_limit_m: float = 42.0                  # predicted pass beyond this -> trim
    v51_check_period_s: float = 1.0
    v51_check_dt_s: float = 1.0
    v51_horizon_s: float = 90.0
    v51_base_margin_m: float = 6.0
    v51_margin_growth_mps: float = 0.07
    v51_tentative_growth_mps: float = 0.8
    v51_shuttle_radius_m: float = 3.0
    v51_relevance_radius_m: float = 250.0
    v51_avoid_magnitudes: tuple = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8)
    v51_lateral_cost: float = 3.0
    v51_speedup_cost: float = 0.8
    v51_slowdown_cost: float = 1.5
    v51_avoid_commit_s: float = 8.0
    v51_tracks_ready_fraction: float = 0.9
    # V5.2: live confirmed fraction plateaus at 93-94 % (tracker reacquisitions);
    # 0.95 was unreachable live (launch fell back to 30 s). Must hold for 1 s.
    v51_launch_confirmed_fraction: float = 0.90
    v51_launch_stable_s: float = 1.0
    v51_max_launch_wait_s: float = 30.0
    v51_action_horizon_s: float = 90.0
    v51_far_cost_cap: float = 1.0e9
    v51_terminal_ranges_m: tuple = (350.0, 180.0)
    v51_urgent_s: float = 25.0
    v51_reaim_cost: float = 1.0
    v51_fold_cost: float = 1.0
    # V5.2 (live run bde2f2fb fixes)
    # WAIT gravity hold: per-tick commands below ~dt*authority are delivered
    # proportionally by the official actuator (per-tick dv == command).
    v51_wait_hold: bool = True
    v51_wait_hold_speed_mps: float = 0.01     # small climb, away from the pad
    v51_wait_gain: float = 0.2                # fraction of velocity error per tick
    v51_wait_max_cmd: float = 0.0015          # stays in the proportional regime
    # V5.3 track reliability (replaces the V5.2 speed cap, disproved live):
    # range-rate residual tolerance (the tracker fuses range-rate, sigma
    # 0.02 m/s), margin growth = max(base, k x filter velocity sigma), and the
    # sigma used when a track has no covariance.
    v53_rr_tolerance_mps: float = 1.0
    v53_sigma_k: float = 1.0
    v53_default_vel_sigma_mps: float = 1.0
    v53_growth_cap_mps: float = 3.0           # cap on uncertainty-driven margin growth
    v53_confident_sigma_mps: float = 1.0      # velocity sigma above this = low confidence
    v53_lowconf_horizon_s: float = 12.0       # low-confidence tracks guard >= 2 radar revisits
    v53_lowconf_emergency_s: float = 8.0      # emergency (non-clearing) dodge for low-conf only if this close
    v53_nis_suspect: float = 13.28            # chi2(4 dof) 99 %: update innovation -> suspect track
    v53_vjump_mps: float = 1.5                # velocity jump between updates of a mature track -> suspect
    # WAIT phase (launch states, pad monitoring, launch gate)
    v53_wait_check_period_s: float = 1.0
    v53_wait_horizon_s: float = 60.0          # pad / emergency-lift-off look-ahead
    v53_launch_gate_s: float = 40.0           # launch corridor must be clear this long
    v53_safe_hold_max_s: float = 45.0         # SAFE_HOLD limit after first readiness
    v53_liftoff_window_margin_s: float = 15.0 # lift-off judged over time-to-conflict + this
    v53_pad_threat_s: float = 30.0            # pad conflicts act within this time-to-conflict
    # WAIT escape may only climb: candidate elevations above the local horizon
    # (local outward = radial from Phobos' centre) and the minimum accepted
    v53_escape_elevations_deg: tuple = (20.0, 45.0, 70.0)
    v53_escape_min_elevation_deg: float = 15.0
    v51_tentative_horizon_s: float = 25.0     # unreliable tracks guard only the near term
    v51_emergency_horizon_s: float = 20.0     # no blind (non-clearing) dodge for later conflicts
    v51_persist_checks: int = 2               # consecutive conflict checks before acting...
    v51_immediate_s: float = 8.0              # ...unless the conflict is this close
    v51_terminal_zone_m: float = 400.0
    v51_ca_min_m: float = 20.0



def _offline_overrides():
    """Offline tuning only: PHOBOS_CONFIG_OVERRIDES (JSON) is never set in STPro."""
    import json
    import os
    raw = os.environ.get("PHOBOS_CONFIG_OVERRIDES")
    if not raw:
        return {}
    data = json.loads(raw)
    return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}


CONFIG = Config(**_offline_overrides())
