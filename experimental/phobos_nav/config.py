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
    association_gate_m: float = 25.0
    association_speed_gate_mps: float = 10.0
    diameter_gate_min_m: float = 2.0
    diameter_gate_fraction: float = 0.50
    alpha: float = 0.70
    beta: float = 0.18
    confirmation_hits: int = 3
    max_track_speed_mps: float = 100.0
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


CONFIG = Config()
