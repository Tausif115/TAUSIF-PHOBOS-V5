"""Frozen Phase 1 reference (commit 307cacb) of per-track threat loops.

threat_ttc_seconds is copied verbatim from Controller.py:300-353 and
nearest_track_distance reproduces Controller.py:550-558. Used only by
equivalence tests; never imported by the controller.
"""

import numpy as np


def threat_ttc_seconds(tracks, shuttle_position, shuttle_velocity, timestamp, config):
    """Earliest predicted CPA time that enters the uncertainty-expanded warning envelope."""
    best_ttc = float("inf")
    for track in tracks:
        try:
            age = max(0.0, float(timestamp) - float(track.timestamp))
            relative_position = (
                np.asarray(track.position_at(timestamp), dtype=float)
                - shuttle_position
            )
            relative_velocity = (
                np.asarray(track.velocity, dtype=float)
                - shuttle_velocity
            )
            speed_squared = float(np.dot(
                relative_velocity, relative_velocity
            ))
            if speed_squared <= 1.0e-9:
                continue
            t_cpa = -float(np.dot(
                relative_position, relative_velocity
            )) / speed_squared
            if t_cpa < 0.0 or t_cpa > config.horizon_s:
                continue
            separation = float(np.linalg.norm(
                relative_position + relative_velocity * t_cpa
            ))
            tentative = not bool(getattr(track, "confirmed", False))
            total_age = age + t_cpa
            velocity_uncertainty = (
                config.tentative_velocity_uncertainty_mps
                if tentative
                else config.confirmed_velocity_uncertainty_mps
            )
            envelope = (
                config.shuttle_radius_m
                + 0.5 * float(track.diameter)
                + config.clearance_margin_m
                + config.initial_uncertainty_m
                + (config.new_track_uncertainty_m if tentative else 0.0)
                + config.uncertainty_growth_mps * total_age
                + 0.5 * config.uncertainty_accel_mps2 * total_age ** 2
                + velocity_uncertainty * total_age
            )
            warning_radius = (
                envelope
                + config.replan_buffer_m
                + config.ttc_early_warning_margin_m
            )
            if separation <= warning_radius:
                best_ttc = min(best_ttc, t_cpa)
        except (ValueError, TypeError, AttributeError):
            continue
    return best_ttc


def nearest_track_distance(tracks, loc, t_from_start):
    return min(
        (
            float(np.linalg.norm(
                track.position_at(t_from_start) - loc
            ))
            for track in tracks
        ),
        default=float("inf")
    )
