"""Vectorized per-track threat geometry (V5 Phase 2).

Replaces per-track Python loops that ran every controller tick over all
retained tracks (up to 550):
  * nearest predicted track distance   (V4.3 Controller.py:550-558)
  * earliest warning-envelope CPA time (V4.3 Controller.py:300-353)
and provides the shared array extraction used by the planner shortlist.

The mathematics is unchanged; tests/test_v5_performance_equivalence.py
compares every function against frozen copies of the Phase 1 code.
"""

from dataclasses import dataclass
import numpy as np

from .tracking import DebrisTrack, velocity_sigma

_MISSING = object()


@dataclass
class TrackStateArrays:
    """Usable tracks predicted to one timestamp, as aligned arrays."""
    positions: np.ndarray        # (N,3) position at `timestamp`
    velocities: np.ndarray       # (N,3)
    ages: np.ndarray             # (N,)  max(0, timestamp - track.timestamp)
    diameters: np.ndarray        # (N,)
    confirmed: np.ndarray        # (N,)  getattr(track, "confirmed", False)
    confirmed_or_hits: np.ndarray  # (N,) getattr(track, "confirmed", hits >= 3)
    identifiers: np.ndarray      # (N,)  int
    timestamp: float
    # V4.3 controller loops required track.position_at (tracks without it
    # were skipped); the planner extrapolated them. Kept for exact parity.
    has_position_at: np.ndarray = None  # (N,) bool
    # V5.3 (None when unavailable): radar-measured absolute radial velocity and
    # line of sight of each track's last update, 1-sigma velocity uncertainty
    # from the filter covariance, and the tracker's speed-clamp flag.
    radial_meas: np.ndarray = None      # (N,)
    los: np.ndarray = None              # (N,3)
    vel_sigma: np.ndarray = None        # (N,)
    clamped: np.ndarray = None          # (N,) bool
    suspect: np.ndarray = None          # (N,) bool (V5.3 per-update evidence)

    def __len__(self):
        return int(self.positions.shape[0])


def _empty(timestamp):
    return TrackStateArrays(
        positions=np.zeros((0, 3)), velocities=np.zeros((0, 3)),
        ages=np.zeros(0), diameters=np.zeros(0),
        confirmed=np.zeros(0, dtype=bool), confirmed_or_hits=np.zeros(0, dtype=bool),
        identifiers=np.zeros(0, dtype=int), timestamp=float(timestamp),
        has_position_at=np.zeros(0, dtype=bool),
    )


def track_state_arrays(tracks, timestamp):
    """Extract usable tracks at ``timestamp``.

    A track is usable when its predicted position, velocity, age and
    diameter are finite and diameter >= 0 (the V4.3 planner criteria).
    DebrisTrack positions use the closed form position + velocity * dt,
    identical to DebrisTrack.position_at; other objects keep their own
    position_at (or position + velocity * dt when they have none).
    """
    timestamp = float(timestamp)
    if not tracks:
        return _empty(timestamp)

    if all(type(track) is DebrisTrack for track in tracks):
        try:
            base = np.asarray([track.position for track in tracks], dtype=float).reshape(-1, 3)
            velocities = np.asarray([track.velocity for track in tracks], dtype=float).reshape(-1, 3)
            times = np.asarray([track.timestamp for track in tracks], dtype=float)
            diameters = np.asarray([track.diameter for track in tracks], dtype=float)
            confirmed = np.asarray([bool(track.confirmed) for track in tracks], dtype=bool)
            identifiers = np.asarray([int(track.identifier) for track in tracks], dtype=int)
            positions = base + velocities * (timestamp - times)[:, None]
            confirmed_or_hits = confirmed.copy()
        except (TypeError, ValueError):
            return _general_track_state_arrays(tracks, timestamp)
    else:
        return _general_track_state_arrays(tracks, timestamp)

    ages = np.maximum(0.0, timestamp - times)
    nan3 = (float("nan"),) * 3
    extra = dict(
        radial_meas=np.asarray([track.radial_velocity_meas for track in tracks], dtype=float),
        los=np.asarray([nan3 if track.los is None else track.los for track in tracks],
                       dtype=float).reshape(-1, 3),
        vel_sigma=np.asarray([track.vel_sigma for track in tracks], dtype=float),
        clamped=np.asarray([bool(track.clamped) for track in tracks], dtype=bool),
        suspect=np.asarray([bool(track.suspect) for track in tracks], dtype=bool),
    )
    return _filtered(positions, velocities, ages, diameters, confirmed,
                     confirmed_or_hits, identifiers, timestamp,
                     np.ones(len(tracks), dtype=bool), **extra)


_velocity_sigma = velocity_sigma     # V5.3 diagnostics alias (Controller.py)


def _general_track_state_arrays(tracks, timestamp):
    rows = []
    for track in tracks:
        try:
            track_time = float(track.timestamp)
            velocity = np.asarray(track.velocity, dtype=float).reshape(3)
            if hasattr(track, "position_at"):
                position = np.asarray(track.position_at(timestamp), dtype=float).reshape(3)
            else:
                position = (
                    np.asarray(track.position, dtype=float).reshape(3)
                    + velocity * (timestamp - track_time)
                )
            confirmed_attr = getattr(track, "confirmed", _MISSING)
            has_confirmed = confirmed_attr is not _MISSING
            rows.append((
                position, velocity, max(0.0, timestamp - track_time),
                float(track.diameter),
                bool(confirmed_attr) if has_confirmed else False,
                bool(confirmed_attr) if has_confirmed
                else getattr(track, "hits", 0) >= 3,
                int(getattr(track, "identifier", -1)),
                hasattr(track, "position_at"),
            ))
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
    if not rows:
        return _empty(timestamp)
    return _filtered(
        np.asarray([r[0] for r in rows], dtype=float),
        np.asarray([r[1] for r in rows], dtype=float),
        np.asarray([r[2] for r in rows], dtype=float),
        np.asarray([r[3] for r in rows], dtype=float),
        np.asarray([r[4] for r in rows], dtype=bool),
        np.asarray([r[5] for r in rows], dtype=bool),
        np.asarray([r[6] for r in rows], dtype=int),
        timestamp,
        np.asarray([r[7] for r in rows], dtype=bool),
    )


def _filtered(positions, velocities, ages, diameters, confirmed,
              confirmed_or_hits, identifiers, timestamp, has_position_at, **extra):
    usable = (
        np.all(np.isfinite(positions), axis=1)
        & np.all(np.isfinite(velocities), axis=1)
        & np.isfinite(ages)
        & np.isfinite(diameters)
        & (diameters >= 0.0)
    )
    return TrackStateArrays(
        positions=positions[usable], velocities=velocities[usable],
        ages=ages[usable], diameters=diameters[usable],
        confirmed=confirmed[usable], confirmed_or_hits=confirmed_or_hits[usable],
        identifiers=identifiers[usable], timestamp=float(timestamp),
        has_position_at=has_position_at[usable],
        **{name: (None if value is None else value[usable]) for name, value in extra.items()},
    )


def _controller_subset(state):
    """Tracks the V4.3 controller loops evaluated (those with position_at)."""
    mask = state.has_position_at
    if mask is None or bool(np.all(mask)):
        return state.positions, state.velocities, state.ages, state.diameters, state.confirmed
    return (state.positions[mask], state.velocities[mask], state.ages[mask],
            state.diameters[mask], state.confirmed[mask])


def nearest_track_distance(state, shuttle_position):
    """Smallest distance from the shuttle to any usable predicted track."""
    positions = _controller_subset(state)[0]
    if positions.shape[0] == 0:
        return float("inf")
    offsets = positions - np.asarray(shuttle_position, dtype=float).reshape(1, 3)
    return float(np.min(np.sqrt(np.einsum("ij,ij->i", offsets, offsets))))


def threat_ttc_seconds(state, shuttle_position, shuttle_velocity, config):
    """Earliest predicted CPA time inside the uncertainty-expanded warning envelope."""
    positions, velocities, ages, diameters, confirmed = _controller_subset(state)
    if positions.shape[0] == 0:
        return float("inf")
    relative_position = positions - np.asarray(shuttle_position, dtype=float).reshape(1, 3)
    relative_velocity = velocities - np.asarray(shuttle_velocity, dtype=float).reshape(1, 3)
    speed_squared = np.einsum("ij,ij->i", relative_velocity, relative_velocity)
    moving = speed_squared > 1.0e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cpa = np.where(
            moving,
            -np.einsum("ij,ij->i", relative_position, relative_velocity)
            / np.where(moving, speed_squared, 1.0),
            np.nan,
        )
    in_horizon = moving & (t_cpa >= 0.0) & (t_cpa <= float(config.horizon_s))
    if not np.any(in_horizon):
        return float("inf")
    t_cpa = t_cpa[in_horizon]
    closest = relative_position[in_horizon] + relative_velocity[in_horizon] * t_cpa[:, None]
    separation = np.sqrt(np.einsum("ij,ij->i", closest, closest))
    tentative = ~confirmed[in_horizon]
    total_age = ages[in_horizon] + t_cpa
    velocity_uncertainty = np.where(
        tentative,
        float(config.tentative_velocity_uncertainty_mps),
        float(config.confirmed_velocity_uncertainty_mps),
    )
    envelope = (
        float(config.shuttle_radius_m)
        + 0.5 * diameters[in_horizon]
        + float(config.clearance_margin_m)
        + float(config.initial_uncertainty_m)
        + np.where(tentative, float(config.new_track_uncertainty_m), 0.0)
        + float(config.uncertainty_growth_mps) * total_age
        + 0.5 * float(config.uncertainty_accel_mps2) * total_age ** 2
        + velocity_uncertainty * total_age
    )
    warning_radius = (
        envelope + float(config.replan_buffer_m) + float(config.ttc_early_warning_margin_m)
    )
    hits = separation <= warning_radius
    return float(np.min(t_cpa[hits])) if np.any(hits) else float("inf")
