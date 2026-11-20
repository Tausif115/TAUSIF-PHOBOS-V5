from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from .geometry import vec3, norm, unit, limit_norm


class StateHistory:
    """Short shuttle-state history for delayed radar measurement reconstruction."""

    def __init__(self, max_age_s=8.0, max_samples=1200):
        self.max_age_s = float(max_age_s)
        self.samples = deque(maxlen=int(max_samples))

    def __len__(self):
        return len(self.samples)

    def clear(self):
        self.samples.clear()

    def add(self, timestamp, position, velocity):
        timestamp = float(timestamp)
        position = vec3(position).copy()
        velocity = vec3(velocity).copy()

        if not np.isfinite(timestamp):
            return

        if self.samples and timestamp < self.samples[-1][0]:
            self.clear()

        if self.samples and abs(timestamp - self.samples[-1][0]) < 1.0e-9:
            self.samples[-1] = (timestamp, position, velocity)
        else:
            self.samples.append((timestamp, position, velocity))

        while (
            len(self.samples) > 2
            and timestamp - self.samples[0][0] > self.max_age_s
        ):
            self.samples.popleft()

    def sample(self, timestamp):
        if not self.samples:
            raise RuntimeError("StateHistory is empty")

        timestamp = float(timestamp)
        first_t, first_p, first_v = self.samples[0]
        last_t, last_p, last_v = self.samples[-1]

        if timestamp <= first_t:
            dt = timestamp - first_t
            return first_p + first_v * dt, first_v.copy()

        if timestamp >= last_t:
            dt = timestamp - last_t
            return last_p + last_v * dt, last_v.copy()

        previous = self.samples[0]
        for following in self.samples:
            if following[0] <= first_t:
                continue
            t0, p0, v0 = previous
            t1, p1, v1 = following
            if t0 <= timestamp <= t1:
                span = max(t1 - t0, 1.0e-9)
                fraction = (timestamp - t0) / span
                position = p0 + fraction * (p1 - p0)
                velocity = v0 + fraction * (v1 - v0)
                return position, velocity
            previous = following

        return last_p.copy(), last_v.copy()


@dataclass
class DebrisTrack:
    identifier: int
    position: np.ndarray
    velocity: np.ndarray
    diameter: float
    timestamp: float
    hits: int = 1
    confirmed: bool = False
    range_rate: float = 0.0

    def position_at(self, timestamp):
        dt = float(timestamp) - self.timestamp
        return self.position + self.velocity * dt


class DebrisTracker:
    """One-to-one alpha-beta tracker with timestamp-aware duplicate rejection."""

    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_identifier = 1
        self.last_timestamp = None
        self._seen_measurements = {}
        self.last_update_stats = {}

    def reset(self):
        self.tracks.clear()
        self.next_identifier = 1
        self.last_timestamp = None
        self._seen_measurements.clear()
        self.last_update_stats = {}

    @staticmethod
    def _measurement_key(position, diameter, timestamp):
        rounded = np.round(vec3(position), 1)
        return (
            round(float(timestamp), 3),
            round(float(rounded[0]), 1),
            round(float(rounded[1]), 1),
            round(float(rounded[2]), 1),
            round(float(diameter), 1)
        )

    def _record_stats(
        self, observations, accepted, duplicate_count, stale_count,
        future_count, invalid_count, too_soon_count, matched_count,
        created_count, observed_ages, fresh_ages
    ):
        self.last_update_stats = {
            "raw": len(observations),
            "decoded": len(observations),
            "accepted": len(accepted),
            "fresh": len(accepted),
            "duplicates": duplicate_count,
            "stale": stale_count,
            "future": future_count,
            "invalid": invalid_count,
            "too_soon": too_soon_count,
            "matched": matched_count,
            "created": created_count,
            # Age over all finite decoded measurements, including cached rows.
            "age_min": min(observed_ages) if observed_ages else float("nan"),
            "age_max": max(observed_ages) if observed_ages else float("nan"),
            # Age over unique, usable measurements only.
            "fresh_age_min": min(fresh_ages) if fresh_ages else float("nan"),
            "fresh_age_max": max(fresh_ages) if fresh_ages else float("nan")
        }

    def update(
        self,
        observations,
        timestamp,
        shuttle_position,
        shuttle_velocity
    ):
        cfg = self.config
        timestamp = float(timestamp)

        if not np.isfinite(timestamp):
            return self.tracks

        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            self.reset()

        self.last_timestamp = timestamp
        shuttle_position = vec3(shuttle_position)
        shuttle_velocity = vec3(shuttle_velocity)

        # Prune by the timestamp of the last genuine measurement.
        self.tracks = [
            track for track in self.tracks
            if -cfg.future_timestamp_tolerance_s
            <= timestamp - track.timestamp
            <= cfg.track_max_age_s
        ]

        valid = []
        observed_ages = []
        fresh_ages = []
        stale_count = 0
        future_count = 0
        duplicate_count = 0
        invalid_count = 0
        too_soon_count = 0

        for obs in observations:
            try:
                position = vec3(obs.position)
                diameter = float(obs.diameter)
                range_rate = float(obs.range_rate)
                obs_time_value = getattr(obs, "timestamp", None)
                obs_time = (
                    timestamp if obs_time_value is None
                    else float(obs_time_value)
                )

                if not np.all(np.isfinite(position)):
                    invalid_count += 1
                    continue
                if (
                    not np.isfinite(diameter)
                    or diameter < 0.0
                    or not np.isfinite(range_rate)
                    or not np.isfinite(obs_time)
                ):
                    invalid_count += 1
                    continue

                age = timestamp - obs_time
                observed_ages.append(max(0.0, age))

                if age < -cfg.future_timestamp_tolerance_s:
                    future_count += 1
                    continue
                if age > cfg.track_max_age_s:
                    stale_count += 1
                    continue

                key = self._measurement_key(position, diameter, obs_time)
                if key in self._seen_measurements:
                    duplicate_count += 1
                    continue

                self._seen_measurements[key] = timestamp
                valid.append(
                    (obs, position, diameter, range_rate, obs_time, key)
                )
                fresh_ages.append(max(0.0, age))

            except (TypeError, ValueError, AttributeError):
                invalid_count += 1
                continue

        candidate_pairs = []
        if self.tracks and valid:
            observation_positions = np.asarray(
                [item[1] for item in valid], dtype=float
            )
            observation_diameters = np.asarray(
                [item[2] for item in valid], dtype=float
            )
            observation_times = np.asarray(
                [item[4] for item in valid], dtype=float
            )

            invalid_cost = 1.0e9
            cost_matrix = np.full(
                (len(self.tracks), len(valid)),
                invalid_cost,
                dtype=float
            )

            for ti, track in enumerate(self.tracks):
                dt_values = observation_times - track.timestamp
                predicted = (
                    track.position[None, :]
                    + track.velocity[None, :] * dt_values[:, None]
                )
                distances = np.linalg.norm(
                    observation_positions - predicted, axis=1
                )
                diameter_differences = np.abs(
                    observation_diameters - track.diameter
                )
                diameter_gates = np.maximum(
                    cfg.diameter_gate_min_m,
                    cfg.diameter_gate_fraction
                    * np.maximum(observation_diameters, track.diameter)
                )
                distance_gates = (
                    cfg.association_gate_m
                    + cfg.association_speed_gate_mps * np.abs(dt_values)
                )

                valid_pairs = (
                    (distances <= distance_gates)
                    & (diameter_differences <= diameter_gates)
                )
                cost_matrix[ti, valid_pairs] = (
                    distances[valid_pairs]
                    + 0.35 * diameter_differences[valid_pairs]
                )

            if linear_sum_assignment is not None:
                row_indices, column_indices = linear_sum_assignment(cost_matrix)
                candidate_pairs = [
                    (float(cost_matrix[ti, oi]), int(ti), int(oi))
                    for ti, oi in zip(row_indices, column_indices)
                    if cost_matrix[ti, oi] < invalid_cost
                ]
            else:
                candidate_pairs = [
                    (float(cost_matrix[ti, oi]), ti, oi)
                    for ti in range(cost_matrix.shape[0])
                    for oi in range(cost_matrix.shape[1])
                    if cost_matrix[ti, oi] < invalid_cost
                ]
                candidate_pairs.sort(key=lambda item: item[0])

        assigned_tracks = set()
        assigned_observations = set()
        matched_count = 0

        for _, ti, oi in candidate_pairs:
            if ti in assigned_tracks or oi in assigned_observations:
                continue

            track = self.tracks[ti]
            obs, position, diameter, range_rate, obs_time, key = valid[oi]
            dt = obs_time - track.timestamp

            assigned_tracks.add(ti)
            assigned_observations.add(oi)

            # 5 ms allows distinct 60 Hz radar samples (16.7 ms apart) to update.
            # Same-time/cached measurements never move a track backward.
            if dt <= cfg.min_measurement_dt_s:
                too_soon_count += 1
                continue

            predicted = track.position + track.velocity * dt
            residual = position - predicted

            track.position = predicted + cfg.alpha * residual
            track.velocity = (
                track.velocity + cfg.beta * residual / dt
            )
            # The radar also measures line-of-sight relative speed. Fuse that
            # scalar observation into the radial component of the absolute
            # debris velocity; leave tangential components to the alpha-beta
            # position history.
            relative_measurement = getattr(obs, "relative_position", None)
            if relative_measurement is None:
                relative_measurement = position - shuttle_position
            radial = unit(relative_measurement)
            obs_shuttle_velocity = getattr(obs, "shuttle_velocity", None)
            if obs_shuttle_velocity is None:
                obs_shuttle_velocity = shuttle_velocity
            measured_radial_velocity = (
                vec3(obs_shuttle_velocity) + range_rate * radial
            )
            radial_error = float(np.dot(
                measured_radial_velocity - track.velocity, radial
            ))
            track.velocity = track.velocity + 0.35 * radial_error * radial
            track.velocity = limit_norm(
                track.velocity, cfg.max_track_speed_mps
            )
            track.diameter = diameter
            track.range_rate = range_rate
            track.timestamp = obs_time
            track.hits += 1
            track.confirmed = track.hits >= cfg.confirmation_hits
            matched_count += 1

        created_count = 0
        for oi, item in enumerate(valid):
            if oi in assigned_observations:
                continue

            obs, position, diameter, range_rate, obs_time, key = item

            relative_position = getattr(obs, "relative_position", None)
            if relative_position is None:
                relative_position = position - shuttle_position
            else:
                relative_position = vec3(relative_position)

            radial_direction = unit(relative_position)
            obs_shuttle_velocity = getattr(obs, "shuttle_velocity", None)
            if obs_shuttle_velocity is None:
                obs_shuttle_velocity = shuttle_velocity
            else:
                obs_shuttle_velocity = vec3(obs_shuttle_velocity)

            # First detection only observes radial relative velocity.
            velocity_estimate = (
                obs_shuttle_velocity + range_rate * radial_direction
            )

            if len(self.tracks) >= cfg.max_tracks:
                # Prefer evicting the oldest tentative track. If all are
                # confirmed, evict the oldest measurement to bound CPU/memory.
                tentative_indices = [
                    i for i, track in enumerate(self.tracks)
                    if not track.confirmed
                ]
                pool = tentative_indices or list(range(len(self.tracks)))
                evict_index = min(
                    pool, key=lambda i: self.tracks[i].timestamp
                )
                self.tracks.pop(evict_index)

            self.tracks.append(
                DebrisTrack(
                    identifier=self.next_identifier,
                    position=position.copy(),
                    velocity=velocity_estimate.copy(),
                    diameter=diameter,
                    timestamp=obs_time,
                    hits=1,
                    confirmed=False,
                    range_rate=range_rate
                )
            )
            self.next_identifier += 1
            created_count += 1

        cutoff = timestamp - cfg.track_max_age_s - 1.0
        self._seen_measurements = {
            key: seen_at
            for key, seen_at in self._seen_measurements.items()
            if seen_at >= cutoff
        }

        self._record_stats(
            observations, valid, duplicate_count, stale_count,
            future_count, invalid_count, too_soon_count, matched_count,
            created_count, observed_ages, fresh_ages
        )
        return self.tracks
