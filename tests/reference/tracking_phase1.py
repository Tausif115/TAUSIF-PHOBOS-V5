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
    # V5: 6x6 state covariance [position, velocity] in Phobos axes.
    covariance: np.ndarray | None = None

    def position_at(self, timestamp):
        dt = float(timestamp) - self.timestamp
        return self.position + self.velocity * dt


_I3 = np.eye(3, dtype=float)


def _process_noise(dt, psd):
    """White-noise-acceleration process covariance for a constant-velocity state."""
    dt = abs(float(dt))
    q = np.zeros((6, 6), dtype=float)
    q[:3, :3] = psd * dt ** 3 / 3.0 * _I3
    q[:3, 3:] = psd * dt ** 2 / 2.0 * _I3
    q[3:, :3] = q[:3, 3:]
    q[3:, 3:] = psd * dt * _I3
    return q


class DebrisTracker:
    """One-to-one constant-velocity Kalman tracker for round-robin radar data.

    V5 corrections to the V4.3 alpha-beta tracker:
      * cached radar records are identified by their immutable raw values
        (``obs.raw_key``), independent of the estimated clock offset;
      * no track is updated from two measurements closer than
        ``track_min_update_interval_s`` (no 1/dt velocity amplification);
      * a Kalman filter fuses position and range-rate with per-track
        covariance, so gating and velocity gain follow the real uncertainty;
      * replayed or out-of-order copies are dropped instead of spawning
        tracks; overdue tracks are re-acquired instead of duplicated;
      * track expiry follows the measured per-object revisit period;
      * capacity evictions are explicit and counted.
    """

    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_identifier = 1
        self.last_timestamp = None
        self._seen_measurements = {}
        self._revisit_samples = deque(maxlen=500)
        self._last_prune_s = -float("inf")
        self.last_update_stats = {}
        self.cumulative = {
            "evicted": 0, "evicted_confirmed": 0, "reacquired": 0,
            "expired": 0, "replays": 0, "velocity_clamped": 0,
        }

    def reset(self):
        self.tracks.clear()
        self.next_identifier = 1
        self.last_timestamp = None
        self._seen_measurements.clear()
        self._revisit_samples.clear()
        self._last_prune_s = -float("inf")
        self.last_update_stats = {}

    # ------------------------------------------------------------------
    # Configuration helpers (getattr keeps duck-typed test configs working)
    # ------------------------------------------------------------------
    def _cfg(self, name, default):
        return getattr(self.config, name, default)

    @staticmethod
    def _measurement_key(position, diameter, timestamp):
        """Legacy identity for observations that carry no raw radar record."""
        rounded = np.round(vec3(position), 1)
        return (
            round(float(timestamp), 3),
            round(float(rounded[0]), 1),
            round(float(rounded[1]), 1),
            round(float(rounded[2]), 1),
            round(float(diameter), 1)
        )

    def revisit_period_s(self):
        """Median measured interval between genuine updates of one object."""
        if len(self._revisit_samples) < int(self._cfg("track_revisit_min_samples", 50)):
            return float("nan")
        return float(np.median(np.asarray(self._revisit_samples, dtype=float)))

    def expiry_age_s(self):
        base = float(self.config.track_max_age_s)
        revisit = self.revisit_period_s()
        if not np.isfinite(revisit):
            return base
        return float(min(
            max(base, float(self._cfg("track_revisit_expiry_factor", 1.3)) * revisit),
            max(base, float(self._cfg("track_max_age_cap_s", 9.0))),
        ))

    def _default_position_covariance(self):
        sigma = float(self._cfg("radar_position_sigma_m", 2.0))
        extra = float(self._cfg("radar_measurement_extra_sigma_m", 0.0))
        return (sigma ** 2 + extra ** 2) * _I3

    def _measurement_covariance(self, obs, relative_position):
        """3x3 position covariance: cross-range per-axis variance, fused
        (smaller) line-of-sight variance, plus an isotropic term for
        state-history interpolation and timing error."""
        explicit = getattr(obs, "measurement_covariance", None)
        if explicit is not None:
            return np.asarray(explicit, dtype=float).reshape(3, 3)
        cross = float(getattr(obs, "measurement_cross_variance", float("nan")))
        radial_variance = float(getattr(obs, "measurement_radial_variance", float("nan")))
        if not (np.isfinite(cross) and np.isfinite(radial_variance)):
            return self._default_position_covariance()
        extra = float(self._cfg("radar_measurement_extra_sigma_m", 0.0)) ** 2
        radial = unit(relative_position)
        return (cross + extra) * _I3 + (radial_variance - cross) * np.outer(radial, radial)

    def _range_rate_variance(self, relative_position, relative_velocity_estimate_sq_tangential):
        """Range-rate variance: radar noise plus line-of-sight direction error."""
        sigma_rr = float(self._cfg("radar_range_rate_sigma_mps", 0.05))
        sigma_pos = float(self._cfg("radar_position_sigma_m", 2.0))
        distance = max(norm(relative_position), 1.0)
        return sigma_rr ** 2 + (sigma_pos / distance) ** 2 * max(
            relative_velocity_estimate_sq_tangential, 0.0
        )

    def _initial_state(self, position, relative_position, range_rate, shuttle_velocity, position_cov):
        """Birth state: measured position, radial velocity only, tangential prior 0."""
        radial = unit(relative_position)
        # Absolute radial velocity = shuttle radial velocity + measured range-rate.
        # The absolute tangential velocity is unobserved; its prior mean is
        # zero (debris is slow in the Phobos frame), NOT the shuttle velocity.
        radial_speed = float(np.dot(shuttle_velocity, radial)) + float(range_rate)
        velocity = radial_speed * radial
        sigma_t = float(self._cfg("track_initial_tangential_velocity_sigma_mps", 5.0))
        shuttle_tangential = shuttle_velocity - float(np.dot(shuttle_velocity, radial)) * radial
        var_rr = self._range_rate_variance(
            relative_position,
            float(np.dot(shuttle_tangential, shuttle_tangential)) + 2.0 * sigma_t ** 2,
        )
        radial_outer = np.outer(radial, radial)
        covariance = np.zeros((6, 6), dtype=float)
        covariance[:3, :3] = position_cov
        covariance[3:, 3:] = sigma_t ** 2 * (_I3 - radial_outer) + var_rr * radial_outer
        return velocity, covariance

    # ------------------------------------------------------------------
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
        min_interval = float(self._cfg("track_min_update_interval_s", 0.25))
        psd = float(self._cfg("track_process_accel_psd", 1.0e-3))
        chi2_gate = float(self._cfg("association_chi2_gate", 23.5))
        expiry = self.expiry_age_s()
        revisit = self.revisit_period_s()

        valid = []
        observed_ages = []
        fresh_ages = []
        stale_count = 0
        future_count = 0
        duplicate_count = 0
        invalid_count = 0

        for obs in observations:
            try:
                raw_key = getattr(obs, "raw_key", None)
                if raw_key is not None and raw_key in self._seen_measurements:
                    # Cached copy of an already-processed radar record.
                    self._seen_measurements[raw_key] = timestamp
                    duplicate_count += 1
                    continue

                position = vec3(obs.position)
                diameter = float(obs.diameter)
                range_rate = float(obs.range_rate)
                obs_time_value = getattr(obs, "timestamp", None)
                obs_time = (
                    timestamp if obs_time_value is None
                    else float(obs_time_value)
                )
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

                key = raw_key if raw_key is not None else self._measurement_key(
                    position, diameter, obs_time
                )
                if key in self._seen_measurements:
                    self._seen_measurements[key] = timestamp
                    duplicate_count += 1
                    continue
                self._seen_measurements[key] = timestamp

                if age > expiry:
                    stale_count += 1
                    continue

                relative_position = getattr(obs, "relative_position", None)
                relative_position = (
                    position - shuttle_position if relative_position is None
                    else vec3(relative_position)
                )
                obs_shuttle_velocity = getattr(obs, "shuttle_velocity", None)
                obs_shuttle_velocity = (
                    shuttle_velocity if obs_shuttle_velocity is None
                    else vec3(obs_shuttle_velocity)
                )
                position_cov = self._measurement_covariance(obs, relative_position)
                valid.append({
                    "position": position,
                    "diameter": diameter,
                    "range_rate": range_rate,
                    "time": obs_time,
                    "relative_position": relative_position,
                    "radial": unit(relative_position),
                    "shuttle_velocity": obs_shuttle_velocity,
                    "cov": position_cov,
                })
                fresh_ages.append(max(0.0, age))
            except (TypeError, ValueError, AttributeError):
                invalid_count += 1
                continue

        assigned = []          # (track index, observation index, pair data)
        replay = set()         # observation indices consistent with a newer/same-time track state
        coarse_candidates = {}  # observation index -> list of (distance, track index, dt)

        if self.tracks and valid:
            assigned, replay, coarse_candidates = self._associate(
                valid, min_interval, psd, chi2_gate
            )

        matched_count = 0
        clamped_count = 0
        max_speed = float(cfg.max_track_speed_mps)
        updated_tracks = set()
        for ti, oi, pair in assigned:
            track = self.tracks[ti]
            item = valid[oi]
            dt = item["time"] - track.timestamp
            self._kf_update(track, item, dt, psd, pair)
            speed = norm(track.velocity)
            if speed > max_speed:
                track.velocity = limit_norm(track.velocity, max_speed)
                clamped_count += 1
            track.diameter = item["diameter"]
            track.range_rate = item["range_rate"]
            track.timestamp = item["time"]
            track.hits += 1
            track.confirmed = track.hits >= cfg.confirmation_hits
            self._revisit_samples.append(dt)
            updated_tracks.add(ti)
            matched_count += 1

        assigned_observations = {oi for _, oi, _ in assigned}
        created_count = 0
        reacquired_count = 0
        evicted_count = 0
        evicted_confirmed = 0
        replay_count = 0
        reacquire_after = max(
            min_interval,
            float(self._cfg("track_reacquire_min_fraction", 0.75))
            * (revisit if np.isfinite(revisit) else float(cfg.track_max_age_s) / 1.3),
        )

        new_tracks = []
        evicted = set()
        for oi, item in enumerate(valid):
            if oi in assigned_observations:
                continue
            if oi in replay:
                # Same measurement (or an older one) of an already-updated
                # object: incorporating or spawning from it would be wrong.
                replay_count += 1
                continue

            velocity, covariance = self._initial_state(
                item["position"], item["relative_position"], item["range_rate"],
                item["shuttle_velocity"], item["cov"],
            )

            # Re-acquire the nearest overdue track instead of creating a
            # duplicate that would leave the old track as a ghost.
            reacquire_index = None
            for distance, ti, dt in sorted(coarse_candidates.get(oi, [])):
                if ti in updated_tracks or ti in evicted or dt < reacquire_after:
                    continue
                reacquire_index = ti
                break
            if reacquire_index is not None:
                track = self.tracks[reacquire_index]
                track.position = item["position"].copy()
                track.velocity = velocity
                track.covariance = covariance
                track.diameter = item["diameter"]
                track.range_rate = item["range_rate"]
                track.timestamp = item["time"]
                track.hits = 1
                track.confirmed = False
                updated_tracks.add(reacquire_index)
                reacquired_count += 1
                continue

            if len(self.tracks) - len(evicted) + len(new_tracks) >= cfg.max_tracks:
                index = self._eviction_index(
                    timestamp, revisit, updated_tracks | evicted
                )
                if index is None:
                    continue
                evicted.add(index)
                evicted_count += 1
                evicted_confirmed += int(self.tracks[index].confirmed)

            new_tracks.append(
                DebrisTrack(
                    identifier=self.next_identifier,
                    position=item["position"].copy(),
                    velocity=velocity,
                    diameter=item["diameter"],
                    timestamp=item["time"],
                    hits=1,
                    confirmed=False,
                    range_rate=item["range_rate"],
                    covariance=covariance,
                )
            )
            self.next_identifier += 1
            created_count += 1

        if evicted or new_tracks:
            self.tracks = [
                track for index, track in enumerate(self.tracks)
                if index not in evicted
            ] + new_tracks

        # Expire AFTER association: after a controller stall, a track's fresh
        # measurement may be in this very batch and must be used, not lost.
        before = len(self.tracks)
        self.tracks = [
            track for track in self.tracks
            if -cfg.future_timestamp_tolerance_s
            <= timestamp - track.timestamp
            <= expiry
        ]
        expired_count = before - len(self.tracks)
        self.cumulative["expired"] += expired_count

        # Seen-record keys are refreshed on every sighting, so a key is only
        # forgotten after its record has left the radar array for longer than
        # the retention period.
        retention = float(self._cfg("radar_seen_key_retention_s", 15.0))
        if timestamp - self._last_prune_s >= 1.0:
            cutoff = timestamp - max(retention, expiry + 1.0)
            self._seen_measurements = {
                key: seen_at
                for key, seen_at in self._seen_measurements.items()
                if seen_at >= cutoff
            }
            self._last_prune_s = timestamp

        self.cumulative["evicted"] += evicted_count
        self.cumulative["evicted_confirmed"] += evicted_confirmed
        self.cumulative["reacquired"] += reacquired_count
        self.cumulative["replays"] += replay_count
        self.cumulative["velocity_clamped"] += clamped_count

        self.last_update_stats = {
            "raw": len(observations),
            "decoded": len(observations),
            "accepted": len(valid),
            "fresh": len(valid),
            "duplicates": duplicate_count,
            "stale": stale_count,
            "future": future_count,
            "invalid": invalid_count,
            "too_soon": replay_count,
            "matched": matched_count,
            "created": created_count,
            "reacquired": reacquired_count,
            "evicted": evicted_count,
            "evicted_confirmed": evicted_confirmed,
            "expired": expired_count,
            "velocity_clamped": clamped_count,
            "revisit_s": revisit,
            "expiry_s": expiry,
            "age_min": min(observed_ages) if observed_ages else float("nan"),
            "age_max": max(observed_ages) if observed_ages else float("nan"),
            "fresh_age_min": min(fresh_ages) if fresh_ages else float("nan"),
            "fresh_age_max": max(fresh_ages) if fresh_ages else float("nan"),
        }
        return self.tracks

    # ------------------------------------------------------------------
    def _eviction_index(self, timestamp, revisit, protected):
        """Least valuable track: overdue first, then fewest hits, then oldest."""
        overdue_after = (
            revisit if np.isfinite(revisit)
            else float(self.config.track_max_age_s) / 1.3
        )
        best = None
        for index, track in enumerate(self.tracks):
            if index in protected:
                continue
            age = timestamp - float(track.timestamp)
            key = (0 if age > overdue_after else 1, int(track.hits), -age)
            if best is None or key < best[0]:
                best = (key, index)
        return None if best is None else best[1]

    def _track_covariances(self):
        default = None
        covariances = []
        for track in self.tracks:
            covariance = track.covariance
            if covariance is None:
                if default is None:
                    default = np.zeros((6, 6), dtype=float)
                    default[:3, :3] = self._default_position_covariance()
                    sigma_t = float(self._cfg(
                        "track_initial_tangential_velocity_sigma_mps", 5.0
                    ))
                    default[3:, 3:] = sigma_t ** 2 * _I3
                covariance = default.copy()
                track.covariance = covariance
            covariances.append(covariance)
        return np.asarray(covariances, dtype=float)

    def _associate(self, valid, min_interval, psd, chi2_gate):
        cfg = self.config
        track_positions = np.asarray([t.position for t in self.tracks], dtype=float)
        track_velocities = np.asarray([t.velocity for t in self.tracks], dtype=float)
        track_times = np.asarray([t.timestamp for t in self.tracks], dtype=float)
        track_diameters = np.asarray([t.diameter for t in self.tracks], dtype=float)
        obs_positions = np.asarray([v["position"] for v in valid], dtype=float)
        obs_times = np.asarray([v["time"] for v in valid], dtype=float)
        obs_diameters = np.asarray([v["diameter"] for v in valid], dtype=float)

        # Coarse Euclidean + diameter prefilter over all (track, obs) pairs.
        dt_matrix = obs_times[None, :] - track_times[:, None]
        predicted = (
            track_positions[:, None, :]
            + track_velocities[:, None, :] * dt_matrix[:, :, None]
        )
        distances = np.linalg.norm(obs_positions[None, :, :] - predicted, axis=2)
        coarse_gate = cfg.association_gate_m + cfg.association_speed_gate_mps * np.abs(dt_matrix)
        diameter_gate = np.maximum(
            cfg.diameter_gate_min_m,
            cfg.diameter_gate_fraction * np.maximum(
                obs_diameters[None, :], track_diameters[:, None]
            ),
        )
        coarse = (
            (distances <= coarse_gate)
            & (np.abs(obs_diameters[None, :] - track_diameters[:, None]) <= diameter_gate)
        )
        track_index, obs_index = np.nonzero(coarse)

        coarse_candidates = {}
        for ti, oi in zip(track_index.tolist(), obs_index.tolist()):
            coarse_candidates.setdefault(oi, []).append(
                (float(distances[ti, oi]), ti, float(dt_matrix[ti, oi]))
            )
        if track_index.size == 0:
            return [], set(), coarse_candidates

        # Statistical (Mahalanobis) gate for the coarse candidates only.
        covariances = self._track_covariances()[track_index]
        dt = dt_matrix[track_index, obs_index]
        abs_dt = np.abs(dt)
        p_pp = (
            covariances[:, :3, :3]
            + dt[:, None, None] * (covariances[:, :3, 3:] + covariances[:, 3:, :3])
            + (dt ** 2)[:, None, None] * covariances[:, 3:, 3:]
            + (psd * abs_dt ** 3 / 3.0)[:, None, None] * _I3
        )
        p_vv = covariances[:, 3:, 3:] + (psd * abs_dt)[:, None, None] * _I3
        obs_covariances = np.asarray([valid[oi]["cov"] for oi in obs_index], dtype=float)
        innovation_cov = p_pp + obs_covariances
        residual = obs_positions[obs_index] - predicted[track_index, obs_index]
        solved = np.linalg.solve(innovation_cov, residual[:, :, None])[:, :, 0]
        d2_position = np.einsum("ki,ki->k", residual, solved)
        _, logdet = np.linalg.slogdet(innovation_cov)

        radial = np.asarray([valid[oi]["radial"] for oi in obs_index], dtype=float)
        shuttle_v = np.asarray([valid[oi]["shuttle_velocity"] for oi in obs_index], dtype=float)
        relative_positions = np.asarray(
            [valid[oi]["relative_position"] for oi in obs_index], dtype=float
        )
        range_rates = np.asarray([valid[oi]["range_rate"] for oi in obs_index], dtype=float)
        velocity_estimate = track_velocities[track_index]
        relative_velocity = velocity_estimate - shuttle_v
        radial_component = np.einsum("ki,ki->k", relative_velocity, radial)
        tangential_sq = np.maximum(
            np.einsum("ki,ki->k", relative_velocity, relative_velocity) - radial_component ** 2,
            0.0,
        )
        sigma_rr = float(self._cfg("radar_range_rate_sigma_mps", 0.05))
        sigma_pos = float(self._cfg("radar_position_sigma_m", 2.0))
        ranges = np.maximum(np.linalg.norm(relative_positions, axis=1), 1.0)
        # Measurement noise of range-rate (radar + line-of-sight direction
        # error), and its total innovation variance including the state.
        noise_rr = sigma_rr ** 2 + (sigma_pos / ranges) ** 2 * tangential_sq
        var_rr = np.einsum("ki,kij,kj->k", radial, p_vv, radial) + noise_rr
        d2_rr = (range_rates - radial_component) ** 2 / np.maximum(var_rr, 1.0e-12)
        d2 = d2_position + d2_rr
        statistically_valid = d2 <= chi2_gate

        # Pairs closer in time than any genuine revisit (or older than the
        # track's last update) that are statistically the same object are
        # replays: they must neither update a track nor create one.
        replay = set(
            obs_index[statistically_valid & (dt < min_interval)].tolist()
        )
        assignable = statistically_valid & (dt >= min_interval)
        pair_rows = track_index[assignable]
        pair_cols = obs_index[assignable]
        pair_costs = (d2 + logdet)[assignable]
        pair_data = {
            (int(r), int(c)): {"noise_rr": float(v)}
            for r, c, v in zip(pair_rows, pair_cols, noise_rr[assignable])
        }
        if pair_rows.size == 0:
            return [], replay, coarse_candidates

        rows = np.unique(pair_rows)
        cols = np.unique(pair_cols)
        row_lookup = {int(r): i for i, r in enumerate(rows)}
        col_lookup = {int(c): j for j, c in enumerate(cols)}
        invalid_cost = 1.0e9
        cost = np.full((rows.size, cols.size), invalid_cost, dtype=float)
        for r, c, value in zip(pair_rows, pair_cols, pair_costs):
            cost[row_lookup[int(r)], col_lookup[int(c)]] = value

        assignments = []
        if linear_sum_assignment is not None:
            row_ids, col_ids = linear_sum_assignment(cost)
            for i, j in zip(row_ids, col_ids):
                if cost[i, j] < invalid_cost:
                    assignments.append((int(rows[i]), int(cols[j])))
        else:
            used_rows, used_cols = set(), set()
            order = np.argsort(cost, axis=None)
            for flat in order:
                i, j = np.unravel_index(flat, cost.shape)
                if cost[i, j] >= invalid_cost:
                    break
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                assignments.append((int(rows[i]), int(cols[j])))

        assigned = [(ti, oi, pair_data[(ti, oi)]) for ti, oi in assignments]
        # An observation that was assigned is not a replay.
        assigned_obs = {oi for _, oi in assignments}
        replay -= assigned_obs
        return assigned, replay, coarse_candidates

    def _kf_update(self, track, item, dt, psd, pair):
        """Predict to the measurement time and fuse position + range-rate."""
        covariance = track.covariance
        if covariance is None:
            covariance = self._track_covariances()[self.tracks.index(track)]
        state = np.concatenate((track.position, track.velocity))
        transition = np.eye(6, dtype=float)
        transition[:3, 3:] = dt * _I3
        state = transition @ state
        covariance = transition @ covariance @ transition.T + _process_noise(dt, psd)

        radial = item["radial"]
        observation_matrix = np.zeros((4, 6), dtype=float)
        observation_matrix[:3, :3] = _I3
        observation_matrix[3, 3:] = radial
        noise = np.zeros((4, 4), dtype=float)
        noise[:3, :3] = item["cov"]
        noise[3, 3] = float(pair["noise_rr"])
        # Range-rate measured relative to the shuttle -> absolute radial speed.
        measurement = np.concatenate((
            item["position"],
            [float(item["range_rate"]) + float(np.dot(item["shuttle_velocity"], radial))],
        ))
        innovation = measurement - observation_matrix @ state
        innovation_cov = observation_matrix @ covariance @ observation_matrix.T + noise
        gain = np.linalg.solve(innovation_cov, observation_matrix @ covariance).T
        state = state + gain @ innovation
        identity_minus = np.eye(6) - gain @ observation_matrix
        covariance = identity_minus @ covariance @ identity_minus.T + gain @ noise @ gain.T
        track.position = state[:3].copy()
        track.velocity = state[3:].copy()
        track.covariance = 0.5 * (covariance + covariance.T)
