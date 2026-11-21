from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from .geometry import vec3, norm, unit, limit_norm
from .radar import raw_record_key


class StateHistory:
    """Short shuttle-state history for delayed radar measurement reconstruction."""

    def __init__(self, max_age_s=8.0, max_samples=1200):
        self.max_age_s = float(max_age_s)
        self.samples = deque(maxlen=int(max_samples))
        # V5 Phase 2b: parallel timestamps for O(log n) interval lookup.
        self._times = deque(maxlen=int(max_samples))

    def __len__(self):
        return len(self.samples)

    def clear(self):
        self.samples.clear()
        self._times.clear()

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
            self._times[-1] = timestamp
        else:
            self.samples.append((timestamp, position, velocity))
            self._times.append(timestamp)

        while (
            len(self.samples) > 2
            and timestamp - self.samples[0][0] > self.max_age_s
        ):
            self.samples.popleft()
            self._times.popleft()

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

        # first_t < timestamp < last_t: the bracketing interval is the first
        # sample at or after `timestamp` and its predecessor (the same
        # interval the former linear scan selected).
        index = bisect_left(self._times, timestamp)
        t0, p0, v0 = self.samples[index - 1]
        t1, p1, v1 = self.samples[index]
        span = max(t1 - t0, 1.0e-9)
        fraction = (timestamp - t0) / span
        position = p0 + fraction * (p1 - p0)
        velocity = v0 + fraction * (v1 - v0)
        return position, velocity


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
    # V5 Phase A: born from Radar.py's initial snapshot (raw timestamp 0.0),
    # whose epoch/state is not consistent with runtime measurements (live
    # run 0555d4ef: ~77 % of such tracks failed their first loop update).
    # A provisional track is replaced by its first loop measurement.
    provisional: bool = False
    # V5.3 diagnostics: absolute radial velocity measured by the radar at the
    # last update (range-rate + shuttle velocity along the line of sight), the
    # line-of-sight unit vector, and whether the speed was clamped then.
    radial_velocity_meas: float = float("nan")
    los: np.ndarray | None = None
    clamped: bool = False
    # V5.3: per-update consistency evidence (bookkeeping only; the estimator
    # is unchanged). suspect = last update had a large normalised innovation
    # (likely mis-association), or the velocity jumped between consecutive
    # updates of a mature track (debris accelerate only ~0.01 m/s^2), or the
    # speed was clamped. vel_sigma = RMS per-axis velocity sigma after the
    # update (cached so it is not recomputed every controller tick).
    suspect: bool = False
    last_nis: float = float("nan")
    vel_sigma: float = float("nan")
    v_last_update: np.ndarray | None = None

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


def velocity_sigma(covariance):
    """RMS per-axis 1-sigma velocity uncertainty of a 6x6 covariance (nan if unknown)."""
    if covariance is None:
        return float("nan")
    try:
        return float(np.sqrt(max(np.trace(np.asarray(covariance)[3:, 3:]), 0.0) / 3.0))
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _record_radial(track, item):
    """V5.3 diagnostics: store the radar's absolute radial velocity and the
    line of sight of the measurement that last updated `track`."""
    try:
        rel = np.asarray(item["relative_position"], dtype=float)
        los = rel / np.linalg.norm(rel)
        track.los = los
        track.radial_velocity_meas = float(item["range_rate"]) + float(
            np.dot(np.asarray(item["shuttle_velocity"], dtype=float), los))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        track.los, track.radial_velocity_meas = None, float("nan")


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
        self.deferred_new_records = 0
        self.last_update_stats = {}
        self.cumulative = {
            "evicted": 0, "evicted_confirmed": 0, "reacquired": 0, "provisional_replaced": 0,
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
        """Age beyond which a track has certainly missed its radar refresh.

        Radar.py refreshes each object once per sweep and republishes every
        sweep/10 (sortFrequency = N/10), so a correctly tracked object is at
        most ~sweep + sweep/10 + one controller poll old. Phase A: expiry is
        `track_expiry_publication_factor` x p99 measured revisit plus
        `track_expiry_margin_s` (radar jitter, controller overruns). Before
        revisits are measured (startup) track_max_age_s applies. The result
        never exceeds track_max_age_cap_s; it can exceed track_max_age_s if
        the radar itself is that slow (coverage first).
        """
        base = float(self.config.track_max_age_s)
        if len(self._revisit_samples) < int(self._cfg("track_revisit_min_samples", 50)):
            return base
        if self._cfg("track_expiry_mode", "revisit_p99") == "legacy":
            # Phase 1 rule, kept for regression comparison only.
            revisit = float(np.median(np.asarray(self._revisit_samples, dtype=float)))
            return float(min(
                max(base, float(self._cfg("track_revisit_expiry_factor", 1.3)) * revisit),
                max(base, float(self._cfg("track_max_age_cap_s", 9.0))),
            ))
        revisit_p99 = float(np.percentile(np.asarray(self._revisit_samples, dtype=float), 99.0))
        bound = (
            float(self._cfg("track_expiry_publication_factor", 1.1)) * revisit_p99
            + float(self._cfg("track_expiry_margin_s", 0.60))
        )
        return float(min(bound, float(self._cfg("track_max_age_cap_s", 9.0))))

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
    def select_new_records(self, raw_records, timestamp, limit=None):
        """Split raw Radar.py records into (unseen records, cached-copy count).

        `limit` (V5 Phase 2b) caps the unseen records returned for this tick;
        the rest are left unmarked and are therefore returned on the next
        tick. Radar.py publishes records sorted by range, so the nearest
        unseen objects are ingested first. In steady state a publication
        carries ~N/10 new records and the cap does not bind; it spreads the
        one-time bootstrap (all N records new) over a few ticks.
        self.deferred_new_records reports how many were held back.

        V5 Phase 2: lets the controller decode only genuinely new records.
        Each publication carries ~N/10 new rows and ~9N/10 bit-identical
        cached copies; decoding the copies (state-history interpolation,
        range fusion) dominated the per-tick cost. Cached keys have their
        last-seen time refreshed exactly as update() would. Records are only
        marked as seen by update(), after they have been decoded, so an
        undecodable or future-dated record is retried on the next poll.
        """
        timestamp = float(timestamp)
        new_records = []
        cached = 0
        deferred = 0
        cap = float("inf") if limit is None else max(1, int(limit))
        seen = self._seen_measurements
        for item in raw_records:
            try:
                key = raw_record_key(item)
            except (TypeError, ValueError):
                key = None
            if key is not None and key in seen:
                seen[key] = timestamp
                cached += 1
            elif len(new_records) < cap:
                new_records.append(item)
            else:
                deferred += 1
        self.deferred_new_records = deferred
        return new_records, cached

    def update(
        self,
        observations,
        timestamp,
        shuttle_position,
        shuttle_velocity,
        prefiltered_duplicates=0,
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
        duplicate_count = int(prefiltered_duplicates)
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
                position_cov = obs                  # resolved in bulk below
                raw_time = getattr(obs, "raw_timestamp", None)
                valid.append({
                    "provisional": bool(
                        self._cfg("track_provisional_init_snapshot", True)
                        and raw_time is not None and float(raw_time) == 0.0
                    ),
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

        if valid:
            self._resolve_measurement_covariances(valid)

        assigned = []          # (track index, observation index, pair data)
        replay = set()         # observation indices consistent with a newer/same-time track state
        coarse_candidates = None  # coarse (track, obs) pairs, see _associate

        if self.tracks and valid:
            assigned, replay, coarse_candidates = self._associate(
                valid, min_interval, psd, chi2_gate, revisit
            )

        matched_count = 0
        clamped_count = 0
        max_speed = float(cfg.max_track_speed_mps)
        updated_tracks = set()
        if assigned:
            # V5 Phase 2b: one batched predict/update for every assigned
            # (track, observation) pair; identical equations to _kf_update.
            self._kf_update_batch(assigned, valid, psd)
        for ti, oi, pair in assigned:
            track = self.tracks[ti]
            item = valid[oi]
            dt = item["time"] - track.timestamp
            speed = norm(track.velocity)
            track.clamped = speed > max_speed
            if track.clamped:
                track.velocity = limit_norm(track.velocity, max_speed)
                clamped_count += 1
            track.diameter = item["diameter"]
            track.range_rate = item["range_rate"]
            _record_radial(track, item)
            nis = float(pair.get("d2", float("nan"))) if isinstance(pair, dict) else float("nan")
            jump = (norm(track.velocity - track.v_last_update)
                    if track.v_last_update is not None and track.hits >= cfg.confirmation_hits else 0.0)
            track.last_nis = nis
            track.suspect = bool(track.clamped
                                 or (np.isfinite(nis) and nis > float(self._cfg("v53_nis_suspect", 13.28)))
                                 or jump > float(self._cfg("v53_vjump_mps", 1.5)))
            track.v_last_update = np.array(track.velocity, dtype=float)
            track.vel_sigma = velocity_sigma(track.covariance)
            track.timestamp = item["time"]
            track.hits += 1
            track.confirmed = track.hits >= cfg.confirmation_hits
            # Snapshot-to-first-measurement intervals (uniform over one sweep)
            # are not revisits; including them biased the revisit estimate low.
            if not track.provisional:
                self._revisit_samples.append(dt)
            track.provisional = False
            updated_tracks.add(ti)
            matched_count += 1

        assigned_observations = {oi for _, oi, _ in assigned}
        created_count = 0
        reacquired_count = 0
        provisional_replaced = 0
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
            # duplicate that would leave the old track as a ghost. "Overdue"
            # means it missed its expected revisit (track_reacquire_min_fraction
            # x measured revisit); a track still due for its own measurement
            # must not be taken over. A provisional (initial-snapshot) track
            # may be replaced by any later loop measurement of its object.
            reacquire_index = None
            for distance, ti, dt in self._coarse_candidates_for(coarse_candidates, oi):
                if ti in updated_tracks or ti in evicted or dt < min_interval:
                    continue
                if not (self.tracks[ti].provisional or dt >= reacquire_after):
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
                _record_radial(track, item)
                track.clamped = False
                track.suspect, track.last_nis = False, float("nan")
                track.v_last_update = np.array(velocity, dtype=float)
                track.vel_sigma = velocity_sigma(covariance)
                track.timestamp = item["time"]
                track.hits = 1
                track.confirmed = False
                if track.provisional:
                    provisional_replaced += 1
                else:
                    reacquired_count += 1
                track.provisional = item["provisional"]
                updated_tracks.add(reacquire_index)
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
                    provisional=item["provisional"],
                )
            )
            _record_radial(new_tracks[-1], item)
            new_tracks[-1].v_last_update = np.array(velocity, dtype=float)
            new_tracks[-1].vel_sigma = velocity_sigma(covariance)
            self.next_identifier += 1
            created_count += 1

        if evicted or new_tracks:
            self.tracks = [
                track for index, track in enumerate(self.tracks)
                if index not in evicted
            ] + new_tracks

        # Expire AFTER association: after a controller stall, a track's fresh
        # measurement may be in this very batch and must be used, not lost.
        # Genuine revisit samples exist only once objects have been measured
        # twice by the radar loop, i.e. after the first full sweep. Every
        # object has then had a loop measurement, so a provisional
        # (initial-snapshot) track still present was never matched by its
        # object: it is a snapshot artefact, not a missed object.
        sweep_complete = np.isfinite(self.revisit_period_s())
        before = len(self.tracks)
        self.tracks = [
            track for track in self.tracks
            if -cfg.future_timestamp_tolerance_s
            <= timestamp - track.timestamp
            <= expiry
            and not (sweep_complete and track.provisional)
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
        self.cumulative["provisional_replaced"] += provisional_replaced
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
            "provisional_replaced": provisional_replaced,
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
        """Least valuable track: provisional, then overdue, then fewest hits, then oldest."""
        overdue_after = (
            revisit if np.isfinite(revisit)
            else float(self.config.track_max_age_s) / 1.3
        )
        best = None
        for index, track in enumerate(self.tracks):
            if index in protected:
                continue
            age = timestamp - float(track.timestamp)
            # Provisional snapshot placeholders first, then overdue tracks,
            # then fewest hits, then oldest.
            key = (
                0 if track.provisional else 1,
                0 if age > overdue_after else 1,
                int(track.hits),
                -age,
            )
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

    def _associate(self, valid, min_interval, psd, chi2_gate, revisit=float("nan")):
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

        # Coarse pairs as arrays (queried lazily, only for observations that
        # end up unassigned; see _coarse_candidates_for).
        coarse_candidates = (
            track_index, obs_index,
            distances[track_index, obs_index], dt_matrix[track_index, obs_index],
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
        # Official Radar.py refreshes object `iteration % numDebris` of an
        # unsorted list (strict round-robin), so the same object is measured
        # again only after ~one full sweep; jitter averages out over the sweep
        # and hitches only lengthen it. Once the revisit is measured, a pairing
        # sooner than track_min_revisit_fraction x revisit cannot be the same
        # object and may not update that track (Phase A: replay seed 100, an
        # 8.3 m/s object's measurement was assigned to a track updated 2.47 s
        # earlier, corrupting it and orphaning the object's own track). The
        # measurement stays available to its own track; it is not a replay.
        # Provisional snapshot tracks are exempt (first loop update any time).
        pair_interval = np.full(dt.shape, float(min_interval))
        if np.isfinite(revisit):
            provisional = np.fromiter(
                (self.tracks[ti].provisional for ti in track_index.tolist()),
                dtype=bool, count=len(track_index),
            )
            pair_interval = np.where(
                provisional, pair_interval,
                np.maximum(pair_interval, float(self._cfg("track_min_revisit_fraction", 0.75)) * revisit),
            )
        assignable = statistically_valid & (dt >= pair_interval)
        pair_rows = track_index[assignable]
        pair_cols = obs_index[assignable]
        pair_costs = (d2 + logdet)[assignable]
        pair_data = {
            (int(r), int(c)): {"noise_rr": float(v), "d2": float(q)}
            for r, c, v, q in zip(pair_rows, pair_cols, noise_rr[assignable], d2[assignable])
        }
        if pair_rows.size == 0:
            return [], replay, coarse_candidates

        rows = np.unique(pair_rows)
        cols = np.unique(pair_cols)
        invalid_cost = 1.0e9
        cost = np.full((rows.size, cols.size), invalid_cost, dtype=float)
        cost[np.searchsorted(rows, pair_rows), np.searchsorted(cols, pair_cols)] = pair_costs

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

    @staticmethod
    def _coarse_candidates_for(coarse_candidates, observation_index):
        """(distance, track index, dt) for one observation, ordered exactly as
        sorted() ordered the former tuples: distance, then track, then dt."""
        if coarse_candidates is None:
            return []
        track_index, obs_index, distances, dts = coarse_candidates
        mask = obs_index == observation_index
        if not np.any(mask):
            return []
        tracks, dist, dt = track_index[mask], distances[mask], dts[mask]
        order = np.lexsort((dt, tracks, dist))
        return [(float(dist[i]), int(tracks[i]), float(dt[i])) for i in order]

    def _resolve_measurement_covariances(self, valid):
        """Bulk version of _measurement_covariance for all fresh observations."""
        extra = float(self._cfg("radar_measurement_extra_sigma_m", 0.0)) ** 2
        default = self._default_position_covariance()
        cross = np.asarray([
            float(getattr(item["cov"], "measurement_cross_variance", float("nan")))
            for item in valid
        ])
        radial_variance = np.asarray([
            float(getattr(item["cov"], "measurement_radial_variance", float("nan")))
            for item in valid
        ])
        radial = np.asarray([item["radial"] for item in valid], dtype=float)
        bulk = (
            (cross + extra)[:, None, None] * _I3[None, :, :]
            + (radial_variance - cross)[:, None, None] * np.einsum("ki,kj->kij", radial, radial)
        )
        usable = np.isfinite(cross) & np.isfinite(radial_variance)
        for index, item in enumerate(valid):
            explicit = getattr(item["cov"], "measurement_covariance", None)
            if explicit is not None:
                item["cov"] = np.asarray(explicit, dtype=float).reshape(3, 3)
            elif usable[index]:
                item["cov"] = bulk[index]
            else:
                item["cov"] = default

    def _kf_update_batch(self, assigned, valid, psd):
        """Batched constant-velocity Kalman predict/update.

        Same equations as _kf_update (kept below as the per-track reference):
        predict with F(dt) and white-acceleration Q(|dt|), fuse position and
        absolute radial speed, Joseph-form covariance update, symmetrize.
        """
        count = len(assigned)
        state = np.empty((count, 6))
        covariance = np.empty((count, 6, 6))
        dt = np.empty(count)
        radial = np.empty((count, 3))
        noise = np.zeros((count, 4, 4))
        measurement = np.empty((count, 4))
        for k, (ti, oi, pair) in enumerate(assigned):
            track = self.tracks[ti]
            item = valid[oi]
            if track.covariance is None:
                self._track_covariances()
            state[k, :3] = track.position
            state[k, 3:] = track.velocity
            covariance[k] = track.covariance
            dt[k] = item["time"] - track.timestamp
            radial[k] = item["radial"]
            noise[k, :3, :3] = item["cov"]
            noise[k, 3, 3] = float(pair["noise_rr"])
            measurement[k, :3] = item["position"]
            measurement[k, 3] = float(item["range_rate"]) + float(
                np.dot(item["shuttle_velocity"], item["radial"])
            )

        transition = np.broadcast_to(np.eye(6), (count, 6, 6)).copy()
        transition[:, :3, 3:] = dt[:, None, None] * _I3
        abs_dt = np.abs(dt)
        process = np.zeros((count, 6, 6))
        process[:, :3, :3] = (psd * abs_dt ** 3 / 3.0)[:, None, None] * _I3
        process[:, :3, 3:] = (psd * abs_dt ** 2 / 2.0)[:, None, None] * _I3
        process[:, 3:, :3] = process[:, :3, 3:]
        process[:, 3:, 3:] = (psd * abs_dt)[:, None, None] * _I3
        state = np.einsum("kij,kj->ki", transition, state)
        covariance = transition @ covariance @ np.transpose(transition, (0, 2, 1)) + process

        observation = np.zeros((count, 4, 6))
        observation[:, :3, :3] = _I3
        observation[:, 3, 3:] = radial
        innovation = measurement - np.einsum("kij,kj->ki", observation, state)
        observation_t = np.transpose(observation, (0, 2, 1))
        innovation_cov = observation @ covariance @ observation_t + noise
        gain = np.transpose(np.linalg.solve(innovation_cov, observation @ covariance), (0, 2, 1))
        state = state + np.einsum("kij,kj->ki", gain, innovation)
        identity_minus = np.eye(6)[None, :, :] - gain @ observation
        covariance = (
            identity_minus @ covariance @ np.transpose(identity_minus, (0, 2, 1))
            + gain @ noise @ np.transpose(gain, (0, 2, 1))
        )
        covariance = 0.5 * (covariance + np.transpose(covariance, (0, 2, 1)))
        for k, (ti, _, _) in enumerate(assigned):
            track = self.tracks[ti]
            track.position = state[k, :3].copy()
            track.velocity = state[k, 3:].copy()
            track.covariance = covariance[k].copy()

    def _kf_update(self, track, item, dt, psd, pair):
        """Per-track reference of the predict/update used by _kf_update_batch."""
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
