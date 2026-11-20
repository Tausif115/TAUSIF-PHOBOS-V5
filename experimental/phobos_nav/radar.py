from collections import deque
from dataclasses import dataclass
import numpy as np

from .geometry import vec3, norm


@dataclass
class RadarObservation:
    # Absolute position in the Phobos body-fixed frame.
    position: np.ndarray
    diameter: float
    range_rate: float
    timestamp: float | None = None
    closest_point: float = float("nan")
    relative_position: np.ndarray | None = None
    shuttle_velocity: np.ndarray | None = None


class RadarClockSynchronizer:
    """Estimate the radar-process epoch offset from the newest radar record.

    Radar.py timestamps measurements from its own process-start time. The
    controller timestamps its state history from simulation StartDateTAI.
    These clocks are not guaranteed to share an epoch. The radar array is
    sorted/published periodically; the newest record therefore provides a
    repeatable clock-offset observation.
    """

    def __init__(self, config):
        self.config = config
        self.offset_samples = deque(maxlen=int(config.radar_sync_window))
        self.offset_s = 0.0
        self.last_raw_max_s = None
        self.last_controller_time_s = None
        self.latest_age_s = float("inf")
        self.sample_count = 0
        self.ready = False

    @staticmethod
    def _max_timestamp(raw_data):
        values = []
        for item in raw_data:
            try:
                matrix = np.asarray(item, dtype=float)
                if matrix.shape == (3, 3):
                    value = float(matrix[2, 0])
                    if np.isfinite(value) and value >= 0.0:
                        values.append(value)
            except (TypeError, ValueError, IndexError):
                continue
        return max(values) if values else None

    def reset(self):
        self.offset_samples.clear()
        self.offset_s = 0.0
        self.last_raw_max_s = None
        self.last_controller_time_s = None
        self.latest_age_s = float("inf")
        self.sample_count = 0
        self.ready = False

    def update(self, raw_data, controller_time_s):
        now = float(controller_time_s)
        if not np.isfinite(now):
            self.reset()
            return self.offset_s, False, float("inf")

        if (
            self.last_controller_time_s is not None
            and now < self.last_controller_time_s - 0.5
        ):
            self.reset()

        raw_max = self._max_timestamp(raw_data)
        if raw_max is None:
            self.last_controller_time_s = now
            self.ready = False
            return self.offset_s, False, float("inf")

        if (
            self.last_raw_max_s is not None
            and raw_max < self.last_raw_max_s - 0.5
        ):
            # Radar process restarted or its timebase reset.
            self.reset()

        if raw_max > 0.02 and (
            self.last_raw_max_s is None
            or raw_max > self.last_raw_max_s + 1.0e-5
        ):
            # Controller samples at 60 Hz; the first read after the radar's
            # periodic array publication is normally within one controller
            # period. Subtract half a controller period from that read delay.
            poll_delay = 0.5 / max(
                float(self.config.radar_controller_poll_hz), 1.0
            )
            offset_sample = now - raw_max - poll_delay
            if np.isfinite(offset_sample):
                self.offset_samples.append(float(offset_sample))
                self.sample_count += 1
                self.offset_s = float(np.median(
                    np.asarray(self.offset_samples, dtype=float)
                ))

        self.last_raw_max_s = raw_max
        self.last_controller_time_s = now
        mapped_latest_time = raw_max + self.offset_s
        self.latest_age_s = max(0.0, now - mapped_latest_time)
        self.ready = bool(
            len(self.offset_samples) >= int(self.config.radar_sync_min_samples)
            and self.latest_age_s <= float(self.config.radar_stale_limit_s)
        )
        return self.offset_s, self.ready, self.latest_age_s


def decode_radar(
    raw_data,
    shuttle_position,
    shuttle_velocity=None,
    current_timestamp=None,
    state_history=None,
    radar_time_offset_s=0.0,
    config=None,
):
    """Decode timestamped 3x3 radar records into absolute observations.

    Radar.py row layout:
      row 0: debris position relative to shuttle, in Phobos axes
      row 1: closest surface range, radial range-rate, debris diameter
      row 2, col 0: measurement timestamp in the radar process timebase

    The supplied offset converts that process timebase to controller elapsed
    simulation time before sampling the shuttle state history.
    """
    shuttle_position = vec3(shuttle_position)
    if shuttle_velocity is None:
        shuttle_velocity = np.zeros(3, dtype=float)
    shuttle_velocity = vec3(shuttle_velocity)

    if current_timestamp is None:
        current_timestamp = 0.0
    current_timestamp = float(current_timestamp)
    offset = float(radar_time_offset_s)
    if not np.isfinite(current_timestamp) or not np.isfinite(offset):
        raise ValueError("Radar/controller timestamps must be finite")

    observations = []
    for item in raw_data:
        try:
            matrix = np.asarray(item, dtype=float)
            if matrix.shape != (3, 3):
                continue

            relative_position = vec3(matrix[0, :])
            closest_point = float(matrix[1, 0])
            range_rate = float(matrix[1, 1])
            diameter = float(matrix[1, 2])
            raw_measurement_time = float(matrix[2, 0])

            measured_center_range = norm(relative_position)
            if measured_center_range < 1.0e-9:
                continue
            if not np.isfinite(closest_point):
                continue
            if not np.isfinite(range_rate):
                continue
            if not np.isfinite(diameter) or diameter < 0.0:
                continue

            # Radar.py publishes both a noisy 3-D relative-position vector and
            # an independently noisy closest-surface range. Since its source
            # computes closest_point = true_center_range - diameter/2, fuse
            # the two independent radial measurements while preserving the
            # measured 3-D line of sight.
            if config is not None:
                position_sigma = float(config.radar_position_sigma_m)
                surface_sigma = float(config.radar_closest_range_sigma_m)
                diameter_fraction = float(
                    config.radar_diameter_sigma_fraction
                )
                center_range_from_surface = closest_point + 0.5 * diameter
                variance_position = position_sigma ** 2
                variance_surface = (
                    surface_sigma ** 2
                    + (0.5 * diameter * diameter_fraction) ** 2
                )
                if (
                    center_range_from_surface > 0.0
                    and variance_position > 0.0
                    and variance_surface > 0.0
                    and np.all(np.isfinite([
                        center_range_from_surface,
                        variance_position,
                        variance_surface,
                    ]))
                ):
                    fused_range = (
                        measured_center_range / variance_position
                        + center_range_from_surface / variance_surface
                    ) / (
                        1.0 / variance_position
                        + 1.0 / variance_surface
                    )
                    relative_position = (
                        relative_position
                        * (fused_range / measured_center_range)
                    )
            if (
                not np.isfinite(raw_measurement_time)
                or raw_measurement_time < 0.0
            ):
                continue

            measurement_time = raw_measurement_time + offset
            if state_history is not None and len(state_history) > 0:
                try:
                    shuttle_position_at_obs, shuttle_velocity_at_obs = (
                        state_history.sample(measurement_time)
                    )
                except (RuntimeError, ValueError):
                    age = current_timestamp - measurement_time
                    shuttle_position_at_obs = shuttle_position - shuttle_velocity * age
                    shuttle_velocity_at_obs = shuttle_velocity
            else:
                age = current_timestamp - measurement_time
                shuttle_position_at_obs = shuttle_position - shuttle_velocity * age
                shuttle_velocity_at_obs = shuttle_velocity

            absolute_position = (
                vec3(shuttle_position_at_obs) + relative_position
            )
            observations.append(
                RadarObservation(
                    position=absolute_position,
                    diameter=diameter,
                    range_rate=range_rate,
                    timestamp=measurement_time,
                    closest_point=closest_point,
                    relative_position=relative_position.copy(),
                    shuttle_velocity=vec3(shuttle_velocity_at_obs).copy(),
                )
            )
        except (ValueError, TypeError, IndexError, AttributeError):
            continue

    return observations
