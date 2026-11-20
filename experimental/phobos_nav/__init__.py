"""Phobos V4.2 navigation modules; controller-only competition implementation."""

from .config import CONFIG, Config
from .radar import RadarObservation, RadarClockSynchronizer, decode_radar
from .tracking import DebrisTracker, DebrisTrack, StateHistory
from .planning import PredictivePlanner
from .guidance import approach_speed, desired_target_velocity, guidance_acceleration
from .actuator_model import (
    read_thruster_csv,
    radial_acceleration_limit,
    limit_acceleration_body,
    limit_acceleration_body_batch,
    build_activity_matrix,
    compute_axis_authorities,
)
