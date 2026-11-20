import numpy as np

from .geometry import limit_norm, norm, unit


def approach_speed(distance_m, config, max_accel):
    """Stopping-distance speed profile for a moving-target approach."""
    distance = max(0.0, float(distance_m))
    remaining = max(distance - float(config.capture_radius_m), 0.0)
    braking_accel = max(
        float(max_accel) * float(config.braking_accel_fraction),
        1.0e-5,
    )
    speed = np.sqrt(
        float(config.terminal_speed_mps) ** 2
        + 2.0 * braking_accel * remaining
    )
    return float(min(float(config.goal_speed), speed))


def desired_target_velocity(
    position,
    target_position,
    target_velocity,
    config,
    max_accel,
    direction_bias=None,
    speed_scale=1.0,
    hold=False,
):
    """Return desired world velocity, with a damped terminal capture region.

    Outside the capture radius the vehicle follows a stopping-distance speed
    profile. Inside it, the desired relative velocity points gently toward the
    target and goes to zero at the target centre. Lateral bias fades near the
    capture region so avoidance does not become a permanent sideways orbit.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    target_velocity = np.asarray(target_velocity, dtype=float).reshape(3)
    relative_position = target_position - position
    distance = norm(relative_position)

    if hold:
        return target_velocity.copy(), 0.0, distance

    if distance <= float(config.capture_radius_m):
        direction = unit(relative_position, fallback=[1.0, 0.0, 0.0])
        standoff_error = distance - float(config.capture_standoff_m)
        relative_desired = limit_norm(
            float(config.capture_gain_per_s) * standoff_error * direction,
            float(config.capture_max_speed_mps),
        )
        return target_velocity + relative_desired, norm(relative_desired), distance

    direction = unit(relative_position)
    if direction_bias is not None:
        bias = np.asarray(direction_bias, dtype=float).reshape(3)
        bias = bias - float(np.dot(bias, direction)) * direction
        fade = float(np.clip(
            (distance - float(config.capture_radius_m))
            / max(float(config.bias_fade_distance_m), 1.0),
            0.0,
            1.0,
        ))
        direction = unit(direction + fade * bias, fallback=direction)

    speed = approach_speed(distance, config, max_accel)
    speed *= float(np.clip(speed_scale, 0.0, 1.0))
    speed = max(0.0, min(speed, float(config.goal_speed)))
    relative_desired = speed * direction
    return target_velocity + relative_desired, speed, distance


def guidance_acceleration(
    position,
    velocity,
    target_position,
    target_velocity,
    config,
    max_accel,
    direction_bias=None,
    speed_scale=1.0,
    hold=False,
):
    desired_velocity, desired_relative_speed, distance = desired_target_velocity(
        position,
        target_position,
        target_velocity,
        config,
        max_accel,
        direction_bias=direction_bias,
        speed_scale=speed_scale,
        hold=hold,
    )
    acceleration = float(config.velocity_gain) * (
        np.asarray(desired_velocity, dtype=float).reshape(3)
        - np.asarray(velocity, dtype=float).reshape(3)
    )
    return acceleration, desired_velocity, desired_relative_speed, distance


def rate_damping(current_angular_velocity, gain, max_angular_accel):
    raw = -float(gain) * np.asarray(
        current_angular_velocity, dtype=float
    ).reshape(3)
    return limit_norm(raw, max_angular_accel)
