import numpy as np

EPS = 1.0e-9


def vec3(value):
    v = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError("Nonfinite 3D vector")
    return v


def norm(value):
    return float(np.linalg.norm(value))


def unit(value, fallback=None):
    v = vec3(value)
    length = norm(v)
    if length > EPS:
        return v / length
    if fallback is not None:
        return unit(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def limit_norm(value, maximum):
    v = vec3(value)
    maximum = float(maximum)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum must be finite and nonnegative")
    length = norm(v)
    if length > maximum and length > EPS:
        return v * (maximum / length)
    return v


def perpendicular_basis(direction, previous_side=None):
    """Return a continuous orthonormal side basis around direction.

    The previous side vector is projected into the new perpendicular plane.
    If it becomes degenerate, a least-aligned Cartesian axis is used. The
    sign is chosen to avoid 180-degree flips between successive updates.
    """
    forward = unit(direction)
    previous = None
    if previous_side is not None:
        try:
            previous = vec3(previous_side)
            projected = previous - float(np.dot(previous, forward)) * forward
            if norm(projected) > 0.15:
                side1 = unit(projected)
                if float(np.dot(side1, previous)) < 0.0:
                    side1 = -side1
                side2 = unit(np.cross(forward, side1))
                side1 = unit(np.cross(side2, forward))
                return side1, side2
        except (TypeError, ValueError):
            previous = None

    # Select the coordinate axis least aligned with the forward vector.
    reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(forward)))]
    side1 = unit(np.cross(forward, reference))
    side2 = unit(np.cross(forward, side1))
    if previous is not None and float(np.dot(side1, previous)) < 0.0:
        side1 = -side1
        side2 = -side2
    return side1, side2


def closest_approach(position_a, velocity_a, position_b, velocity_b, horizon_s):
    relative_position = vec3(position_b) - vec3(position_a)
    relative_velocity = vec3(velocity_b) - vec3(velocity_a)
    speed_squared = float(np.dot(relative_velocity, relative_velocity))
    if speed_squared < EPS:
        time_cpa = 0.0
    else:
        time_cpa = -float(np.dot(relative_position, relative_velocity)) / speed_squared
    time_cpa = float(np.clip(time_cpa, 0.0, max(0.0, float(horizon_s))))
    separation = relative_position + relative_velocity * time_cpa
    return time_cpa, norm(separation)


def swept_separation(shuttle_start, shuttle_end, debris_start, debris_end):
    """Minimum separation of two linearly moving objects over one interval."""
    relative_start = vec3(shuttle_start) - vec3(debris_start)
    relative_end = vec3(shuttle_end) - vec3(debris_end)
    delta = relative_end - relative_start
    denominator = float(np.dot(delta, delta))
    if denominator < EPS:
        fraction = 0.0
    else:
        fraction = float(np.clip(
            -np.dot(relative_start, delta) / denominator, 0.0, 1.0
        ))
    return norm(relative_start + fraction * delta), fraction
