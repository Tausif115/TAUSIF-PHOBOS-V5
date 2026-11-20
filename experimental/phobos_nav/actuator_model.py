"""Read-only thruster geometry and conservative acceleration authority.

This module does not command jets or modify simulator parameters. It derives
an inscribed translational authority set from the RCSConfig records populated
by the simulator's configured actuator system.
"""

import csv
import numpy as np

_EPS = 1.0e-9


def radial_acceleration_limit(
    command_body,
    positive_axis_authority,
    negative_axis_authority,
    safety_factor=0.82,
):
    command = np.asarray(command_body, dtype=float).reshape(3)
    positive = np.asarray(positive_axis_authority, dtype=float).reshape(3)
    negative = np.asarray(negative_axis_authority, dtype=float).reshape(3)
    factor = float(safety_factor)

    if (
        not np.all(np.isfinite(command))
        or not np.all(np.isfinite(positive))
        or not np.all(np.isfinite(negative))
        or not np.isfinite(factor)
        or factor <= 0.0
        or factor > 1.0
    ):
        return 0.0

    magnitude = float(np.linalg.norm(command))
    if magnitude <= _EPS:
        return float("inf")

    direction = command / magnitude
    active = np.abs(direction) > _EPS
    axis_authority = np.where(direction >= 0.0, positive, negative)
    if np.any(axis_authority[active] <= _EPS):
        return 0.0

    denominator = float(np.sum(
        np.abs(direction[active]) / axis_authority[active]
    ))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return 0.0
    return factor / denominator


def limit_acceleration_body(
    command_body,
    positive_axis_authority,
    negative_axis_authority,
    safety_factor=0.82,
):
    command = np.asarray(command_body, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(command))
    if not np.isfinite(magnitude):
        return np.zeros(3, dtype=float), 0.0
    if magnitude <= _EPS:
        return np.zeros(3, dtype=float), float("inf")
    limit = radial_acceleration_limit(
        command, positive_axis_authority, negative_axis_authority,
        safety_factor=safety_factor,
    )
    if limit <= 0.0:
        return np.zeros(3, dtype=float), 0.0
    achieved = min(magnitude, limit)
    return command * (achieved / magnitude), float(limit)


def limit_acceleration_body_batch(
    commands_body,
    positive_axis_authority,
    negative_axis_authority,
    safety_factor=0.82,
):
    commands = np.asarray(commands_body, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands_body must have shape (N,3)")
    positive = np.asarray(positive_axis_authority, dtype=float).reshape(3)
    negative = np.asarray(negative_axis_authority, dtype=float).reshape(3)
    factor = float(safety_factor)

    if (
        not np.all(np.isfinite(commands))
        or not np.all(np.isfinite(positive))
        or not np.all(np.isfinite(negative))
        or not np.isfinite(factor)
        or factor <= 0.0
        or factor > 1.0
    ):
        return np.zeros_like(commands)

    magnitudes = np.linalg.norm(commands, axis=1)
    output = np.zeros_like(commands)
    moving = magnitudes > _EPS
    if not np.any(moving):
        return output

    directions = commands[moving] / magnitudes[moving, None]
    axis_authority = np.where(
        directions >= 0.0, positive[None, :], negative[None, :]
    )
    active = np.abs(directions) > _EPS
    invalid = np.any((axis_authority <= _EPS) & active, axis=1)
    denominators = np.sum(
        np.divide(
            np.abs(directions),
            axis_authority,
            out=np.zeros_like(directions),
            where=axis_authority > _EPS,
        ),
        axis=1,
    )
    valid = (
        (~invalid)
        & (denominators > _EPS)
        & np.isfinite(denominators)
    )
    limits = np.zeros_like(denominators)
    limits[valid] = factor / denominators[valid]
    scales = np.minimum(1.0, limits / np.maximum(magnitudes[moving], _EPS))
    scales[~valid] = 0.0
    output[moving] = commands[moving] * scales[:, None]
    return output



def read_thruster_csv(path):
    """Read the seven force-geometry columns used by the simulator actuator."""
    records = []
    with open(path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 7:
                raise ValueError(
                    f"Thruster CSV row {row_number} has fewer than 7 columns"
                )
            try:
                values = [float(value) for value in row[:7]]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid thruster CSV row {row_number}: {error}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Nonfinite thruster CSV row {row_number}"
                )
            records.append(values)
    if not records or len(records) > 256:
        raise ValueError(f"Unexpected thruster count in CSV: {len(records)}")
    return records


def build_activity_matrix(thruster_records, mass, inertia_body):
    """Build per-jet [linear accel, angular accel] rows from RCSConfig data."""
    mass = float(mass)
    inertia = np.asarray(inertia_body, dtype=float).reshape(3, 3)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Invalid vehicle mass")
    if not np.all(np.isfinite(inertia)):
        raise ValueError("Nonfinite inertia tensor")
    inertia = 0.5 * (inertia + inertia.T)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 1.0e-8):
        raise ValueError("Inertia tensor is not positive definite")
    inertia_inv = np.linalg.inv(inertia)

    rows = []
    for record in thruster_records:
        values = np.asarray(record, dtype=float).reshape(-1)
        if values.size < 7 or not np.all(np.isfinite(values[:7])):
            raise ValueError("Invalid RCSConfig thruster record")
        position = values[0:3]
        normal = values[3:6]
        thrust = float(values[6])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-10 or thrust <= 0.0:
            raise ValueError("Invalid thruster direction or force")
        force = -thrust * normal / normal_norm
        torque = np.cross(position, force)
        linear_accel = force / mass
        angular_accel = inertia_inv @ torque
        rows.append(np.concatenate((linear_accel, angular_accel)))
    if not rows:
        raise ValueError("No thruster records")
    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Nonfinite activity matrix")
    return matrix


def _solve_axis_authority(activity_matrix, direction, linprog):
    matrix = np.asarray(activity_matrix, dtype=float)
    direction = np.asarray(direction, dtype=float).reshape(3)
    n = matrix.shape[0]
    target_wrench = np.concatenate((direction, np.zeros(3, dtype=float)))
    equality = np.hstack((matrix.T, -target_wrench.reshape(6, 1)))
    objective = np.zeros(n + 1, dtype=float)
    objective[-1] = -1.0
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=np.zeros(6, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        return 0.0, False, float("inf")
    solution = np.asarray(result.x, dtype=float).reshape(-1)
    if solution.shape != (n + 1,) or not np.all(np.isfinite(solution)):
        return 0.0, False, float("inf")
    duties = np.clip(solution[:-1], 0.0, 1.0)
    authority = max(0.0, float(solution[-1]))
    achieved = matrix.T @ duties
    residual = float(np.linalg.norm(
        achieved - authority * target_wrench, ord=np.inf
    ))
    tolerance = 1.0e-5 * max(1.0, authority)
    valid = bool(
        authority > 1.0e-7
        and np.all(duties >= -1.0e-9)
        and np.all(duties <= 1.0 + 1.0e-9)
        and np.isfinite(residual)
        and residual <= tolerance
    )
    return (authority if valid else 0.0), valid, residual


def compute_axis_authorities(activity_matrix):
    """Compute feasible +/- body-axis translation with zero angular accel."""
    try:
        from scipy.optimize import linprog
    except Exception as error:
        raise RuntimeError("SciPy linprog is required for thruster authority") from error

    matrix = np.asarray(activity_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 6 or not np.all(np.isfinite(matrix)):
        raise ValueError("Activity matrix must have finite shape (N,6)")

    positive = np.zeros(3, dtype=float)
    negative = np.zeros(3, dtype=float)
    residuals = []
    valid = True
    for axis in range(3):
        direction = np.zeros(3, dtype=float)
        direction[axis] = 1.0
        positive[axis], ok_pos, res_pos = _solve_axis_authority(
            matrix, direction, linprog
        )
        negative[axis], ok_neg, res_neg = _solve_axis_authority(
            matrix, -direction, linprog
        )
        valid = valid and ok_pos and ok_neg
        residuals.extend((res_pos, res_neg))
    return positive, negative, bool(valid), max(residuals, default=0.0)


def transform_matrix_phobos_to_body(basis_vectors_body):
    basis = np.asarray(basis_vectors_body, dtype=float).reshape(3, 3)
    rotation = basis.T
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Nonfinite frame transformation")
    if np.linalg.norm(
        rotation.T @ rotation - np.eye(3), ord=np.inf
    ) > 1.0e-3:
        raise ValueError("Frame transformation is not orthonormal")
    return rotation
