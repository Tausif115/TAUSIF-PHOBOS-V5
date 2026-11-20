import numpy as np


def format_radar_diagnostic(
    timestamp,
    shuttle_position,
    shuttle_velocity,
    raw_data,
    max_objects=5,
    radar_time_offset_s=0.0,
):
    """Format raw radar timestamps and geometry without changing tracker state."""
    shuttle_position = np.asarray(
        shuttle_position, dtype=float
    ).reshape(3)
    shuttle_velocity = np.asarray(
        shuttle_velocity, dtype=float
    ).reshape(3)

    lines = [
        f"t={float(timestamp):.3f}",
        f"shuttle_pos={np.round(shuttle_position, 2)}",
        f"shuttle_vel={np.round(shuttle_velocity, 2)}",
        f"records={len(raw_data)}"
    ]

    for index, item in enumerate(raw_data[:max_objects]):
        try:
            matrix = np.asarray(item, dtype=float)
            if matrix.shape != (3, 3):
                lines.append(f"debris[{index}] invalid_shape={matrix.shape}")
                continue

            relative = matrix[0, :]
            raw_measurement_time = float(matrix[2, 0])
            measurement_time = raw_measurement_time + float(
                radar_time_offset_s
            )
            age = float(timestamp) - measurement_time
            lines.extend([
                f"debris[{index}] rel={np.round(relative, 2)}",
                f"debris[{index}] range={matrix[1, 0]:.2f}",
                f"debris[{index}] range_rate={matrix[1, 1]:.3f}",
                f"debris[{index}] diameter={matrix[1, 2]:.2f}",
                f"debris[{index}] measurement_t={measurement_time:.3f}",
                f"debris[{index}] age_s={age:.3f}"
            ])
        except (ValueError, TypeError, IndexError):
            lines.append(f"debris[{index}] decode_error")

    return " | ".join(lines)
