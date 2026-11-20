"""Persistent 3-D receding-horizon planner for Phobos debris avoidance."""

import numpy as np

from .geometry import norm, unit, limit_norm, perpendicular_basis
from .guidance import approach_speed, desired_target_velocity


def collision_envelope_radius(
    config, diameter_m, track_age_s, prediction_time_s, tentative=False
):
    """Conservative physical + measurement + future-process collision radius."""
    age = np.maximum(0.0, np.asarray(track_age_s, dtype=float))
    future = np.maximum(0.0, np.asarray(prediction_time_s, dtype=float))
    total_age = age + future
    tentative_array = np.asarray(tentative, dtype=bool)
    velocity_uncertainty = np.where(
        tentative_array,
        float(config.tentative_velocity_uncertainty_mps),
        float(config.confirmed_velocity_uncertainty_mps),
    )
    tentative_margin = (
        tentative_array.astype(float) * float(config.new_track_uncertainty_m)
    )
    return (
        float(config.shuttle_radius_m)
        + 0.5 * np.asarray(diameter_m, dtype=float)
        + float(config.clearance_margin_m)
        + float(config.initial_uncertainty_m)
        + tentative_margin
        + float(config.uncertainty_growth_mps) * total_age
        + 0.5 * float(config.uncertainty_accel_mps2) * total_age ** 2
        + velocity_uncertainty * total_age
    )


class PredictivePlanner:
    """Clearance-first MPC with target capture, stable lateral frame and hysteresis."""

    def __init__(self, config):
        self.config = config
        self.last_choice = "beeline"
        self.last_metrics = {}
        self.last_bias = np.zeros(3, dtype=float)
        self.last_speed_scale = 1.0
        self._side_basis1 = None
        self.avoidance_active = False
        self._clear_cycles = 0

    def reset(self):
        self.last_choice = "beeline"
        self.last_metrics = {}
        self.last_bias = np.zeros(3, dtype=float)
        self.last_speed_scale = 1.0
        self._side_basis1 = None
        self.avoidance_active = False
        self._clear_cycles = 0

    def _prepare_track_arrays(self, tracks, position, velocity, timestamp):
        if not tracks:
            return None

        cfg = self.config
        horizon = float(cfg.horizon_s)
        shuttle_position = np.asarray(position, dtype=float).reshape(3)
        shuttle_velocity = np.asarray(velocity, dtype=float).reshape(3)
        usable = []

        for track in tracks:
            try:
                track_time = float(track.timestamp)
                age = max(0.0, float(timestamp) - track_time)
                track_position = np.asarray(
                    track.position_at(timestamp)
                    if hasattr(track, "position_at")
                    else np.asarray(track.position, dtype=float)
                    + np.asarray(track.velocity, dtype=float)
                    * (float(timestamp) - track_time),
                    dtype=float,
                ).reshape(3)
                track_velocity = np.asarray(
                    track.velocity, dtype=float
                ).reshape(3)
                diameter = float(track.diameter)
                if not (
                    np.all(np.isfinite(track_position))
                    and np.all(np.isfinite(track_velocity))
                    and np.isfinite(age)
                    and np.isfinite(diameter)
                    and diameter >= 0.0
                ):
                    continue

                confirmed = bool(getattr(
                    track, "confirmed", getattr(track, "hits", 0) >= 3
                ))
                relative_position = track_position - shuttle_position
                relative_velocity = track_velocity - shuttle_velocity
                speed_squared = float(np.dot(
                    relative_velocity, relative_velocity
                ))
                if speed_squared <= 1.0e-9:
                    raw_t_cpa = float("inf")
                    t_cpa = 0.0
                else:
                    raw_t_cpa = -float(np.dot(
                        relative_position, relative_velocity
                    )) / speed_squared
                    t_cpa = float(np.clip(raw_t_cpa, 0.0, horizon))
                approaching = bool(
                    np.isfinite(raw_t_cpa) and raw_t_cpa >= 0.0
                )
                cpa_vector = relative_position + relative_velocity * t_cpa
                cpa_distance = norm(cpa_vector)
                envelope = float(collision_envelope_radius(
                    cfg, diameter, age, t_cpa, tentative=not confirmed
                ))
                usable.append({
                    "position": track_position,
                    "velocity": track_velocity,
                    "age": age,
                    "diameter": diameter,
                    "tentative": not confirmed,
                    "identifier": int(getattr(track, "identifier", -1)),
                    "cpa_time": t_cpa,
                    "cpa_vector": cpa_vector,
                    "cpa_clearance": cpa_distance - envelope,
                    "approaching": approaching,
                    "relative_position": relative_position,
                    "relative_velocity": relative_velocity,
                })
            except (TypeError, ValueError, AttributeError, OverflowError):
                continue

        if not usable:
            return None

        # Rank by predicted CPA clearance, not instantaneous range.
        usable.sort(key=lambda item: item["cpa_clearance"])
        usable = usable[:max(1, int(cfg.max_planning_tracks))]
        return {
            "position": np.asarray([x["position"] for x in usable], dtype=float),
            "velocity": np.asarray([x["velocity"] for x in usable], dtype=float),
            "age": np.asarray([x["age"] for x in usable], dtype=float),
            "diameter": np.asarray([x["diameter"] for x in usable], dtype=float),
            "tentative": np.asarray([x["tentative"] for x in usable], dtype=bool),
            "identifier": np.asarray([x["identifier"] for x in usable], dtype=int),
            "cpa_time": np.asarray([x["cpa_time"] for x in usable], dtype=float),
            "cpa_clearance": np.asarray([x["cpa_clearance"] for x in usable], dtype=float),
            "approaching": np.asarray([x["approaching"] for x in usable], dtype=bool),
            "cpa_vector": np.asarray([x["cpa_vector"] for x in usable], dtype=float),
            "relative_position": np.asarray([x["relative_position"] for x in usable], dtype=float),
            "relative_velocity": np.asarray([x["relative_velocity"] for x in usable], dtype=float),
        }

    def _candidate_specs(
        self, goal_direction, track_arrays, position, velocity
    ):
        cfg = self.config
        goal = unit(goal_direction)
        side1, side2 = perpendicular_basis(goal, self._side_basis1)
        self._side_basis1 = side1.copy()

        specs = [
            ("beeline", np.zeros(3), 1.00, False),
            ("slow_forward", np.zeros(3), 0.65, False),
            ("hold_relative", np.zeros(3), 0.00, True),
        ]

        for offset in (0.55, 1.45):
            specs.extend([
                (f"side1+{offset:.2f}", offset * side1, 1.00, False),
                (f"side1-{offset:.2f}", -offset * side1, 1.00, False),
                (f"side2+{offset:.2f}", offset * side2, 1.00, False),
                (f"side2-{offset:.2f}", -offset * side2, 1.00, False),
            ])

        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                specs.append((
                    f"diagonal_{int(s1):+d}_{int(s2):+d}",
                    0.90 * (s1 * side1 + s2 * side2),
                    1.00,
                    False,
                ))

        specs.extend([
            ("lateral1+", 2.2 * side1, 0.90, False),
            ("lateral1-", -2.2 * side1, 0.90, False),
            ("lateral2+", 2.2 * side2, 0.90, False),
            ("lateral2-", -2.2 * side2, 0.90, False),
        ])

        # Keep the previous selected maneuver available to prevent side-flips
        # when small radar updates make two paths nearly equal in cost.
        if norm(self.last_bias) > 0.05:
            specs.append((
                "previous_maneuver",
                self.last_bias.copy(),
                float(self.last_speed_scale),
                False,
            ))

        if track_arrays is not None:
            order = np.argsort(track_arrays["cpa_clearance"])
            added = 0
            for index in order:
                if added >= int(cfg.threat_candidate_count):
                    break
                if not track_arrays["approaching"][index]:
                    continue
                if (
                    track_arrays["cpa_clearance"][index]
                    > float(cfg.threat_candidate_threshold_m)
                ):
                    continue

                cpa_vector = track_arrays["cpa_vector"][index]
                relative_velocity = track_arrays["relative_velocity"][index]
                away = (
                    -unit(cpa_vector)
                    if norm(cpa_vector) > 1.0e-6
                    else -unit(track_arrays["relative_position"][index])
                )
                cross = np.cross(relative_velocity, cpa_vector)
                if norm(cross) <= 1.0e-6:
                    threat_side, _ = perpendicular_basis(relative_velocity)
                else:
                    threat_side = unit(cross)

                track_id = int(track_arrays["identifier"][index])
                specs.extend([
                    (f"threat{track_id}_away", 1.8 * away, 1.00, False),
                    (
                        f"threat{track_id}_cross+",
                        1.25 * away + 0.90 * threat_side,
                        1.00,
                        False,
                    ),
                    (
                        f"threat{track_id}_cross-",
                        1.25 * away - 0.90 * threat_side,
                        1.00,
                        False,
                    ),
                    (
                        f"threat{track_id}_slow_away",
                        1.8 * away,
                        0.70,
                        False,
                    ),
                ])
                added += 1

        unique = []
        seen = set()
        for name, bias, speed_scale, hold in specs:
            bias = np.asarray(bias, dtype=float).reshape(3)
            if not np.all(np.isfinite(bias)):
                continue
            key = (
                tuple(np.round(bias, 3)),
                round(float(speed_scale), 3),
                bool(hold),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append((
                name,
                bias,
                float(np.clip(speed_scale, 0.0, 1.0)),
                bool(hold),
            ))
        return unique

    def _limit_batch(self, requested, max_accel, acceleration_limiter):
        requested = np.asarray(requested, dtype=float)
        if acceleration_limiter is not None:
            try:
                limited = np.asarray(
                    acceleration_limiter(requested), dtype=float
                )
                if limited.shape == requested.shape and np.all(np.isfinite(limited)):
                    return limited
            except Exception:
                return np.zeros_like(requested)
            return np.zeros_like(requested)
        return np.asarray([
            limit_norm(row, max_accel) for row in requested
        ], dtype=float)

    def _rollout_candidates(
        self,
        position,
        velocity,
        target_position,
        target_velocity,
        timestamp,
        biases,
        speed_scales,
        holds,
        max_accel,
        track_arrays=None,
        acceleration_limiter=None,
    ):
        cfg = self.config
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_velocity = np.asarray(target_velocity, dtype=float).reshape(3)
        biases = np.asarray(biases, dtype=float).reshape(-1, 3)
        speed_scales = np.asarray(speed_scales, dtype=float).reshape(-1)
        holds = np.asarray(holds, dtype=bool).reshape(-1)
        count = len(biases)

        p = np.repeat(position[None, :], count, axis=0)
        v = np.repeat(velocity[None, :], count, axis=0)
        risk_cost = np.zeros(count, dtype=float)
        minimum_clearance = np.full(count, float("inf"), dtype=float)
        closest_time = np.full(count, float("nan"), dtype=float)
        closest_track_id = np.full(count, -1, dtype=int)

        steps = max(1, int(np.ceil(
            float(cfg.horizon_s) / max(float(cfg.step_s), 1.0e-3)
        )))
        dt = float(cfg.horizon_s) / steps

        if track_arrays is not None:
            debris_positions = track_arrays["position"]
            debris_velocities = track_arrays["velocity"]
            ages = track_arrays["age"]
            diameters = track_arrays["diameter"]
            tentative = track_arrays["tentative"]
            identifiers = track_arrays["identifier"]
            velocity_uncertainty = np.where(
                tentative,
                float(cfg.tentative_velocity_uncertainty_mps),
                float(cfg.confirmed_velocity_uncertainty_mps),
            )
            base_radii = (
                float(cfg.shuttle_radius_m)
                + 0.5 * diameters
                + float(cfg.clearance_margin_m)
                + float(cfg.initial_uncertainty_m)
                + tentative.astype(float) * float(cfg.new_track_uncertainty_m)
            )

        for step in range(steps):
            tau0 = step * dt
            tau1 = (step + 1) * dt
            predicted_target = target_position + target_velocity * tau0
            relative_to_target = predicted_target[None, :] - p
            distances = np.linalg.norm(relative_to_target, axis=1)
            safe_distances = np.maximum(distances, 1.0e-9)
            goal_directions = relative_to_target / safe_distances[:, None]

            outside = distances > float(cfg.capture_radius_m)
            remaining = np.maximum(
                distances - float(cfg.capture_radius_m), 0.0
            )
            braking_accel = max(
                float(max_accel) * float(cfg.braking_accel_fraction),
                1.0e-5,
            )
            speeds = np.sqrt(
                float(cfg.terminal_speed_mps) ** 2
                + 2.0 * braking_accel * remaining
            )
            speeds = np.minimum(speeds, float(cfg.goal_speed))
            speeds *= speed_scales

            bias_fade = np.clip(
                remaining / max(float(cfg.bias_fade_distance_m), 1.0),
                0.0,
                1.0,
            )
            # CRITICAL MPC CONSISTENCY FIX:
            # desired_target_velocity() removes the component of each bias
            # parallel to the instantaneous goal direction before execution.
            # The old rollout omitted that projection, so it evaluated a
            # different trajectory from the command that was actually sent.
            # Project each candidate bias into the plane normal to the current
            # target line-of-sight at every rollout step, exactly as guidance does.
            bias_parallel = np.einsum(
                "ni,ni->n", biases, goal_directions
            )
            projected_biases = (
                biases - bias_parallel[:, None] * goal_directions
            )
            biased_directions = goal_directions + (
                projected_biases * bias_fade[:, None]
            )
            biased_norms = np.linalg.norm(biased_directions, axis=1)
            biased_directions = np.divide(
                biased_directions,
                np.maximum(biased_norms[:, None], 1.0e-9),
            )
            desired_relative = speeds[:, None] * biased_directions

            # Terminal capture is the same for every avoidance candidate:
            # settle relative position and velocity instead of orbiting past.
            capture_directions = goal_directions.copy()
            zero_range = distances <= 1.0e-9
            capture_directions[zero_range] = np.array([1.0, 0.0, 0.0])
            standoff_error = (
                distances - float(cfg.capture_standoff_m)
            )
            capture_velocity = (
                float(cfg.capture_gain_per_s)
                * standoff_error[:, None]
                * capture_directions
            )
            capture_norms = np.linalg.norm(capture_velocity, axis=1)
            capture_scale = np.minimum(
                1.0,
                float(cfg.capture_max_speed_mps)
                / np.maximum(capture_norms, 1.0e-9),
            )
            capture_velocity *= capture_scale[:, None]
            desired_relative[outside == False] = capture_velocity[outside == False]
            desired_relative[holds] = 0.0

            desired_world_velocity = (
                target_velocity[None, :] + desired_relative
            )
            requested = float(cfg.velocity_gain) * (
                desired_world_velocity - v
            )

            latency = float(cfg.control_latency_s)
            if tau0 < latency:
                active_fraction = float(np.clip(
                    (tau1 - latency) / max(dt, 1.0e-9), 0.0, 1.0
                ))
                requested *= active_fraction

            acceleration = self._limit_batch(
                requested, max_accel, acceleration_limiter
            )
            next_position = p + v * dt + 0.5 * acceleration * dt * dt
            next_velocity = v + acceleration * dt

            if track_arrays is not None:
                debris_start = (
                    debris_positions + debris_velocities * tau0
                )
                debris_end = (
                    debris_positions + debris_velocities * tau1
                )
                relative_start = p[:, None, :] - debris_start[None, :, :]
                relative_end = (
                    next_position[:, None, :] - debris_end[None, :, :]
                )
                delta = relative_end - relative_start
                denominator = np.einsum("mni,mni->mn", delta, delta)
                numerator = -np.einsum("mni,mni->mn", relative_start, delta)
                fractions = np.zeros_like(denominator)
                moving = denominator > 1.0e-9
                fractions[moving] = np.clip(
                    numerator[moving] / denominator[moving], 0.0, 1.0
                )
                closest = relative_start + fractions[:, :, None] * delta
                separations = np.linalg.norm(closest, axis=2)
                time_at_closest = tau0 + fractions * dt
                total_age = ages[None, :] + time_at_closest
                uncertainty = (
                    float(cfg.uncertainty_growth_mps) * total_age
                    + 0.5 * float(cfg.uncertainty_accel_mps2) * total_age ** 2
                    + velocity_uncertainty[None, :] * total_age
                )
                clearances = separations - (
                    base_radii[None, :] + uncertainty
                )

                local_indices = np.argmin(clearances, axis=1)
                local_minima = clearances[
                    np.arange(count), local_indices
                ]
                improved = local_minima < minimum_clearance
                if np.any(improved):
                    rows = np.where(improved)[0]
                    cols = local_indices[improved]
                    minimum_clearance[improved] = local_minima[improved]
                    closest_time[improved] = time_at_closest[rows, cols]
                    closest_track_id[improved] = identifiers[cols]

                hard = float(cfg.minimum_safe_clearance_m)
                preferred = float(cfg.replan_buffer_m)
                collision = clearances < hard
                near = (
                    (clearances >= hard)
                    & (clearances < preferred)
                )
                risk_cost += np.sum(
                    5000.0
                    * np.square(np.maximum(0.0, hard - clearances))
                    * collision,
                    axis=1,
                )
                risk_cost += np.sum(
                    35.0
                    * np.square(np.maximum(0.0, preferred - clearances))
                    * near,
                    axis=1,
                )

            p = next_position
            v = next_velocity

        target_final = (
            target_position + target_velocity * float(cfg.horizon_s)
        )
        terminal_range = np.linalg.norm(target_final[None, :] - p, axis=1)
        terminal_distance = np.abs(
            terminal_range - float(cfg.capture_standoff_m)
        )
        terminal_relative_speed = np.linalg.norm(
            v - target_velocity[None, :], axis=1
        )
        return {
            "risk": risk_cost,
            "distance": terminal_distance,
            "terminal_relative_speed": terminal_relative_speed,
            "clearance": minimum_clearance,
            "closest_time_s": closest_time,
            "closest_track_id": closest_track_id,
            "terminal_position": p,
            "terminal_velocity": v,
        }

    def plan(
        self,
        position,
        velocity,
        target_position,
        target_velocity,
        tracks,
        timestamp,
        max_accel=None,
        acceleration_limiter=None,
    ):
        cfg = self.config
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_velocity = np.asarray(target_velocity, dtype=float).reshape(3)
        if max_accel is None:
            max_accel = float(cfg.predicted_max_accel)
        max_accel = max(float(max_accel), 1.0e-5)

        goal_vector = target_position - position
        goal_direction = unit(goal_vector)
        track_arrays = self._prepare_track_arrays(
            tracks, position, velocity, timestamp
        )
        specs = self._candidate_specs(
            goal_direction, track_arrays, position, velocity
        )
        names = [item[0] for item in specs]
        biases = np.asarray([item[1] for item in specs], dtype=float)
        speed_scales = np.asarray([item[2] for item in specs], dtype=float)
        holds = np.asarray([item[3] for item in specs], dtype=bool)

        rollout = self._rollout_candidates(
            position,
            velocity,
            target_position,
            target_velocity,
            timestamp,
            biases,
            speed_scales,
            holds,
            max_accel,
            track_arrays=track_arrays,
            acceleration_limiter=acceleration_limiter,
        )

        initial_distance = norm(goal_vector)
        costs = (
            rollout["risk"]
            + 2.0 * rollout["distance"]
            + 3.0 * rollout["terminal_relative_speed"]
            + 5.0 * np.linalg.norm(biases, axis=1)
            + 5.0 * np.linalg.norm(
                biases - self.last_bias[None, :], axis=1
            )
            + 12.0 * (1.0 - speed_scales)
        )

        evaluated = []
        for index, name in enumerate(names):
            evaluated.append({
                "index": index,
                "name": name,
                "bias": biases[index],
                "speed_scale": float(speed_scales[index]),
                "hold": bool(holds[index]),
                "cost": float(costs[index]),
                "risk": float(rollout["risk"][index]),
                "distance": float(rollout["distance"][index]),
                "terminal_relative_speed": float(
                    rollout["terminal_relative_speed"][index]
                ),
                "clearance": float(rollout["clearance"][index]),
                "closest_time_s": float(rollout["closest_time_s"][index]),
                "closest_track_id": int(
                    rollout["closest_track_id"][index]
                ),
                "terminal_position": rollout["terminal_position"][index],
                "terminal_velocity": rollout["terminal_velocity"][index],
            })

        if not evaluated:
            fallback = limit_norm(target_velocity - velocity, max_accel)
            metrics = {
                "risk": float("inf"),
                "distance": initial_distance,
                "terminal_relative_speed": norm(velocity - target_velocity),
                "clearance": float("-inf"),
                "closest_time_s": 0.0,
                "closest_track_id": -1,
                "emergency": True,
                "safety_state": "PLANNER_EMPTY",
                "candidate_count": 0,
                "safe_candidate_count": 0,
                "preferred_candidate_count": 0,
                "best_candidate_clearance": float("-inf"),
                "selected_candidate": "fallback_brake",
                "selected_bias": [0.0, 0.0, 0.0],
            }
            self.last_choice = "fallback_brake"
            self.last_metrics = metrics
            return fallback, metrics

        nominal = evaluated[0]
        best_clearance = max(item["clearance"] for item in evaluated)
        hard_floor = float(cfg.minimum_safe_clearance_m)
        preferred_floor = float(cfg.replan_buffer_m)
        recovery_floor = preferred_floor + float(cfg.recovery_hysteresis_m)

        if nominal["clearance"] < preferred_floor:
            self.avoidance_active = True
            self._clear_cycles = 0
        elif self.avoidance_active:
            if nominal["clearance"] >= recovery_floor:
                self._clear_cycles += 1
                if self._clear_cycles >= int(cfg.recovery_release_cycles):
                    self.avoidance_active = False
                    self._clear_cycles = 0
            else:
                self._clear_cycles = 0

        # Do not immediately return to the nominal path after one clear
        # sample. Require the configured number of consecutive recovery
        # samples so the selected lateral route does not chatter.
        nominal_allowed = not self.avoidance_active
        if nominal["clearance"] >= preferred_floor and nominal_allowed:
            chosen = nominal
            safety_state = "CLEAR"
        else:
            preferred = [
                item for item in evaluated
                if item["clearance"] >= preferred_floor
                and (
                    item is not nominal
                    or nominal_allowed
                )
            ]
            safe = [
                item for item in evaluated
                if item["clearance"] >= hard_floor
            ]

            if preferred:
                # Once a trajectory clears the preferred safety envelope,
                # rank ALL such trajectories by mission cost. The old code
                # first shortlisted only candidates within 5 m of the single
                # largest clearance. That made a stationary/hold trajectory
                # beat a perfectly safe forward route by hundreds of metres
                # of extra clearance, causing the vehicle to stop making
                # progress. Clearance is a constraint here, not the objective.
                chosen = min(preferred, key=lambda x: x["cost"])
                safety_state = "BUFFERED"
            elif safe:
                # No preferred-buffer path exists. Stay above the hard floor,
                # then trade risk against progress using the same mission cost.
                chosen = min(safe, key=lambda x: x["cost"])
                safety_state = "CONSTRAINED"
            else:
                # No predicted candidate meets the hard envelope. Choose the
                # trajectory with maximum predicted clearance and flag it.
                chosen = min(
                    evaluated,
                    key=lambda x: (-x["clearance"], x["cost"]),
                )
                # A trajectory may be unsafe somewhere in the long planning
                # horizon without representing an immediate collision. Keep
                # that condition distinct from an imminent emergency so the
                # controller does not treat a 30-40 s forecast as a current
                # impact. The selected trajectory still maximizes predicted
                # clearance over the full horizon.
                safety_state = "NO_SAFE_HORIZON"

        no_safe_horizon = safety_state == "NO_SAFE_HORIZON"
        chosen_closest_time = float(chosen["closest_time_s"])
        emergency = bool(
            no_safe_horizon
            and np.isfinite(chosen_closest_time)
            and chosen_closest_time <= float(cfg.critical_replan_ttc_s)
        )
        if emergency:
            safety_state = "UNAVOIDABLE"
        chosen_index = int(chosen["index"])
        chosen_bias = np.asarray(chosen["bias"], dtype=float)
        chosen_speed_scale = float(chosen["speed_scale"])
        desired_velocity, desired_relative_speed, current_distance = (
            desired_target_velocity(
                position,
                target_position,
                target_velocity,
                cfg,
                max_accel,
                direction_bias=chosen_bias,
                speed_scale=chosen_speed_scale,
                hold=chosen["hold"],
            )
        )
        requested_acceleration = float(cfg.velocity_gain) * (
            desired_velocity - velocity
        )

        if acceleration_limiter is not None:
            try:
                acceleration = np.asarray(
                    acceleration_limiter(requested_acceleration),
                    dtype=float,
                ).reshape(3)
                if not np.all(np.isfinite(acceleration)):
                    acceleration = np.zeros(3, dtype=float)
            except Exception:
                acceleration = np.zeros(3, dtype=float)
        else:
            acceleration = limit_norm(
                requested_acceleration, max_accel
            )

        metrics = {
            "risk": chosen["risk"],
            "distance": chosen["distance"],
            "terminal_relative_speed": chosen["terminal_relative_speed"],
            "clearance": chosen["clearance"],
            "closest_time_s": chosen["closest_time_s"],
            "closest_track_id": chosen["closest_track_id"],
            "emergency": emergency,
            "no_safe_horizon": no_safe_horizon,
            "safety_state": safety_state,
            "candidate_count": len(evaluated),
            "safe_candidate_count": sum(
                item["clearance"] >= hard_floor for item in evaluated
            ),
            "preferred_candidate_count": sum(
                item["clearance"] >= preferred_floor for item in evaluated
            ),
            "best_candidate_clearance": best_clearance,
            "selected_candidate": chosen["name"],
            "selected_bias": chosen_bias.tolist(),
            "selected_speed_scale": chosen_speed_scale,
            "selected_hold": bool(chosen["hold"]),
            "emergency_clearance_margin_m": float(chosen["clearance"] - hard_floor),
            "desired_velocity": desired_velocity.tolist(),
            "desired_relative_speed_mps": desired_relative_speed,
            "target_distance_m": current_distance,
            "planning_track_count": (
                0 if track_arrays is None else len(track_arrays["position"])
            ),
            "avoidance_active": self.avoidance_active,
            "avoidance_clear_cycles": self._clear_cycles,
            "nominal_clearance_m": nominal["clearance"],
            "initial_target_distance_m": initial_distance,
        }

        self.last_choice = chosen["name"]
        self.last_bias = chosen_bias.copy()
        self.last_speed_scale = chosen_speed_scale
        self.last_metrics = metrics
        return acceleration, metrics

    def rollout(
        self,
        position,
        velocity,
        target_position,
        target_velocity,
        tracks,
        timestamp,
        desired_velocity,
        max_accel,
        track_arrays=None,
        acceleration_limiter=None,
    ):
        """Evaluate a fixed world-velocity command over the configured horizon."""
        cfg = self.config
        p = np.asarray(position, dtype=float).reshape(3).copy()
        v = np.asarray(velocity, dtype=float).reshape(3).copy()
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_velocity = np.asarray(target_velocity, dtype=float).reshape(3)
        desired_velocity = np.asarray(desired_velocity, dtype=float).reshape(3)
        if track_arrays is None:
            track_arrays = self._prepare_track_arrays(
                tracks, p, v, timestamp
            )

        minimum_clearance = float("inf")
        closest_time = float("nan")
        closest_track_id = -1
        risk_cost = 0.0
        steps = max(1, int(np.ceil(
            float(cfg.horizon_s) / max(float(cfg.step_s), 1.0e-3)
        )))
        dt = float(cfg.horizon_s) / steps

        if track_arrays is not None:
            positions = track_arrays["position"]
            velocities = track_arrays["velocity"]
            ages = track_arrays["age"]
            diameters = track_arrays["diameter"]
            tentative = track_arrays["tentative"]
            identifiers = track_arrays["identifier"]
            velocity_uncertainty = np.where(
                tentative,
                float(cfg.tentative_velocity_uncertainty_mps),
                float(cfg.confirmed_velocity_uncertainty_mps),
            )
            base_radii = (
                float(cfg.shuttle_radius_m)
                + 0.5 * diameters
                + float(cfg.clearance_margin_m)
                + float(cfg.initial_uncertainty_m)
                + tentative.astype(float) * float(cfg.new_track_uncertainty_m)
            )

        for step in range(steps):
            tau0 = step * dt
            tau1 = (step + 1) * dt
            requested = float(cfg.velocity_gain) * (desired_velocity - v)
            if tau0 < float(cfg.control_latency_s):
                fraction = float(np.clip(
                    (tau1 - float(cfg.control_latency_s)) / max(dt, 1.0e-9),
                    0.0, 1.0
                ))
                requested *= fraction
            acceleration = self._limit_batch(
                requested.reshape(1, 3), max_accel, acceleration_limiter
            )[0]
            next_position = p + v * dt + 0.5 * acceleration * dt * dt
            next_velocity = v + acceleration * dt

            if track_arrays is not None:
                debris_start = positions + velocities * tau0
                debris_end = positions + velocities * tau1
                relative_start = p[None, :] - debris_start
                relative_end = next_position[None, :] - debris_end
                delta = relative_end - relative_start
                denominator = np.einsum("ij,ij->i", delta, delta)
                numerator = -np.einsum("ij,ij->i", relative_start, delta)
                fractions = np.zeros_like(denominator)
                moving = denominator > 1.0e-9
                fractions[moving] = np.clip(
                    numerator[moving] / denominator[moving], 0.0, 1.0
                )
                closest = relative_start + fractions[:, None] * delta
                separations = np.linalg.norm(closest, axis=1)
                times = tau0 + fractions * dt
                total_age = ages + times
                uncertainty = (
                    float(cfg.uncertainty_growth_mps) * total_age
                    + 0.5 * float(cfg.uncertainty_accel_mps2) * total_age ** 2
                    + velocity_uncertainty * total_age
                )
                clearances = separations - (base_radii + uncertainty)
                local_index = int(np.argmin(clearances))
                local_clearance = float(clearances[local_index])
                if local_clearance < minimum_clearance:
                    minimum_clearance = local_clearance
                    closest_time = float(times[local_index])
                    closest_track_id = int(identifiers[local_index])
                hard = float(cfg.minimum_safe_clearance_m)
                preferred = float(cfg.replan_buffer_m)
                risk_cost += float(np.sum(
                    5000.0 * np.square(np.maximum(0.0, hard - clearances))
                    * (clearances < hard)
                ))
                risk_cost += float(np.sum(
                    35.0 * np.square(np.maximum(0.0, preferred - clearances))
                    * ((clearances >= hard) & (clearances < preferred))
                ))

            p, v = next_position, next_velocity

        target_final = target_position + target_velocity * float(cfg.horizon_s)
        return {
            "risk": float(risk_cost),
            "distance": abs(
                norm(target_final - p) - float(cfg.capture_standoff_m)
            ),
            "terminal_relative_speed": norm(v - target_velocity),
            "clearance": float(minimum_clearance),
            "closest_time_s": float(closest_time),
            "closest_track_id": int(closest_track_id),
        }


def beeline_guidance(
    position,
    velocity,
    target_position,
    config,
    target_velocity=None,
    desired_speed=None,
    max_accel=None,
):
    """Legacy helper retained for existing notebooks and tests."""
    if target_velocity is None:
        target_velocity = np.zeros(3, dtype=float)
    if max_accel is None:
        max_accel = float(config.predicted_max_accel)
    if desired_speed is None:
        desired_speed = approach_speed(
            norm(np.asarray(target_position) - np.asarray(position)),
            config,
            max_accel,
        )
    direction = unit(np.asarray(target_position) - np.asarray(position))
    desired = np.asarray(target_velocity, dtype=float) + float(desired_speed) * direction
    return limit_norm(
        float(config.velocity_gain)
        * (desired - np.asarray(velocity, dtype=float)),
        max_accel,
    )
