"""Deterministic controller-level simulations using the competition initial asset.

These tests are not a substitute for a live SpaceTeams PRO run. They verify
that the planner, target capture law, read-only thruster envelope and
clearance-first avoidance cooperate under a simplified constant-velocity
environment.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.planning import PredictivePlanner
from phobos_nav.actuator_model import (
    read_thruster_csv,
    build_activity_matrix,
    compute_axis_authorities,
    limit_acceleration_body,
    limit_acceleration_body_batch,
)


def competition_limiter():
    fixture = Path(__file__).resolve().parent / "fixtures" / "SpaceBuggyJetConfigNoMains.csv"
    records = read_thruster_csv(fixture)
    inertia = np.array([
        [3025.8, -171.7, 461.7],
        [-171.7, 5927.5, 313.6],
        [461.7, 313.6, 6756.0],
    ])
    activity = build_activity_matrix(records, 3388.7, inertia)
    positive, negative, valid, _ = compute_axis_authorities(activity)
    if not valid:
        raise RuntimeError("Competition thruster authority fixture is infeasible")

    # SpaceBuggy's initial scalar-last quaternion from COMPETITION_SIM.simconf.
    q = np.array([
        0.9848614817870478,
        0.16855626602590107,
        -0.03977804037301115,
        0.007372542282663707,
    ])
    x, y, z, w = q
    body_to_phobos = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    phobos_to_body = body_to_phobos.T

    def limiter(command_phobos):
        command = np.asarray(command_phobos, dtype=float)
        if command.ndim == 2:
            body = command @ phobos_to_body.T
            limited = limit_acceleration_body_batch(
                body, positive, negative,
                safety_factor=CONFIG.authority_safety_factor,
            )
            return limited @ phobos_to_body
        body = phobos_to_body @ command.reshape(3)
        limited, _ = limit_acceleration_body(
            body, positive, negative,
            safety_factor=CONFIG.authority_safety_factor,
        )
        return phobos_to_body.T @ limited

    return limiter


class SyntheticMissionTests(unittest.TestCase):
    def setUp(self):
        self.position0 = np.array([
            -651.0228484243066, 95.19902620579852, -8239.668001620705
        ])
        self.target = np.array([-850.0, 120.0, -9530.0])
        self.target_velocity = np.zeros(3)
        self.limiter = competition_limiter()

    def test_unobstructed_guidance_reaches_and_holds_inside_scoring_radius(self):
        planner = PredictivePlanner(CONFIG)
        position = self.position0.copy()
        velocity = np.zeros(3)
        dt = 1.0
        elapsed = 0.0
        capture_time = None

        for _ in range(260):
            relative = self.target - position
            distance = float(np.linalg.norm(relative))
            direction = relative / max(distance, 1.0e-9)
            forward_authority = float(np.linalg.norm(
                self.limiter(direction * 0.5)
            ))
            reverse_authority = float(np.linalg.norm(
                self.limiter(-direction * 0.5)
            ))
            available_accel = min(forward_authority, reverse_authority)
            command, _ = planner.plan(
                position, velocity, self.target, self.target_velocity,
                [], elapsed, max_accel=available_accel,
                acceleration_limiter=self.limiter,
            )
            position = position + velocity * dt + 0.5 * command * dt * dt
            velocity = velocity + command * dt
            elapsed += dt
            current_range = float(np.linalg.norm(self.target - position))
            if capture_time is None and current_range <= 50.0:
                capture_time = elapsed

        final_range = float(np.linalg.norm(self.target - position))
        self.assertIsNotNone(capture_time)
        self.assertLess(capture_time, 180.0)
        self.assertLessEqual(final_range, 50.0)
        self.assertGreater(final_range, 20.0)
        self.assertLess(final_range, 40.0)
        self.assertLess(np.linalg.norm(velocity), 0.20)

    def test_predictive_avoidance_clears_debris_and_still_holds_at_target(self):
        planner = PredictivePlanner(CONFIG)
        position = self.position0.copy()
        velocity = np.zeros(3)
        dt = 1.0
        elapsed = 0.0
        direction = (self.target - self.position0)
        direction /= np.linalg.norm(direction)
        debris_position = self.position0 + direction * 650.0
        minimum_center_separation = float("inf")
        capture_time = None

        for _ in range(260):
            relative = self.target - position
            distance = float(np.linalg.norm(relative))
            goal_direction = relative / max(distance, 1.0e-9)
            forward_authority = float(np.linalg.norm(
                self.limiter(goal_direction * 0.5)
            ))
            reverse_authority = float(np.linalg.norm(
                self.limiter(-goal_direction * 0.5)
            ))
            available_accel = min(forward_authority, reverse_authority)
            # A fresh radar update each planner step; the debris is stationary
            # in the Phobos frame for this deterministic regression case.
            track = SimpleNamespace(
                identifier=7,
                position=debris_position.copy(),
                velocity=np.zeros(3),
                diameter=30.0,
                timestamp=elapsed,
                hits=5,
                confirmed=True,
                position_at=lambda _time, p=debris_position: p.copy(),
            )
            command, metrics = planner.plan(
                position, velocity, self.target, self.target_velocity,
                [track], elapsed, max_accel=available_accel,
                acceleration_limiter=self.limiter,
            )
            next_position = (
                position + velocity * dt + 0.5 * command * dt * dt
            )
            next_velocity = velocity + command * dt
            minimum_center_separation = min(
                minimum_center_separation,
                float(np.linalg.norm(next_position - debris_position)),
            )
            position, velocity = next_position, next_velocity
            elapsed += dt
            current_range = float(np.linalg.norm(self.target - position))
            if capture_time is None and current_range <= 50.0:
                capture_time = elapsed

        physical_contact_radius = (
            CONFIG.shuttle_radius_m + 0.5 * 30.0
        )
        final_range = float(np.linalg.norm(self.target - position))
        self.assertGreater(
            minimum_center_separation,
            physical_contact_radius + CONFIG.clearance_margin_m,
        )
        self.assertIsNotNone(capture_time)
        self.assertLess(capture_time, 230.0)
        self.assertLessEqual(final_range, 50.0)
        self.assertGreater(final_range, 20.0)
        self.assertLess(final_range, 40.0)
        self.assertLess(np.linalg.norm(velocity), 0.20)
        self.assertGreater(metrics["candidate_count"], 10)



if __name__ == "__main__":
    unittest.main(verbosity=2)
