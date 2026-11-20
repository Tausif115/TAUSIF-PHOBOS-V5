from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.guidance import desired_target_velocity
from phobos_nav.planning import PredictivePlanner


class MPCExecutionConsistencyTests(unittest.TestCase):
    def test_rollout_first_step_matches_guidance_command_with_parallel_bias(self):
        """The MPC must score the same first-step acceleration it executes."""
        cfg = replace(CONFIG, horizon_s=1.0, step_s=1.0, control_latency_s=0.0)
        planner = PredictivePlanner(cfg)
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.2, -0.1, 0.0])
        target_position = np.array([100.0, 0.0, 0.0])
        target_velocity = np.zeros(3)
        # This candidate has both parallel and lateral components. The
        # executed guidance removes the parallel component; rollout must too.
        bias = np.array([1.3, 1.1, -0.4])
        speed_scale = np.array([0.8])
        holds = np.array([False])
        max_accel = 0.35

        def limiter(command):
            command = np.asarray(command, dtype=float)
            if command.ndim == 2:
                norms = np.linalg.norm(command, axis=1)
                scales = np.minimum(1.0, max_accel / np.maximum(norms, 1e-12))
                return command * scales[:, None]
            magnitude = np.linalg.norm(command)
            return command * min(1.0, max_accel / max(magnitude, 1e-12))

        rollout = planner._rollout_candidates(
            position, velocity, target_position, target_velocity, 0.0,
            bias.reshape(1, 3), speed_scale, holds, max_accel,
            track_arrays=None, acceleration_limiter=limiter,
        )

        desired_velocity, _, _ = desired_target_velocity(
            position, target_position, target_velocity, cfg, max_accel,
            direction_bias=bias, speed_scale=float(speed_scale[0]), hold=False,
        )
        requested = cfg.velocity_gain * (desired_velocity - velocity)
        actual_command = limiter(requested)
        expected_position = position + velocity + 0.5 * actual_command
        np.testing.assert_allclose(
            rollout["terminal_position"][0], expected_position,
            rtol=1e-10, atol=1e-10,
        )

    def test_rollout_and_guidance_agree_for_pure_goal_parallel_bias(self):
        cfg = replace(CONFIG, horizon_s=1.0, step_s=1.0, control_latency_s=0.0)
        planner = PredictivePlanner(cfg)
        p = np.zeros(3)
        v = np.zeros(3)
        target = np.array([100.0, 0.0, 0.0])
        bias = np.array([2.0, 0.0, 0.0])
        rollout = planner._rollout_candidates(
            p, v, target, np.zeros(3), 0.0,
            bias.reshape(1, 3), np.array([1.0]), np.array([False]),
            0.3, track_arrays=None,
        )
        desired, _, _ = desired_target_velocity(
            p, target, np.zeros(3), cfg, 0.3,
            direction_bias=bias, speed_scale=1.0, hold=False,
        )
        expected_accel = cfg.velocity_gain * desired
        expected_accel_norm = np.linalg.norm(expected_accel)
        if expected_accel_norm > 0.3:
            expected_accel *= 0.3 / expected_accel_norm
        expected_position = p + 0.5 * expected_accel
        np.testing.assert_allclose(
            rollout["terminal_position"][0], expected_position,
            rtol=1e-10, atol=1e-10,
        )


    def test_unavoidable_head_on_threat_commands_lateral_escape(self):
        """When every path is unsafe, the selected escape must be executed."""
        from types import SimpleNamespace
        cfg = CONFIG
        planner = PredictivePlanner(cfg)
        position = np.zeros(3)
        velocity = np.array([3.0, 0.0, 0.0])
        target_position = np.array([1000.0, 0.0, 0.0])
        threat = SimpleNamespace(
            identifier=42,
            position=np.array([80.0, 0.0, 0.0]),
            velocity=np.array([-12.0, 0.0, 0.0]),
            diameter=20.0,
            timestamp=0.0,
            hits=5,
            confirmed=True,
            position_at=lambda t: np.array([80.0 - 12.0 * t, 0.0, 0.0]),
        )

        def limiter(command):
            command = np.asarray(command, dtype=float)
            if command.ndim == 2:
                n = np.linalg.norm(command, axis=1)
                return command * np.minimum(1.0, 0.25 / np.maximum(n, 1e-12))[:, None]
            n = np.linalg.norm(command)
            return command * min(1.0, 0.25 / max(n, 1e-12))

        command, metrics = planner.plan(
            position, velocity, target_position, np.zeros(3), [threat], 0.0,
            max_accel=0.25, acceleration_limiter=limiter,
        )
        self.assertTrue(metrics["emergency"])
        self.assertGreater(np.linalg.norm(command[1:]), 1.0e-3)
        # A head-on threat lies on +X; a real escape command must contain
        # lateral acceleration rather than only a forward/hold command.
        self.assertGreater(abs(command[2]), 1.0e-3)
        self.assertFalse(metrics.get("selected_hold", False))



    def test_far_future_no_safe_candidate_is_not_false_immediate_emergency(self):
        """A no-safe result 18 s ahead is a warning, not an immediate impact."""
        from types import SimpleNamespace
        planner = PredictivePlanner(CONFIG)
        threat = SimpleNamespace(
            identifier=43,
            position=np.array([200.0, 0.0, 0.0]),
            velocity=np.array([-10.0, 0.0, 0.0]),
            diameter=20.0,
            timestamp=0.0,
            hits=5,
            confirmed=True,
            position_at=lambda t: np.array([200.0 - 10.0 * t, 0.0, 0.0]),
        )
        command, metrics = planner.plan(
            np.zeros(3), np.zeros(3), np.array([1000.0, 0.0, 0.0]),
            np.zeros(3), [threat], 0.0, max_accel=0.2,
        )
        self.assertTrue(metrics["no_safe_horizon"])
        self.assertEqual(metrics["safety_state"], "NO_SAFE_HORIZON")
        self.assertFalse(metrics["emergency"])
        self.assertGreater(metrics["closest_time_s"], CONFIG.critical_replan_ttc_s)
        self.assertTrue(np.all(np.isfinite(command)))



if __name__ == "__main__":
    unittest.main(verbosity=2)
