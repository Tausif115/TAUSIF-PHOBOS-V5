import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.geometry import perpendicular_basis, norm
from phobos_nav.guidance import desired_target_velocity
from phobos_nav.radar import RadarClockSynchronizer, decode_radar
from phobos_nav.radar_diagnostics import format_radar_diagnostic
from phobos_nav.tracking import StateHistory, DebrisTrack, DebrisTracker
from phobos_nav.planning import PredictivePlanner
from phobos_nav.actuator_model import (
    read_thruster_csv,
    build_activity_matrix,
    compute_axis_authorities,
    limit_acceleration_body,
    limit_acceleration_body_batch,
)


class GeometryTests(unittest.TestCase):
    def test_perpendicular_basis_is_orthonormal(self):
        for direction in (
            [1, 0, 0], [0, -1, 0], [0, 0, 1], [0.1, -0.99, 0.05]
        ):
            side1, side2 = perpendicular_basis(direction)
            forward = np.asarray(direction, dtype=float)
            forward /= np.linalg.norm(forward)
            self.assertAlmostEqual(np.dot(forward, side1), 0.0, places=8)
            self.assertAlmostEqual(np.dot(forward, side2), 0.0, places=8)
            self.assertAlmostEqual(np.dot(side1, side2), 0.0, places=8)
            self.assertAlmostEqual(np.linalg.norm(side1), 1.0, places=8)
            self.assertAlmostEqual(np.linalg.norm(side2), 1.0, places=8)

    def test_basis_does_not_flip_when_heading_crosses_reference_axis(self):
        previous = None
        values = []
        for angle in np.linspace(-0.2, 0.2, 101):
            direction = np.array([np.cos(angle), -np.sin(angle), 0.0])
            side1, _ = perpendicular_basis(direction, previous)
            if previous is not None:
                values.append(float(np.dot(previous, side1)))
            previous = side1
        self.assertGreater(min(values), 0.95)


class GuidanceTests(unittest.TestCase):
    def test_terminal_capture_speed_goes_to_zero_at_target(self):
        target = np.zeros(3)
        target_velocity = np.array([1.0, 0.0, 0.0])
        at_50, speed_50, _ = desired_target_velocity(
            [50.0, 0.0, 0.0], target, target_velocity,
            CONFIG, 0.2
        )
        at_10, speed_10, _ = desired_target_velocity(
            [10.0, 0.0, 0.0], target, target_velocity,
            CONFIG, 0.2
        )
        at_25, speed_25, _ = desired_target_velocity(
            [25.0, 0.0, 0.0], target, target_velocity,
            CONFIG, 0.2
        )
        at_0, speed_0, _ = desired_target_velocity(
            [0.0, 0.0, 0.0], target, target_velocity,
            CONFIG, 0.2
        )
        self.assertAlmostEqual(speed_50, 0.30, places=6)
        self.assertLess(speed_10, speed_50)
        self.assertAlmostEqual(speed_25, 0.0, places=8)
        self.assertAlmostEqual(speed_0, CONFIG.capture_max_speed_mps, places=8)
        np.testing.assert_allclose(at_25, target_velocity, atol=1e-8)
        self.assertNotEqual(at_0[0], target_velocity[0])

    def test_lateral_bias_fades_inside_capture_region(self):
        target = np.zeros(3)
        velocity_far, _, _ = desired_target_velocity(
            [500.0, 0.0, 0.0], target, np.zeros(3), CONFIG, 0.2,
            direction_bias=[0.0, 2.0, 0.0]
        )
        velocity_near, _, _ = desired_target_velocity(
            [60.0, 0.0, 0.0], target, np.zeros(3), CONFIG, 0.2,
            direction_bias=[0.0, 2.0, 0.0]
        )
        self.assertGreater(abs(velocity_far[1]), abs(velocity_near[1]))


class RadarClockTests(unittest.TestCase):
    def test_clock_offset_estimates_separate_process_epoch(self):
        class Cfg:
            radar_sync_window = 21
            radar_sync_min_samples = 3
            radar_controller_poll_hz = 60.0
            radar_stale_limit_s = 0.55

        sync = RadarClockSynchronizer(Cfg())
        offset_true = 3.25
        last = None
        ready = False
        for k in range(1, 15):
            raw_latest = 0.167 * k
            controller_time = offset_true + raw_latest + 0.0083
            row = np.zeros((3, 3), dtype=float)
            row[2, 0] = raw_latest
            offset, ready, age = sync.update([row], controller_time)
            last = offset
        self.assertTrue(ready)
        self.assertAlmostEqual(last, offset_true, places=3)
        self.assertLess(age, 0.03)

    def test_decode_uses_corrected_time_to_sample_shuttle_history(self):
        history = StateHistory(max_age_s=10.0)
        history.add(10.0, [100, 0, 0], [2, 0, 0])
        history.add(11.0, [102, 0, 0], [2, 0, 0])
        row = np.zeros((3, 3), dtype=float)
        row[0, :] = [10, 0, 0]
        row[1, :] = [8, -1, 4]
        row[2, 0] = 1.0  # radar-process time
        obs = decode_radar(
            [row], [102, 0, 0], [2, 0, 0], 11.0,
            state_history=history, radar_time_offset_s=9.0,
            config=CONFIG
        )[0]
        np.testing.assert_allclose(obs.position, [110, 0, 0])
        self.assertAlmostEqual(obs.timestamp, 10.0)


    def test_range_fusion_uses_independent_closest_surface_measurement(self):
        row = np.zeros((3, 3), dtype=float)
        row[0, :] = [100.0, 0.0, 0.0]
        row[1, :] = [108.0, 0.0, 20.0]
        row[2, 0] = 1.0
        observation = decode_radar(
            [row],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            current_timestamp=1.0,
            config=CONFIG,
        )[0]
        # center-range from surface = 108 + 20/2 = 118 m;
        # inverse-variance fusion of 100 m (sigma 2) and 118 m
        # (sigma sqrt(2^2 + (0.5*20*0.05)^2)).
        self.assertAlmostEqual(
            np.linalg.norm(observation.relative_position),
            108.7272727273,
            places=6,
        )

    def test_radar_diagnostics_use_controller_time_offset(self):
        row = np.zeros((3, 3), dtype=float)
        row[0, :] = [10.0, 0.0, 0.0]
        row[1, :] = [8.0, 0.0, 4.0]
        row[2, 0] = 1.0
        message = format_radar_diagnostic(
            10.25, [0, 0, 0], [0, 0, 0], [row],
            radar_time_offset_s=9.0
        )
        self.assertIn("measurement_t=10.000", message)
        self.assertIn("age_s=0.250", message)



class ActuatorModelTests(unittest.TestCase):
    def test_thruster_csv_reader_matches_seven_column_actuator_layout(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as stream:
            stream.write("x,y,z,nx,ny,nz,thrust\n")
            stream.write("0,0,0,-1,0,0,10\n")
            stream.write("0,0,0,1,0,0,10\n")
            path = stream.name
        try:
            records = read_thruster_csv(path)
            self.assertEqual(len(records), 2)
            np.testing.assert_allclose(records[0], [0, 0, 0, -1, 0, 0, 10])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_axis_authority_from_synthetic_symmetric_thrusters(self):
        # Each pair provides pure translation with no torque because the
        # synthetic thrusters act through the centre of mass.
        records = [
            [0, 0, 0, -1, 0, 0, 10],
            [0, 0, 0, 1, 0, 0, 10],
            [0, 0, 0, 0, -1, 0, 12],
            [0, 0, 0, 0, 1, 0, 12],
            [0, 0, 0, 0, 0, -1, 8],
            [0, 0, 0, 0, 0, 1, 8],
        ]
        activity = build_activity_matrix(
            records, mass=100.0, inertia_body=np.eye(3) * 100.0
        )
        positive, negative, valid, residual = compute_axis_authorities(activity)
        self.assertTrue(valid)
        np.testing.assert_allclose(positive, [0.1, 0.12, 0.08], atol=1e-7)
        np.testing.assert_allclose(negative, [0.1, 0.12, 0.08], atol=1e-7)
        self.assertLess(residual, 1e-7)

    def test_competition_no_mains_thruster_asset_initial_authority(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "SpaceBuggyJetConfigNoMains.csv"
        records = read_thruster_csv(fixture)
        self.assertEqual(len(records), 38)

        # Initial SpaceBuggy mass and body inertia from COMPETITION_SIM.simconf.
        inertia = np.array([
            [3025.8, -171.7, 461.7],
            [-171.7, 5927.5, 313.6],
            [461.7, 313.6, 6756.0],
        ])
        activity = build_activity_matrix(records, 3388.7, inertia)
        positive, negative, valid, residual = compute_axis_authorities(activity)
        self.assertTrue(valid)
        self.assertLess(residual, 1.0e-7)
        np.testing.assert_allclose(
            positive, [0.19289583, 0.28933741, 0.38579024], atol=1.0e-5
        )
        np.testing.assert_allclose(
            negative, [0.28934308, 0.28934129, 0.31526554], atol=1.0e-5
        )

    def test_batch_limiter_matches_scalar_limiter(self):
        positive = np.array([0.20, 0.12, 0.08])
        negative = np.array([0.10, 0.15, 0.09])
        commands = np.array([
            [0.4, 0.2, -0.1],
            [-0.3, 0.1, 0.2],
            [0.0, 0.0, 0.0],
        ])
        batch = limit_acceleration_body_batch(
            commands, positive, negative, 0.82
        )
        scalar = np.asarray([
            limit_acceleration_body(
                row, positive, negative, 0.82
            )[0]
            for row in commands
        ])
        np.testing.assert_allclose(batch, scalar, atol=1e-10)


class TrackerTests(unittest.TestCase):
    def test_duplicate_radar_rows_do_not_create_or_move_tracks(self):
        tracker = DebrisTracker(CONFIG)

        class Obs:
            position = np.array([100.0, 0.0, 0.0])
            diameter = 10.0
            range_rate = 1.0
            timestamp = 0.0
            relative_position = np.array([100.0, 0.0, 0.0])
            shuttle_velocity = np.zeros(3)

        first = tracker.update([Obs()], 0.0, [0, 0, 0], [0, 0, 0])
        self.assertEqual(len(first), 1)
        second = tracker.update([Obs()], 0.1, [0, 0, 0], [0, 0, 0])
        self.assertEqual(len(second), 1)
        self.assertEqual(tracker.last_update_stats["duplicates"], 1)
        self.assertEqual(tracker.last_update_stats["created"], 0)

    def test_repeated_measurement_associates_to_existing_track(self):
        tracker = DebrisTracker(CONFIG)

        class Obs:
            def __init__(self, x, timestamp):
                self.position = np.array([x, 0.0, 0.0])
                self.diameter = 10.0
                self.range_rate = 1.0
                self.timestamp = timestamp
                self.relative_position = np.array([x, 0.0, 0.0])
                self.shuttle_velocity = np.zeros(3)

        tracker.update([Obs(100.0, 0.0)], 0.0, [0, 0, 0], [0, 0, 0])
        tracks = tracker.update(
            [Obs(101.0, 1.0)], 1.0, [0, 0, 0], [0, 0, 0]
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].hits, 2)
        self.assertEqual(tracker.last_update_stats["matched"], 1)
        self.assertEqual(tracker.last_update_stats["created"], 0)

    def test_radial_range_rate_is_fused_into_absolute_velocity(self):
        tracker = DebrisTracker(CONFIG)
        class Obs:
            position = np.array([100.0, 0.0, 0.0])
            diameter = 10.0
            range_rate = 2.0
            timestamp = 0.0
            relative_position = np.array([100.0, 0.0, 0.0])
            shuttle_velocity = np.array([5.0, 0.0, 0.0])
        tracks = tracker.update([Obs()], 0.0, [0, 0, 0], [5, 0, 0])
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0].velocity[0], 7.0, places=6)


class PlannerTests(unittest.TestCase):
    def test_head_on_threat_selects_non_beeline_path(self):
        planner = PredictivePlanner(CONFIG)
        threat = DebrisTrack(
            identifier=7,
            position=np.array([150.0, 0.0, 0.0]),
            velocity=np.array([-2.0, 0.0, 0.0]),
            diameter=20.0,
            timestamp=0.0,
            hits=5,
            confirmed=True,
        )
        positive = np.array([0.20, 0.20, 0.20])
        negative = np.array([0.20, 0.20, 0.20])

        def limiter(commands):
            commands = np.asarray(commands, dtype=float)
            if commands.ndim == 1:
                return limit_acceleration_body(
                    commands, positive, negative, 0.82
                )[0]
            return limit_acceleration_body_batch(
                commands, positive, negative, 0.82
            )

        accel, metrics = planner.plan(
            [0, 0, 0],
            [5, 0, 0],
            [500, 0, 0],
            [0, 0, 0],
            [threat],
            0.0,
            max_accel=0.20,
            acceleration_limiter=limiter,
        )
        self.assertTrue(np.all(np.isfinite(accel)))
        self.assertNotEqual(metrics["selected_candidate"], "beeline")
        self.assertTrue(metrics["avoidance_active"])
        self.assertGreater(metrics["candidate_count"], 10)

    def test_no_debris_uses_direct_target_guidance(self):
        planner = PredictivePlanner(CONFIG)
        accel, metrics = planner.plan(
            [0, 0, 0], [0, 0, 0],
            [1000, 0, 0], [0, 0, 0],
            [], 0.0, max_accel=0.20
        )
        self.assertGreater(accel[0], 0.0)
        self.assertAlmostEqual(accel[1], 0.0, places=8)
        self.assertEqual(metrics["selected_candidate"], "beeline")
        self.assertFalse(metrics["emergency"])

    def test_avoidance_hysteresis_requires_repeated_clear_samples(self):
        planner = PredictivePlanner(CONFIG)
        planner.avoidance_active = True
        # This is exercised through repeated plans with no debris; the first
        # few clear samples should not immediately release the avoidance mode.
        for i in range(CONFIG.recovery_release_cycles - 1):
            planner.plan(
                [0, 0, 0], [0, 0, 0],
                [1000, 0, 0], [0, 0, 0],
                [], float(i), max_accel=0.20
            )
            self.assertTrue(planner.avoidance_active)
        planner.plan(
            [0, 0, 0], [0, 0, 0],
            [1000, 0, 0], [0, 0, 0],
            [], 10.0, max_accel=0.20
        )
        self.assertFalse(planner.avoidance_active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
