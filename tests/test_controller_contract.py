import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CompetitionContractTests(unittest.TestCase):
    def test_controller_uses_official_command_helper(self):
        source = (ROOT / "Controller.py").read_text()
        self.assertIn("def SetCommand(", source)
        self.assertIn('["JetSelection", "CommandedAccel"]', source)
        self.assertIn('["JetSelection", "CommandedAngAccel"]', source)
        self.assertIn("SetCommand(", source)

    def test_controller_reads_only_documented_radar_helper(self):
        source = (ROOT / "Controller.py").read_text()
        self.assertIn("GetNearestDebrisLocations(CONFIG.radar_max_objects)", source)
        self.assertIn("config=CONFIG", source)
        self.assertIn("radar_time_offset_s=radar_clock_offset_s", source)

    def test_stopping_profile_uses_weaker_forward_or_reverse_authority(self):
        source = (ROOT / "Controller.py").read_text()
        self.assertIn("min(forward_authority, reverse_authority)", source)
        self.assertIn("limit_accel_phobos(-goal_direction * max_linear_accel)", source)

    def test_controller_does_not_apply_physical_effects_or_write_thruster_data(self):
        source = (ROOT / "Controller.py").read_text()
        self.assertNotIn("setPhysEffect", source)
        self.assertNotIn("PhysEffect(", source)
        self.assertNotIn("AddOrSetParam", source)
        self.assertNotIn("SetParamArray", source)
        self.assertNotRegex(
            source,
            r'SetParam\([^)]*\["RCSConfig"',
        )

    def test_thruster_csv_is_read_only(self):
        source = (ROOT / "Controller.py").read_text()
        self.assertIn("AssetPathToReal(", source)
        self.assertIn("CONFIG.jet_config_asset_path", source)
        self.assertIn("read_thruster_csv(", source)
        self.assertIn("build_activity_matrix(", source)
        self.assertNotIn("RCSConfig", source)

    def test_build_contains_no_sim_config_or_actuator_replacement(self):
        self.assertFalse((ROOT / "COMPETITION_SIM.simconf").exists())
        self.assertFalse((ROOT / "JetSelectionActuator.py").exists())
        deploy = (ROOT / "deploy_v4_3.ps1").read_text()
        self.assertNotIn("$actuatorTarget", deploy)
        self.assertNotIn("Copy-Item $sourceActuator", deploy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
