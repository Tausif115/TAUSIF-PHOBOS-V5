# Test fixture

`SpaceBuggyJetConfigNoMains.csv` is a read-only copy of the exact thruster asset
named by `JetConfigName` in the supplied `COMPETITION_SIM.simconf`.

It is included only so offline tests can verify the CSV parser and initial
axis-authority calculation. The deployment script does not copy this fixture
into `STU4-Sim`, and the runtime controller reads the simulator's configured
asset path directly.
