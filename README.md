# Phobos Escape Controller V4.3 — 500-debris validation candidate

This controller folder is the V4.3 navigation package with validation-only changes to tracking capacity, radar timing tolerance, and track-age diagnostics.

Read the package-level `README_500_DEBRIS_VALIDATION.md` for installation, test sequence, acceptance criteria, and the separate 500-radar simconf.

The included `deploy_v4_3.ps1` retains the guarded V4.3 deployment behavior: it backs up the installed controller, verifies the official actuator against the configured upstream, and does not modify the official actuator, radar source, original competition simconf, scoring code, or thruster CSV.

This is not a final competition release. Do not optimize fuel until the 500-debris live validation gate passes.
