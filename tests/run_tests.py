"""Minimal, dependency-free test runner for the PHOBOS navigation tests.

pytest is not available in the STPro Python environment. This runner executes
both unittest.TestCase classes and plain module-level ``test_*`` functions.

Usage (from V5-DEVELOPMENT; -B prevents writing bytecode anywhere):

    python -B tests/run_tests.py [--site-packages <dir>] [-k substring]

``--site-packages`` may point at the STPro venv site-packages to borrow its
numpy/scipy build without installing anything.
"""

import argparse
import importlib.util
import inspect
import sys
import time
import traceback
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", action="append", default=[])
    parser.add_argument("-k", dest="keyword", default="")
    args = parser.parse_args()

    for path in args.site_packages:
        if path not in sys.path:
            sys.path.append(path)

    sys.dont_write_bytecode = True
    tests_dir = Path(__file__).resolve().parent
    results = []

    for module_path in sorted(tests_dir.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            results.append((f"{module_path.stem} [import]", False, traceback.format_exc(), 0.0))
            continue

        for name, obj in sorted(vars(module).items()):
            if inspect.isclass(obj) and issubclass(obj, unittest.TestCase) and obj.__module__ == module.__name__:
                for case in unittest.TestLoader().loadTestsFromTestCase(obj):
                    test_id = f"{module_path.stem}::{obj.__name__}::{case._testMethodName}"
                    if args.keyword and args.keyword not in test_id:
                        continue
                    outcome = unittest.TestResult()
                    start = time.perf_counter()
                    case.run(outcome)
                    elapsed = time.perf_counter() - start
                    problems = outcome.errors + outcome.failures
                    detail = "".join(text for _, text in problems)
                    if outcome.skipped:
                        detail = "SKIPPED: " + outcome.skipped[0][1]
                    results.append((test_id, not problems, detail, elapsed))
            elif inspect.isfunction(obj) and name.startswith("test_") and obj.__module__ == module.__name__:
                test_id = f"{module_path.stem}::{name}"
                if args.keyword and args.keyword not in test_id:
                    continue
                start = time.perf_counter()
                try:
                    obj()
                    results.append((test_id, True, "", time.perf_counter() - start))
                except Exception:
                    results.append((test_id, False, traceback.format_exc(), time.perf_counter() - start))

    failed = [item for item in results if not item[1]]
    for test_id, ok, detail, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'}  {elapsed * 1000.0:8.1f} ms  {test_id}")
    for test_id, _, detail, _ in failed:
        print(f"\n===== FAILURE: {test_id}\n{detail}")
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed, {len(results)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
