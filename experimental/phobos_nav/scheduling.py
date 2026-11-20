"""Deadline-aware execution of the two planner stages (V5 Phase 2b).

The planning cycle is split into PredictivePlanner.plan_stage_one
(shortlist, candidates, trajectory propagation, clearance bounds) and
plan_stage_two (exact clearance evaluation, selection, command). Together
they compute exactly what plan() computes for the stage-one inputs.

DeadlinePlanner decides, inside each control tick, whether a stage fits the
remaining tick budget, using the recent worst observed duration of that
stage. Guarantees:
  * a requested plan is never dropped (requests latch until started);
  * a stage waits at most `planner_max_stage_wait_ticks` ticks, then runs
    regardless of budget (bounded latency);
  * critical conditions do NOT bypass the budget: splitting a plan across
    two ticks delays it by one tick (16.7 ms), far below the critical TTC
    threshold (6 s) and the planner's modelled control latency (0.1 s),
    while the previous command remains active; overrunning the tick would
    delay every subsequent command instead;
  * a job older than `planner_max_job_age_s` (the planner's own modelled
    control latency) is discarded and re-planned instead of being applied
    stale.
Pure Python, no simulator dependency: the controller and the offline
benchmark use the same class.
"""

from collections import deque
import time

import numpy as np


class StageBudget:
    """Per-stage duration estimates from recent measurements.

    The 90th percentile of recent durations (once enough samples exist):
    robust to isolated operating-system/GC stalls, which would otherwise
    block a stage for the whole history window (a max-based estimate).
    """

    MIN_SAMPLES = 5
    PERCENTILE = 90.0

    def __init__(self, initial_estimates_s, history=20):
        self.initial = {1: float(initial_estimates_s[0]), 2: float(initial_estimates_s[1])}
        self.samples = {1: deque(maxlen=int(history)), 2: deque(maxlen=int(history))}

    def estimate(self, stage):
        samples = self.samples[stage]
        if len(samples) < self.MIN_SAMPLES:
            return max([self.initial[stage]] + list(samples))
        return float(np.percentile(np.asarray(samples), self.PERCENTILE))

    def record(self, stage, duration_s):
        if np.isfinite(duration_s) and duration_s >= 0.0:
            self.samples[stage].append(float(duration_s))


class DeadlinePlanner:
    def __init__(self, planner, config, clock=time.perf_counter):
        self.planner = planner
        self.config = config
        self.clock = clock
        self.tick_budget_s = float(getattr(config, "control_tick_budget_s", 0.014))
        self.max_wait_ticks = int(getattr(config, "planner_max_stage_wait_ticks", 2))
        self.max_job_age_s = float(getattr(config, "planner_max_job_age_s", 0.1))
        self.budget = StageBudget(
            getattr(config, "planner_initial_stage_estimates_s", (0.006, 0.006)),
            history=int(getattr(config, "planner_stage_history", 20)),
        )
        self.requested = False
        self.request_wait = 0
        self.job = None
        self.job_wait = 0
        self.stats = {
            "completed": 0, "deferred": 0, "forced": 0, "discarded": 0,
            "stage1_ms": float("nan"), "stage2_ms": float("nan"), "plan_ms": float("nan"),
        }

    def reset(self):
        """Drop any pending request or half-finished job (e.g. after an error)."""
        self.requested = False
        self.request_wait = 0
        self.job = None
        self.job_wait = 0

    def _fits(self, stage, tick_start):
        return (self.clock() - tick_start) + self.budget.estimate(stage) <= self.tick_budget_s

    def _run_stage_two(self):
        start = self.clock()
        acceleration, metrics = self.planner.plan_stage_two(self.job)
        duration = self.clock() - start
        self.budget.record(2, duration)
        self.stats["stage2_ms"] = 1000.0 * duration
        self.stats["plan_ms"] = self.stats["stage1_ms"] + self.stats["stage2_ms"]
        self.stats["completed"] += 1
        metrics["planner_compute_ms"] = self.stats["plan_ms"]
        metrics["planner_stage1_ms"] = self.stats["stage1_ms"]
        metrics["planner_stage2_ms"] = self.stats["stage2_ms"]
        metrics["planner_job_age_ticks"] = self.job_wait
        self.job = None
        self.job_wait = 0
        return acceleration, metrics

    def tick(self, tick_start, due, critical, now_s, make_stage_one_inputs):
        """Advance planning within this tick.

        Returns (acceleration, metrics) when a plan completes in this tick,
        otherwise None. `make_stage_one_inputs()` is only called when stage
        one actually starts. `critical` is accepted for telemetry and future
        policy; it does not bypass the budget (see module docstring). Planner exceptions propagate to the caller,
        which should call reset() and apply its fail-safe command.
        """
        if due:
            self.requested = True

        if self.job is not None and now_s - float(self.job["timestamp"]) > self.max_job_age_s:
            self.job = None
            self.job_wait = 0
            self.requested = True
            self.stats["discarded"] += 1

        if self.job is not None:
            forced = self.job_wait >= self.max_wait_ticks
            if forced or self._fits(2, tick_start):
                if forced and not self._fits(2, tick_start):
                    self.stats["forced"] += 1
                return self._run_stage_two()
            self.job_wait += 1
            self.stats["deferred"] += 1
            return None

        if not self.requested:
            return None
        forced = self.request_wait >= self.max_wait_ticks
        if not (forced or self._fits(1, tick_start)):
            self.request_wait += 1
            self.stats["deferred"] += 1
            return None
        if forced and not self._fits(1, tick_start):
            self.stats["forced"] += 1

        start = self.clock()
        self.job = self.planner.plan_stage_one(**make_stage_one_inputs())
        duration = self.clock() - start
        self.budget.record(1, duration)
        self.stats["stage1_ms"] = 1000.0 * duration
        self.requested = False
        self.request_wait = 0
        self.job_wait = 0

        if self._fits(2, tick_start):
            return self._run_stage_two()
        return None
