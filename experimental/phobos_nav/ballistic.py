"""V5.1 gravity-aware burn-coast-trim fly-by guidance with predictive avoidance.

Physics basis (official STU4-Sim sources):
- Score latches the first time range to the target origin <= 50 m (Scoring.py);
  nothing afterwards is scored, so the shuttle flies BY the target (aim point
  offset `v51_miss_m` from it) instead of stopping.
- The official actuator fires the busiest jet for the whole tick for ANY non-zero
  command (JetSelectionActuator.py), so thrust is bang-bang per tick: commands are
  either exactly zero (coast) or a full-authority burn toward a velocity target.
- Fuel = sum|jet impulse|/(Isp g0): every m/s costs ~4.3-4.8 kg; burns are minimised.
- Dynamics in the Phobos body-fixed frame: Phobos point-mass gravity plus (optional)
  rotating-frame and Mars-tide terms (spin about +z; Mars along +x).
"""
import math

import numpy as np


class Dynamics:
    def __init__(self, mu, rate, frame_terms):
        self.mu, self.n, self.frame = float(mu), float(rate), bool(frame_terms)

    def accel(self, x, y, z, vx, vy, vz):
        r2 = x * x + y * y + z * z
        k = -self.mu / (r2 * math.sqrt(r2))
        ax, ay, az = k * x, k * y, k * z
        if self.frame:
            n, n2 = self.n, self.n * self.n
            ax += 3.0 * n2 * x + 2.0 * n * vy      # tide (2x) + centrifugal (x) + Coriolis
            ay += -2.0 * n * vx                    # tide (-y) + centrifugal (y) cancel
            az += -n2 * z                          # tide
        return ax, ay, az

    def accel_array(self, p, v):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        a = -self.mu * p / r ** 3
        if self.frame:
            n, n2 = self.n, self.n * self.n
            extra = np.empty_like(p)
            extra[..., 0] = 3.0 * n2 * p[..., 0] + 2.0 * n * v[..., 1]
            extra[..., 1] = -2.0 * n * v[..., 0]
            extra[..., 2] = -n2 * p[..., 2]
            a = a + extra
        return a


def path_on_grid(dyn, p, v, dt, steps, v_target=None, a_thr=0.1):
    """Shuttle positions on a uniform grid (steps+1, 3); optional finite burn
    toward v_target at a_thr (velocity-target guidance, as flown)."""
    x, y, z = (float(c) for c in p)
    vx, vy, vz = (float(c) for c in v)
    out = np.empty((steps + 1, 3))
    out[0] = (x, y, z)
    tgt = None if v_target is None else [float(c) for c in v_target]
    sub = 4
    h = dt / sub
    for k in range(1, steps + 1):
        for _ in range(sub):
            ax, ay, az = dyn.accel(x, y, z, vx, vy, vz)
            if tgt is not None:
                ex, ey, ez = tgt[0] - vx, tgt[1] - vy, tgt[2] - vz
                e = math.sqrt(ex * ex + ey * ey + ez * ez)
                if e > 1e-4:
                    s = min(a_thr, e / h) / e
                    ax += ex * s; ay += ey * s; az += ez * s
                else:
                    tgt = None
            vx += ax * h; vy += ay * h; vz += az * h
            x += vx * h; y += vy * h; z += vz * h
        out[k] = (x, y, z)
    return out


def closest_approach(dyn, p, v, aim, t_max=900.0, h=1.0):
    """(distance, miss vector aim - p_ca, time) of a ballistic coast."""
    x, y, z = (float(c) for c in p)
    vx, vy, vz = (float(c) for c in v)
    ax_, ay_, az_ = (float(c) for c in aim)
    best = (float("inf"), None, 0.0)
    t = 0.0
    while t <= t_max:
        dx, dy, dz = ax_ - x, ay_ - y, az_ - z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        # sub-step refinement: closest point of the local straight segment
        sp2 = vx * vx + vy * vy + vz * vz
        tau = 0.0 if sp2 < 1e-12 else max(-h, min(h, (dx * vx + dy * vy + dz * vz) / sp2))
        if abs(tau) < h:
            mx, my, mz = dx - vx * tau, dy - vy * tau, dz - vz * tau
            dm = math.sqrt(mx * mx + my * my + mz * mz)
            if dm < best[0]:
                best = (dm, np.array([mx, my, mz]), t + tau)
        if d < best[0]:
            best = (d, np.array([dx, dy, dz]), t)
        if d > best[0] + 60.0 and tau <= 0.0:
            break
        a1 = dyn.accel(x, y, z, vx, vy, vz)
        vx += a1[0] * h; vy += a1[1] * h; vz += a1[2] * h
        x += vx * h; y += vy * h; z += vz * h
        t += h
    return best


def solve_velocity(dyn, p, v_guess, aim, fixed_speed=None, iterations=30, tol=0.3):
    v = np.array(v_guess, dtype=float)
    if fixed_speed is not None:
        v = v / max(np.linalg.norm(v), 1e-9) * fixed_speed
    d = float("inf")
    for _ in range(iterations):
        d, vec, tca = closest_approach(dyn, p, v, aim)
        if vec is None or d < tol:
            break
        v = v + 0.85 * vec / max(tca, 5.0)
        if fixed_speed is not None:
            v = v / np.linalg.norm(v) * fixed_speed
    return v, d


class _LazyReliable:
    """Defers FlybyGuidance._reliable until a decision reads the tracks."""

    def __init__(self, guide, raw, t):
        self._guide, self._raw, self._t, self._view = guide, raw, t, None

    def __getattr__(self, name):
        if self._view is None:
            self._view = self._guide._reliable(self._raw, self._t)
        return getattr(self._view, name)


class FlybyGuidance:
    """State machine WAIT -> BURN -> COAST <-> {TRIM, AVOID} -> LATCHED."""

    def __init__(self, cfg, mu):
        self.c = cfg
        self.dyn = Dynamics(mu, cfg.v51_phobos_rate, cfg.v51_frame_terms)
        self.phase = "WAIT"
        self.aim = None
        self.v_target = None
        self.next_trim = None
        self.next_check = 0.0
        self.final_trim_done = False
        self.terminal_done = set()
        self.next_terminal_check = 0.0
        self.avoid_until = -1.0
        self.emergency = False
        self.counts = {"trims": 0, "avoids": 0, "emergencies": 0, "deferred_trims": 0, "reaims": 0}
        self.dv_planned = 0.0
        self.diag = {}
        self.last = {"min_clearance": float("inf"), "threat_id": -1, "threat_cls": -1, "threat_t": float("nan"),
                     "ca_target": float("nan"), "compute_ms": 0.0, "candidates": 0, "feasible": 0}
        self.launch_t = float("nan")
        self.max_ms = 0.0
        self.aim_points = None
        self.start_loc = None
        self.next_solve = 0.0
        self.committed_until = -1.0
        self.resume_burn_target = None
        self.ready_since = None
        self.first_ready_t = None
        self.launch_state = "GRAVITY_HOLD"
        self.next_wait_check = 0.0
        self.pad_streak = 0
        self.gate_streak = 0
        self.conflict_streak = 0
        self.implausible = 0
        self.counts.update({"deferred_emergencies": 0, "aborts": 0})

    # ------------------------------------------------------- track reliability
    def _reliable(self, ts, t):
        """V5.3 track reliability (replaces the V5.2 absolute speed cap, which
        live radar truth disproved: in run 2e9a2483 debris move 1-5 m/s and the
        radar range-rate of the same object is stable over 4.7-20 s).

        The tracker fuses the radar range-rate (sigma 0.02 m/s, independent of
        position noise) into every update, so a track's radial velocity should
        match its last radar radial velocity. A track is UNRELIABLE only on
        evidence: that residual exceeds v53_rr_tolerance_mps (velocity corrupted
        or mis-associated since the update) or the tracker clamped its speed.
        Fallback (conservative, uncertainty-aware): keep the measured radial
        velocity, drop the untrusted tangential part, re-extrapolate from the
        last measured position and treat the track as tentative (tangential
        uncertainty grows with v51_tentative_growth_mps). Every track's margin
        growth is max(v51_margin_growth_mps, v53_sigma_k x filter velocity
        sigma), so young or poorly observed tracks get wider margins."""
        c = self.c
        s = self._subset(ts, slice(None))
        n = len(s.positions)
        v = np.array(ts.velocities, dtype=float).reshape(-1, 3)
        p = np.array(ts.positions, dtype=float).reshape(-1, 3)
        ages = np.asarray(ts.ages, dtype=float).reshape(-1)
        nan = np.full(n, np.nan)
        radial = getattr(ts, "radial_meas", None)
        radial = nan if radial is None else np.asarray(radial, dtype=float).reshape(-1)
        los = getattr(ts, "los", None)
        los = np.full((n, 3), np.nan) if los is None else np.asarray(los, dtype=float).reshape(-1, 3)
        sigma = getattr(ts, "vel_sigma", None)
        sigma = nan if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        clamped = getattr(ts, "clamped", None)
        clamped = np.zeros(n, dtype=bool) if clamped is None else np.asarray(clamped, dtype=bool).reshape(-1)
        known = np.isfinite(radial) & np.all(np.isfinite(los), axis=1)
        los0 = np.nan_to_num(los)
        rad0 = np.nan_to_num(radial)
        residual = np.where(known, np.einsum("ij,ij->i", v, los0) - rad0, np.nan)
        inconsistent = known & (np.abs(np.nan_to_num(residual)) > c.v53_rr_tolerance_mps)
        bad = inconsistent | clamped
        if bad.any():
            v_new = np.where(known[:, None], rad0[:, None] * los0, 0.0)
            p[bad] = p[bad] - (v[bad] - v_new[bad]) * ages[bad, None]
            v[bad] = v_new[bad]
        s.positions, s.velocities = p, v
        s.confirmed = np.asarray(ts.confirmed, dtype=bool).reshape(-1) & ~bad
        known_sigma = np.isfinite(sigma)
        sig = np.where(known_sigma, sigma, c.v53_default_vel_sigma_mps)
        s.growth = np.minimum(np.maximum(c.v51_margin_growth_mps, c.v53_sigma_k * sig), c.v53_growth_cap_mps)
        # low confidence: velocity not yet pinned by repeated radar updates
        # (fresh/reacquired tracks carry a 5 m/s tangential prior), or the
        # evidence above says the estimate is corrupted
        suspect = getattr(ts, "suspect", None)
        suspect = np.zeros(n, dtype=bool) if suspect is None else np.asarray(suspect, dtype=bool).reshape(-1)
        s.lowconf = bad | suspect | (known_sigma & (sigma > c.v53_confident_sigma_mps))
        s.sigma_known = known_sigma
        s.cls = np.where(s.lowconf, 2, np.where(s.confirmed, 0, 1))   # 0 reliable, 1 tentative, 2 low-confidence
        self.implausible = int(bad.sum())
        speeds = np.linalg.norm(v, axis=1)
        fin_res = np.abs(residual[np.isfinite(residual)])
        fin_sig = sigma[np.isfinite(sigma)]
        self.diag = {
            "inconsistent": int(inconsistent.sum()), "clamped": int(clamped.sum()),
            "suspect": int(suspect.sum()), "lowconf": int(s.lowconf.sum()),
            "tentative": int((s.cls == 1).sum()),
            "res_p90": float(np.percentile(fin_res, 90)) if fin_res.size else float("nan"),
            "sig_p50": float(np.median(fin_sig)) if fin_sig.size else float("nan"),
            "sig_p90": float(np.percentile(fin_sig, 90)) if fin_sig.size else float("nan"),
            "spd_p50": float(np.median(speeds)) if n else float("nan"),
            "spd_p90": float(np.percentile(speeds, 90)) if n else float("nan"),
        }
        return s

    # ---------------------------------------------------------------- predict
    def _debris_grid(self, ts, dt, steps):
        p = np.asarray(ts.positions, dtype=float)
        v = np.asarray(ts.velocities, dtype=float)
        grid = np.empty((steps + 1,) + p.shape)
        grid[0] = p
        for k in range(1, steps + 1):
            v = v + self.dyn.accel_array(p, v) * dt
            p = p + v * dt
            grid[k] = p
        return grid

    def _margins(self, ts, times):
        c = self.c
        ages = np.asarray(ts.ages, dtype=float)
        tentative = ~np.asarray(ts.confirmed, dtype=bool)
        base = getattr(ts, "growth", None)          # V5.3: per-track, uncertainty-aware
        base = c.v51_margin_growth_mps if base is None else np.asarray(base, dtype=float)
        # the fixed tentative allowance only stands in for an unknown covariance
        known = getattr(ts, "sigma_known", None)
        extra_t = tentative if known is None else (tentative & ~np.asarray(known, dtype=bool))
        growth = base + np.where(extra_t, c.v51_tentative_growth_mps, 0.0)
        return (c.v51_base_margin_m
                + growth[None, :] * (times[:, None] + 0.5 * ages[None, :]))

    def _clearances(self, paths, grid, ts, dt):
        """paths (K, S+1, 3) -> min clearance per path, plus (track, step) of worst."""
        radii = 0.5 * np.asarray(ts.diameters, dtype=float) + self.c.v51_shuttle_radius_m
        times = np.arange(grid.shape[0]) * dt
        margin = self._margins(ts, times)                       # (S+1, N)
        diff = grid[None, :, :, :] - paths[:, :, None, :]       # (K, S+1, N, 3)
        dist = np.sqrt(np.einsum("ksnd,ksnd->ksn", diff, diff))
        clear = dist - radii[None, None, :] - margin[None, :, :]
        tentative = ~np.asarray(ts.confirmed, dtype=bool)
        if tentative.any():   # V5.2: unreliable tracks only guard the near term
            beyond = times > self.c.v51_tentative_horizon_s
            clear = np.where((beyond[:, None] & tentative[None, :])[None], np.inf, clear)
        lowconf = getattr(ts, "lowconf", None)
        if lowconf is not None and np.any(lowconf):
            # V5.3: low-confidence tracks guard only until their next radar
            # updates (imminent conflicts are still answered immediately)
            beyond = times > self.c.v53_lowconf_horizon_s
            lc = np.asarray(lowconf, dtype=bool)
            clear = np.where((beyond[:, None] & lc[None, :])[None], np.inf, clear)
        flat = clear.reshape(clear.shape[0], -1)
        idx = np.argmin(flat, axis=1)
        return flat[np.arange(len(idx)), idx], idx // clear.shape[2], idx % clear.shape[2]

    def _relevant(self, base_path, grid, ts, dt):
        """Tracks that can matter for paths near the nominal one."""
        diff = grid - base_path[:, None, :]
        dmin = np.min(np.linalg.norm(diff, axis=2), axis=0)
        keep = dmin < self.c.v51_relevance_radius_m
        return keep

    @staticmethod
    def _subset(ts, keep):
        class _S:
            pass
        s = _S()
        for name in ("positions", "velocities", "ages", "diameters", "confirmed", "identifiers",
                     "growth", "cls", "lowconf", "sigma_known"):
            if getattr(ts, name, None) is not None:
                setattr(s, name, np.asarray(getattr(ts, name))[keep])
        return s

    def _nominal_path(self, loc, vel, dt, steps):
        if self.phase in ("BURN", "TRIM", "AVOID") and self.v_target is not None:
            return path_on_grid(self.dyn, loc, vel, dt, steps, self.v_target, self.c.v51_thrust_accel_mps2)
        return path_on_grid(self.dyn, loc, vel, dt, steps)

    def _plan_avoidance(self, loc, vel, base_path, grid, ts, dt, t_conflict=30.0, t_remaining=200.0,
                        dirs=None, weights=None):
        c = self.c
        if dirs is None:
            t_rem = max(t_remaining, t_conflict + 10.0)
            lateral_cost = max(c.v51_lateral_cost, 1.0 + t_rem / (t_rem - t_conflict))
            speed = np.linalg.norm(vel)
            u = vel / speed if speed > 0.2 else (self.aim - loc) / np.linalg.norm(self.aim - loc)
            e1 = np.cross(u, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            dirs, weights = [], []
            for a in range(8):
                ang = a * math.pi / 4.0
                dirs.append(math.cos(ang) * e1 + math.sin(ang) * e2); weights.append(lateral_cost)
            dirs.append(u); weights.append(c.v51_speedup_cost)
            dirs.append(-u); weights.append(c.v51_slowdown_cost)
        mags = np.asarray(c.v51_avoid_magnitudes, dtype=float)
        cand = np.array([d * m for d in dirs for m in mags])          # (K, 3)
        cost = np.array([w * m for w in weights for m in mags])
        times = np.arange(base_path.shape[0]) * dt
        delay = mags / (2.0 * c.v51_thrust_accel_mps2)
        delays = np.tile(delay, len(dirs))
        lever = np.maximum(0.0, times[None, :] - delays[:, None])      # (K, S+1)
        paths = base_path[None] + cand[:, None, :] * lever[:, :, None]
        clear, _, _ = self._clearances(paths, grid, ts, dt)
        feasible = clear >= 0.0
        self.last["candidates"], self.last["feasible"] = len(cand), int(feasible.sum())
        if feasible.any():
            k = int(np.argmin(np.where(feasible, cost, np.inf)))
            return cand[k], False, float(cost[k])
        k = int(np.argmax(clear))
        return cand[k], True, float("inf")

    def _plan_fold(self, loc, vel, grid, ts, dt):
        """Burn phase: candidates added to the burn target, evaluated with the
        exact finite-burn path. Returns (dv, cost) of the cheapest clearing one,
        or (None, inf)."""
        c = self.c
        speed = np.linalg.norm(self.v_target)
        u = self.v_target / speed
        e1 = np.cross(u, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        dirs = [math.cos(a * math.pi / 4.0) * e1 + math.sin(a * math.pi / 4.0) * e2 for a in range(8)]
        best = (None, float("inf"))
        steps = grid.shape[0] - 1
        for m in c.v51_avoid_magnitudes:
            cost = m * c.v51_fold_cost
            if cost >= best[1]:
                break
            paths = np.stack([path_on_grid(self.dyn, loc, vel, dt, steps, self.v_target + d * m,
                                           c.v51_thrust_accel_mps2) for d in dirs])
            clear, _, _ = self._clearances(paths, grid, ts, dt)
            if (clear >= 0.0).any():
                k = int(np.argmax(clear))
                return dirs[k] * m, cost
        return best

    # ----------------------------------------------------------------- aims
    def _aim_points(self, target_loc):
        if self.aim_points is None:
            c = self.c
            los = (target_loc - self.start_loc) / np.linalg.norm(target_loc - self.start_loc)
            e1 = np.cross(los, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
            e2 = np.cross(los, e1)
            self.aim_points = [
                target_loc + c.v51_miss_m * (math.cos(a) * e1 + math.sin(a) * e2)
                for a in (2.0 * math.pi * k / c.v51_aim_candidates for k in range(c.v51_aim_candidates))
            ]
        return self.aim_points

    def _choose_aim(self, loc, vel, target_loc, ts, fixed_speed, horizon_s, dt):
        """Evaluate every aim alternative from the current state (fixed_speed None
        = free re-aim). Returns a list of options with clearance and dv cost."""
        c = self.c
        steps = int(horizon_s / dt)
        grid = self._debris_grid(ts, dt, steps) if len(ts.positions) else None
        options = []
        for k, aim in enumerate(self._aim_points(target_loc)):
            guess = vel if np.linalg.norm(vel) > 0.5 else aim - loc
            v_req, miss = solve_velocity(self.dyn, loc, guess, aim, fixed_speed=fixed_speed)
            if miss > 5.0:
                continue
            if grid is None:
                clear = float("inf")
            else:
                path = path_on_grid(self.dyn, loc, vel, dt, steps, v_req, c.v51_thrust_accel_mps2)
                clear = float(self._clearances(path[None], grid, ts, dt)[0][0])
            options.append({"aim": aim, "v": v_req, "clear": clear,
                            "dv": float(np.linalg.norm(v_req - vel)), "k": k})
        return options

    # ----------------------------------------------------------------- update
    def update(self, t, loc, vel, target_loc, ts, radar_ok, tracks_ready, clock):
        """Returns (accel_phobos, metrics). `clock()` = wall/perf time source."""
        c = self.c
        t_start = clock()
        loc = np.asarray(loc, dtype=float); vel = np.asarray(vel, dtype=float)
        target_loc = np.asarray(target_loc, dtype=float)
        rng = float(np.linalg.norm(target_loc - loc))
        if self.phase != "LATCHED" and rng <= c.capture_radius_m:
            self.phase, self.v_target = "LATCHED", None

        n_tracks = len(ts.positions)
        confirmed = float(np.mean(ts.confirmed)) if n_tracks else 0.0
        ts = _LazyReliable(self, ts, t)

        if self.phase == "WAIT":
            self._wait_phase(t, loc, vel, target_loc, ts, radar_ok, tracks_ready, confirmed, n_tracks)

        if self.phase == "TRIM" and t >= self.next_solve and self.aim is not None:
            v_new, miss = solve_velocity(self.dyn, loc, self.v_target, self.aim)
            if miss < 5.0 and (rng < c.v51_terminal_zone_m or self._path_clear(loc, vel, v_new, ts)):
                self.v_target = v_new
            self.next_solve = t + 2.0

        if self.phase == "BURN" and t >= self.next_solve and t >= self.committed_until:
            v_new, _ = solve_velocity(self.dyn, loc, self.v_target, self.aim,
                                      fixed_speed=c.v51_launch_speed_mps)
            # never steer back into a predicted conflict after an avoidance
            if self.committed_until < 0.0 or self._path_clear(loc, vel, v_new, ts):
                self.v_target = v_new
            self.next_solve = t + 2.0

        # predictive conflict check (also after latch: zero-collision policy)
        if self.phase != "WAIT" and radar_ok and t >= self.next_check and len(ts.positions):
            self.next_check = t + c.v51_check_period_s
            dt, steps = c.v51_check_dt_s, int(c.v51_horizon_s / c.v51_check_dt_s)
            base = self._nominal_path(loc, vel, dt, steps)
            grid = self._debris_grid(ts, dt, steps)
            keep = self._relevant(base, grid, ts, dt)
            if keep.any():
                sub = self._subset(ts, keep)
                g = grid[:, keep]
                clear, step, trk = self._clearances(base[None], g, sub, dt)
                self.last["min_clearance"] = float(clear[0])
                self.last["threat_id"] = int(sub.identifiers[trk[0]])
                cls = getattr(sub, "cls", None)
                self.last["threat_cls"] = int(cls[trk[0]]) if cls is not None else -1
                self.last["threat_t"] = float(step[0] * dt)
                dv = None
                fold_done = False
                t_conflict = float(step[0] * dt)
                # V5.2: a conflict must persist over consecutive checks unless imminent
                self.conflict_streak = self.conflict_streak + 1 if clear[0] < 0.0 else 0
                persistent = (self.conflict_streak >= c.v51_persist_checks
                              or t_conflict <= c.v51_immediate_s)
                # V5.2: stop a dodge as soon as coasting from here is clear
                if (self.phase in ("AVOID", "LATCHED_AVOID") and (self.emergency or t >= self.avoid_until)):
                    coast = path_on_grid(self.dyn, loc, vel, dt, steps)
                    if self._clearances(coast[None], g, sub, dt)[0][0] >= 0.0:
                        self._abort_avoid(t, loc, vel, ts)
                        clear = np.array([0.0])
                act = clear[0] < 0.0 and t >= self.avoid_until and persistent
                far = t_conflict > c.v51_action_horizon_s
                if act:
                    emergency = False
                    burning = self.phase == "BURN"
                    # re-aim alternatives: one manoeuvre that both clears and returns to course
                    opts = [o for o in self._choose_aim(
                        loc, vel, target_loc, ts, c.v51_launch_speed_mps if burning else None,
                        c.v51_horizon_s, dt) if o["clear"] >= 0.0]
                    best = min(opts, key=lambda o: o["dv"]) if opts else None
                    if burning:
                        if best is not None:
                            self.aim, self.v_target = best["aim"], best["v"]
                            self.counts["reaims"] += 1
                            self.avoid_until = t + c.v51_avoid_commit_s
                        else:
                            fold_dv, _ = self._plan_fold(loc, vel, g, sub, dt)
                            if fold_dv is not None:      # nearly free: absorbed by the burn
                                self.v_target = self.v_target + fold_dv
                                self.committed_until = t + c.v51_avoid_commit_s
                                self.avoid_until = t + c.v51_avoid_commit_s
                                self.dv_planned += float(np.linalg.norm(fold_dv))
                                self.counts["folds"] = self.counts.get("folds", 0) + 1
                                fold_done = True
                            else:                        # interrupt the burn and dodge now
                                dv, emergency, _ = self._plan_avoidance(loc, vel, base, g, sub, dt, t_conflict, rng / max(np.linalg.norm(vel), 1.0))
                    else:
                        cand, emergency, cand_cost = self._plan_avoidance(loc, vel, base, g, sub, dt, t_conflict, rng / max(np.linalg.norm(vel), 1.0))
                        if far and cand_cost > c.v51_far_cost_cap:
                            cand, emergency, cand_cost = None, False, float("inf")   # wait for better tracks
                            if best is None or best["dv"] * c.v51_reaim_cost > c.v51_far_cost_cap:
                                best = {"dv": float("inf")}
                        if best is not None and best["dv"] * c.v51_reaim_cost <= cand_cost and np.isfinite(best["dv"]):
                            self.aim, self.v_target = best["aim"], best["v"]
                            self.phase = "AVOID" if self.phase != "LATCHED" else "LATCHED_AVOID"
                            self.counts["reaims"] += 1
                            self.dv_planned += best["dv"]
                            self.avoid_until = t + c.v51_avoid_commit_s
                            self.next_trim = t + c.v51_trim_period_s
                            emergency = False
                        elif cand is not None:
                            dv = cand
                lowconf_threat = bool(getattr(sub, "lowconf", np.zeros(len(sub.positions), bool))[trk[0]])
                if dv is not None and emergency and lowconf_threat and t_conflict > c.v53_lowconf_emergency_s:
                    # V5.3: a low-confidence track is re-measured within ~one radar
                    # revisit; no blind emergency dodge for it unless imminent
                    dv = None
                    self.counts["deferred_lowconf"] = self.counts.get("deferred_lowconf", 0) + 1
                if dv is not None and emergency and t_conflict > c.v51_emergency_horizon_s:
                    # V5.2: no clearing manoeuvre exists for a distant conflict ->
                    # wait for better tracks instead of an expensive blind dodge
                    dv = None
                    self.counts["deferred_emergencies"] += 1
                if dv is not None:
                    if self.phase == "BURN" and self.v_target is not None:
                        # interrupt the launch burn: deliver the dodge NOW (as predicted),
                        # then resume the burn toward the dodged velocity target
                        self.resume_burn_target = self.v_target + dv
                        self.v_target = vel + dv
                        self.phase = "AVOID"
                        self.committed_until = t + c.v51_avoid_commit_s
                    else:
                        self.v_target = vel + dv
                        self.phase = "AVOID" if self.phase != "LATCHED" else "LATCHED_AVOID"
                    self.dv_planned += float(np.linalg.norm(dv))
                    self.counts["avoids"] += 1
                    self.counts["emergencies"] += int(emergency)
                    self.emergency = emergency
                    self.avoid_until = t + c.v51_avoid_commit_s
                    if self.phase not in ("LATCHED", "LATCHED_AVOID"):
                        # try to return to course soon (partial returns if blocked)
                        self.next_trim = t + c.v51_avoid_commit_s
            else:
                self.last["min_clearance"] = float("inf")

        # scheduled / terminal trims toward the aim point
        if self.phase == "COAST" and rng > c.v51_trim_min_range_m:
            due = self.next_trim is not None and t >= self.next_trim
            pending = [r for r in c.v51_terminal_ranges_m if r not in self.terminal_done and rng < r]
            terminal = bool(pending) or (rng < c.v51_terminal_zone_m and t >= self.next_terminal_check)
            if rng < c.v51_terminal_zone_m and t >= self.next_terminal_check:
                self.next_terminal_check = t + 5.0
            if due or terminal:
                ca, _, _ = closest_approach(self.dyn, loc, vel, target_loc)
                self.last["ca_target"] = ca
                v_new, _ = solve_velocity(self.dyn, loc, vel, self.aim)
                dv = v_new - vel
                need = np.linalg.norm(dv) > c.v51_trim_deadband_mps or ca > c.v51_ca_limit_m
                if terminal:
                    self.terminal_done.update(pending)
                partial = False
                in_terminal = rng < c.v51_terminal_zone_m
                if in_terminal:
                    need = ca > c.v51_ca_limit_m or ca < c.v51_ca_min_m or (
                        np.linalg.norm(dv) > c.v51_trim_deadband_mps and due)
                    # terminal accuracy is mandatory; only an imminent conflict blocks it
                    clear_ok = need and self._path_clear(loc, vel, v_new, ts, horizon_s=15.0)
                else:
                    clear_ok = need and self._path_clear(loc, vel, v_new, ts)
                if need and not clear_ok and not in_terminal:
                    for frac in (0.75, 0.5, 0.25):          # partial return toward the aim
                        partial = True
                        v_part = vel + frac * dv
                        if self._path_clear(loc, vel, v_part, ts):
                            v_new, clear_ok = v_part, True
                            dv = v_new - vel
                            self.counts["partial_trims"] = self.counts.get("partial_trims", 0) + 1
                            break
                if need and not clear_ok:
                    opts = [o for o in self._choose_aim(loc, vel, target_loc, ts, None,
                                                        c.v51_horizon_s, c.v51_check_dt_s) if o["clear"] >= 0.0]
                    if opts:
                        best = min(opts, key=lambda o: o["dv"])
                        self.aim, v_new, clear_ok = best["aim"], best["v"], True
                        dv = v_new - vel
                if clear_ok:
                    self.v_target, self.phase = v_new, "TRIM"
                    self.next_solve = t + 2.0
                    self.dv_planned += float(np.linalg.norm(dv))
                    self.counts["trims"] += 1
                    self.next_trim = t + (10.0 if partial else c.v51_trim_period_s)
                elif need:
                    self.counts["deferred_trims"] += 1
                    self.next_trim = t + 5.0
                else:
                    self.next_trim = t + c.v51_trim_period_s

        # execute
        accel = np.zeros(3)
        if self.phase == "WAIT" and c.v51_wait_hold:
            # V5.2: hold against gravity so the shuttle never settles onto the pad
            # (live run bde2f2fb: pad contact at ~16 s, 45.6 J). Commands below
            # ~dt*authority are delivered proportionally by JetSelectionActuator
            # (on-time = LP solution < one tick): per-tick dv equals the command.
            up = (target_loc - loc) / max(rng, 1e-9)
            cmd = (c.v51_wait_hold_speed_mps * up - vel) * c.v51_wait_gain
            n = float(np.linalg.norm(cmd))
            if n > c.v51_wait_max_cmd:
                cmd *= c.v51_wait_max_cmd / n
            accel = cmd
        if self.phase in ("BURN", "TRIM", "AVOID", "LATCHED_AVOID") and self.v_target is not None:
            err = self.v_target - vel
            if np.linalg.norm(err) < c.v51_done_tol_mps:
                if self.phase == "BURN":
                    self.next_trim = t + c.v51_first_trim_s
                if self.phase == "AVOID" and self.resume_burn_target is not None:
                    self.phase, self.v_target = "BURN", self.resume_burn_target
                    self.resume_burn_target = None
                    err = self.v_target - vel
                else:
                    self.phase = "LATCHED" if self.phase == "LATCHED_AVOID" else "COAST"
                    self.v_target = None
            if self.v_target is not None and np.linalg.norm(err) >= c.v51_done_tol_mps:
                accel = err / np.linalg.norm(err) * c.v51_command_accel_mps2
        self.last["compute_ms"] = 1000.0 * (clock() - t_start)
        self.max_ms = max(self.max_ms, self.last["compute_ms"])
        return accel, self.metrics(rng)

    # ------------------------------------------------------------ WAIT phase
    def _hold_path(self, loc, target_loc, dt, steps):
        """Predicted shuttle path while gravity-holding (hold velocity, gravity cancelled)."""
        up = (target_loc - loc) / max(np.linalg.norm(target_loc - loc), 1e-9)
        times = np.arange(steps + 1) * dt
        return loc[None, :] + (self.c.v51_wait_hold_speed_mps * up)[None, :] * times[:, None]

    def _climb_escape(self, loc, vel, target_loc, ts, dt, window, pad_tc):
        """Best climbing escape from the hold point: (dv, emergency, clearance)
        over the imminent window, or None. Candidates only climb (>= the
        configured elevation above the local horizon); a result that is not
        climbing is rejected outright (never toward the surface)."""
        c = self.c
        steps_w = int(window)
        base = self._hold_path(loc, target_loc, dt, steps_w)
        grid_w = self._debris_grid(ts, dt, steps_w)
        up = self._local_up(loc)
        dirs = self._climb_dirs(up)
        saved_aim = self.aim
        if self.aim is None:
            self.aim = target_loc
        dv, emergency, _ = self._plan_avoidance(loc, vel, base, grid_w, ts, dt, pad_tc, 300.0,
                                                dirs=dirs, weights=[1.0] * len(dirs))
        self.aim = saved_aim
        mag = float(np.linalg.norm(dv))
        if mag <= 0.0 or float(np.dot(dv, up)) < math.sin(math.radians(c.v53_escape_min_elevation_deg)) * mag - 1e-9:
            return None
        times = np.arange(steps_w + 1) * dt
        lever = np.maximum(0.0, times - mag / (2.0 * c.v51_thrust_accel_mps2))
        path = base + dv[None, :] * lever[:, None]
        clear = float(self._clearances(path[None], grid_w, ts, dt)[0][0])
        return dv, emergency, clear

    @staticmethod
    def _local_up(loc):
        """Local outward normal: radial from Phobos' centre (the Phobos-frame
        origin; guidance gravity is -mu r/|r|^3), NOT world +z."""
        return np.asarray(loc, dtype=float) / max(float(np.linalg.norm(loc)), 1e-9)

    def _climb_dirs(self, up):
        """WAIT-escape candidates: straight out plus 8 azimuths at each of
        v53_escape_elevations_deg above the local horizon (all climbing)."""
        h1 = np.cross(up, [0.0, 0.0, 1.0])
        if np.linalg.norm(h1) < 1e-6:
            h1 = np.cross(up, [1.0, 0.0, 0.0])
        h1 /= np.linalg.norm(h1)
        h2 = np.cross(up, h1)
        dirs = [up.copy()]
        for elev in self.c.v53_escape_elevations_deg:
            ce, se = math.cos(math.radians(elev)), math.sin(math.radians(elev))
            for a in range(8):
                az = a * math.pi / 4.0
                dirs.append(se * up + ce * (math.cos(az) * h1 + math.sin(az) * h2))
        return dirs

    def _launch(self, t, loc, opt, state):
        self.start_loc = self.start_loc if self.start_loc is not None else loc.copy()
        self.aim, self.v_target = opt["aim"], opt["v"]
        self.phase, self.launch_t = "BURN", t
        self.launch_state = state
        self.next_solve = t + 2.0
        self.last["min_clearance"] = opt["clear"]

    def _wait_phase(self, t, loc, vel, target_loc, ts, radar_ok, tracks_ready, confirmed, n_tracks):
        """V5.3 launch states: GRAVITY_HOLD (no radar) -> RADAR_ACQUISITION ->
        LAUNCH_READY -> LAUNCH_COMMIT, with SAFE_HOLD (launch corridor blocked
        by a persistent conflict while the pad is safe) and EMERGENCY_LIFTOFF
        (the hold point itself is threatened and a launch option clears more).
        Gravity hold is never disabled while waiting. Retained tracks (within
        expiry) keep guarding the pad during radar outages; a normal launch
        needs live radar."""
        c = self.c
        if self.start_loc is None:
            self.start_loc = loc.copy()
        ready_now = (radar_ok and tracks_ready and t >= c.v51_min_launch_time_s
                     and confirmed >= c.v51_launch_confirmed_fraction)
        if not ready_now:
            self.ready_since = None
        elif self.ready_since is None:
            self.ready_since = t
        ready = ready_now and t - self.ready_since >= c.v51_launch_stable_s
        fallback = radar_ok and t >= c.v51_max_launch_wait_s
        if self.launch_state not in ("SAFE_HOLD",):
            self.launch_state = ("GRAVITY_HOLD" if not radar_ok else
                                 "LAUNCH_READY" if ready else "RADAR_ACQUISITION")

        # pad monitoring (1 Hz)
        if n_tracks and t >= self.next_wait_check:
            self.next_wait_check = t + c.v53_wait_check_period_s
            dt, steps = 1.0, int(c.v53_wait_horizon_s)
            grid = self._debris_grid(ts, dt, steps)
            clear, step, trk = self._clearances(self._hold_path(loc, target_loc, dt, steps)[None], grid, ts, dt)
            pad_clear, pad_tc = float(clear[0]), float(step[0] * dt)
            lowconf = bool(np.asarray(ts.lowconf)[trk[0]])
            self.last["pad_clear"], self.last["pad_tc"] = pad_clear, pad_tc
            self.last["pad_cls"] = int(np.asarray(ts.cls)[trk[0]])
            self.last["pad_id"] = int(np.asarray(ts.identifiers)[trk[0]])
            # act only on conflicts near enough to matter (full thrust moves
            # ~0.05 t^2 m: 45 m in 30 s); later ones are re-checked at 1 Hz
            threat = (pad_clear < 0.0 and pad_tc <= c.v53_pad_threat_s
                      and (not lowconf or pad_tc <= c.v53_lowconf_emergency_s))
            self.pad_streak = self.pad_streak + 1 if threat else 0
            if threat and (self.pad_streak >= c.v51_persist_checks or pad_tc <= c.v51_immediate_s):
                # emergency lift-off: staying is a predicted collision. Launch
                # options (finite-burn, real authority) are judged over the
                # imminent window only; later conflicts are handled in flight,
                # where the full dodge candidate set (incl. lateral escapes) is
                # available. Stay only if every option is strictly worse.
                window = min(c.v53_wait_horizon_s, pad_tc + c.v53_liftoff_window_margin_s)
                opts = self._choose_aim(loc, vel, target_loc, ts, c.v51_launch_speed_mps, window, dt)
                pad_window = float(self._clearances(
                    self._hold_path(loc, target_loc, dt, int(window))[None],
                    self._debris_grid(ts, dt, int(window)), ts, dt)[0][0])
                best = max(opts, key=lambda o: o["clear"]) if opts else None
                if best is not None and best["clear"] >= max(pad_window, 0.0):
                    self.counts["liftoffs"] = self.counts.get("liftoffs", 0) + 1
                    self._launch(t, loc, best, "EMERGENCY_LIFTOFF")
                    return
                esc = self._climb_escape(loc, vel, target_loc, ts, dt, window, pad_tc) if best is not None else None
                # take the option with the best predicted clearance in the imminent
                # window: hold, best launch direction or best climbing escape;
                # never move into a worse prediction than staying (tie -> stay)
                choice = "hold"
                top = pad_window
                if best is not None and best["clear"] > top:
                    choice, top = "launch", best["clear"]
                if esc is not None and esc[2] > top:
                    choice, top = "escape", esc[2]
                if choice == "launch":
                    self.counts["liftoffs"] = self.counts.get("liftoffs", 0) + 1
                    self._launch(t, loc, best, "EMERGENCY_LIFTOFF")
                    return
                if choice == "escape":
                    # climbing dodge from the hold point (finite-burn delay model,
                    # never below the local horizon), then resume the launch burn
                    # toward the best aim
                    dv, emergency, _ = esc
                    self.aim = best["aim"]
                    self.counts["liftoffs"] = self.counts.get("liftoffs", 0) + 1
                    self.counts["avoids"] += 1
                    self.counts["emergencies"] += int(emergency)
                    self.dv_planned += float(np.linalg.norm(dv))
                    self.v_target, self.resume_burn_target = vel + dv, best["v"]
                    self.phase, self.launch_t, self.launch_state = "AVOID", t, "EMERGENCY_LIFTOFF"
                    self.emergency = emergency
                    self.committed_until = self.avoid_until = t + c.v51_avoid_commit_s
                    self.next_solve = t + 2.0
                    self.next_trim = t + c.v51_avoid_commit_s
                    return
                self.counts["liftoff_futile"] = self.counts.get("liftoff_futile", 0) + 1

        if not (ready or fallback):
            return
        # launch gate: the corridor must be clear over the first v53_launch_gate_s
        opts = self._choose_aim(loc, vel, target_loc, ts, c.v51_launch_speed_mps,
                                c.v51_launch_horizon_s, 2.0)
        if not opts:
            return
        gate_steps = int(c.v53_launch_gate_s)
        for o in opts:
            if n_tracks:
                path = path_on_grid(self.dyn, loc, vel, 1.0, gate_steps, o["v"], c.v51_thrust_accel_mps2)
                grid = self._debris_grid(ts, 1.0, gate_steps)
                o["near"] = float(self._clearances(path[None], grid, ts, 1.0)[0][0])
            else:
                o["near"] = float("inf")
        clear_near = [o for o in opts if o["near"] >= 0.0]
        if not clear_near and t - self.ready_t0(t) < c.v53_safe_hold_max_s:
            self.gate_streak += 1
            if self.gate_streak >= c.v51_persist_checks:
                if self.launch_state != "SAFE_HOLD":
                    self.counts["safe_holds"] = self.counts.get("safe_holds", 0) + 1
                self.launch_state = "SAFE_HOLD"
            return
        self.gate_streak = 0
        pool = clear_near or opts
        best = max(pool, key=lambda o: (min(o["clear"], c.v51_aim_clearance_cap_m), o["near"], -o["k"]))
        self._launch(t, loc, best, "LAUNCH_COMMIT")

    def ready_t0(self, t):
        if self.first_ready_t is None:
            self.first_ready_t = t
        return self.first_ready_t

    def _abort_avoid(self, t, loc, vel, ts):
        c = self.c
        self.counts["aborts"] += 1
        self.emergency = False
        if self.resume_burn_target is not None:
            v_new, miss = solve_velocity(self.dyn, loc, self.resume_burn_target, self.aim,
                                         fixed_speed=c.v51_launch_speed_mps)
            ok = miss < 5.0 and self._path_clear(loc, vel, v_new, ts)
            self.phase, self.v_target = "BURN", (v_new if ok else self.resume_burn_target)
            self.resume_burn_target = None
            self.next_solve = t + 2.0
        else:
            self.phase = "LATCHED" if self.phase == "LATCHED_AVOID" else "COAST"
            self.v_target = None

    def _path_clear(self, loc, vel, v_new, ts, horizon_s=None):
        if not len(ts.positions):
            return True
        c = self.c
        horizon = c.v51_horizon_s if horizon_s is None else horizon_s
        dt, steps = c.v51_check_dt_s, max(2, int(horizon / c.v51_check_dt_s))
        path = path_on_grid(self.dyn, loc, vel, dt, steps, v_new, c.v51_thrust_accel_mps2)
        grid = self._debris_grid(ts, dt, steps)
        clear, _, _ = self._clearances(path[None], grid, ts, dt)
        return bool(clear[0] >= 0.0)

    def metrics(self, rng):
        state = {"WAIT": "LAUNCH_WAIT", "BURN": "CLEAR", "COAST": "CLEAR", "TRIM": "CLEAR",
                 "AVOID": "AVOIDING", "LATCHED": "CAPTURED", "LATCHED_AVOID": "AVOIDING"}[self.phase]
        if self.emergency and self.phase in ("AVOID", "LATCHED_AVOID"):
            state = "EMERGENCY_AVOID"
        if self.phase == "WAIT" and self.launch_state == "SAFE_HOLD":
            state = "SAFE_HOLD"
        return {
            "risk": 0.0, "distance": rng, "clearance": self.last["min_clearance"],
            "emergency": bool(self.emergency and self.phase in ("AVOID", "LATCHED_AVOID")),
            "safety_state": state, "candidate_count": self.last["candidates"],
            "safe_candidate_count": self.last["feasible"], "preferred_candidate_count": 0,
            "closest_track_id": self.last["threat_id"], "closest_time_s": self.last["threat_t"],
            "planning_track_count": 0, "planner_compute_ms": self.last["compute_ms"],
            "avoidance_active": self.phase in ("AVOID", "LATCHED_AVOID"),
        }

    def log_fields(self):
        d, nan = self.diag, float("nan")
        return (f"v51_phase={self.phase} v51_dv_planned={self.dv_planned:.3f} "
                f"v51_trims={self.counts['trims']} v51_avoids={self.counts['avoids']} v51_reaims={self.counts.get('reaims', 0)} "
                f"v51_emergencies={self.counts['emergencies']} v51_deferred={self.counts['deferred_trims']} "
                f"v51_min_clear_m={self.last['min_clearance']:.1f} v51_ca_target_m={self.last['ca_target']:.1f} "
                f"v51_launch_t={self.launch_t:.1f} v53_unreliable={self.implausible} "
                f"v53_inconsistent={d.get('inconsistent', 0)} v53_clamped={d.get('clamped', 0)} "
                f"v53_tentative={d.get('tentative', 0)} v53_res_p90={d.get('res_p90', nan):.2f} "
                f"v53_vsig_p50={d.get('sig_p50', nan):.2f} v53_vsig_p90={d.get('sig_p90', nan):.2f} "
                f"v53_spd_p50={d.get('spd_p50', nan):.2f} v53_spd_p90={d.get('spd_p90', nan):.2f} "
                f"v53_thr_cls={self.last.get('threat_cls', -1)} v53_suspect={d.get('suspect', 0)} "
                f"v53_lowconf={d.get('lowconf', 0)} v53_launch_state={self.launch_state} "
                f"v53_pad_clear_m={self.last.get('pad_clear', float('inf')):.1f} "
                f"v53_pad_tc_s={self.last.get('pad_tc', float('nan')):.0f} v53_pad_cls={self.last.get('pad_cls', -1)} "
                f"v53_pad_id={self.last.get('pad_id', -1)} "
                f"v53_liftoffs={self.counts.get('liftoffs', 0)} v53_safe_holds={self.counts.get('safe_holds', 0)} "
                f"v53_deferred_lowconf={self.counts.get('deferred_lowconf', 0)} "
                f"v52_deferred_emerg={self.counts['deferred_emergencies']} v52_aborts={self.counts['aborts']} "
                f"v51_ms={self.last['compute_ms']:.1f} v51_ms_max={self.max_ms:.1f}")
