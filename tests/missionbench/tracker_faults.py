"""Live-like tracker faults for the offline bench (test-only; wraps the real
DebrisTracker.update after it runs, so the controller sees corrupted tracks).

Rates calibrated on live runs 0555d4ef / bde2f2fb (500 debris):
  reacquisitions ~6.5 /s  -> track restarted unconfirmed with a poor velocity
  velocity clamps ~0.33 /s -> mis-associated track with an implausible velocity
Phantoms: at scheduled times the track nearest the shuttle (30-150 m) gets a
confirmed velocity aimed at the shuttle (the bde2f2fb track-38 pattern).
"""
import numpy as np

DEFAULTS = dict(seed=0, start_s=4.0, reacq_rate=6.5, reacq_sigma=1.0,
                bad_rate=0.35, bad_speed=(4.0, 15.0), phantoms=())


def install(tracking_module, spec):
    o = dict(DEFAULTS, **(spec or {}))
    rng = np.random.default_rng(int(o["seed"]))
    phantoms = sorted((dict(p) for p in o["phantoms"]), key=lambda p: p["t"])
    state = {"t": None, "stats": {"reacq": 0, "bad": 0, "phantom": 0}}
    cls = tracking_module.DebrisTracker
    original = cls.update

    def update(self, observations, timestamp, shuttle_position, shuttle_velocity, *a, **k):
        result = original(self, observations, timestamp, shuttle_position, shuttle_velocity, *a, **k)
        t = float(timestamp)
        last, state["t"] = state["t"], t
        if last is None or t < o["start_s"] or not self.tracks:
            return result
        dt = max(0.0, t - last)
        n = len(self.tracks)
        for i in rng.integers(0, n, rng.poisson(o["reacq_rate"] * dt)):
            tr = self.tracks[i]
            tr.velocity = np.asarray(tr.velocity, float) + rng.normal(0.0, o["reacq_sigma"], 3)
            tr.hits, tr.confirmed = 2, False      # one more hit re-confirms (live plateau ~94 %)
            state["stats"]["reacq"] += 1
        for i in rng.integers(0, n, rng.poisson(o["bad_rate"] * dt)):
            d = rng.normal(size=3)
            self.tracks[i].velocity = d / np.linalg.norm(d) * rng.uniform(*o["bad_speed"])
            state["stats"]["bad"] += 1
        sp = np.asarray(shuttle_position, float)
        sv = np.asarray(shuttle_velocity, float)
        while phantoms and t >= phantoms[0]["t"]:
            ph = phantoms.pop(0)
            pos = np.array([tr.position_at(t) for tr in self.tracks])
            dist = np.linalg.norm(pos - sp, axis=1)
            ok = np.nonzero((dist > 30.0) & (dist < 150.0))[0]
            if not len(ok):
                continue
            j = int(ok[np.argmin(dist[ok])])
            aim = sp + sv * float(ph.get("lead_s", 5.0)) - pos[j]
            tr = self.tracks[j]
            tr.velocity = aim / np.linalg.norm(aim) * float(ph.get("speed", 6.0))
            tr.confirmed = True
            state["stats"]["phantom"] += 1
        return result

    cls.update = update
    return state["stats"]
