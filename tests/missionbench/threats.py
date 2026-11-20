"""Deliberately dangerous scenarios: repurpose the largest debris objects as
threats that sit on the nominal fly-by path when the shuttle passes.

Each threat: {"fraction": f (0..1 along start->aim line), "t": crossing time s,
"cross_speed": m/s across the path}. The threat state at time t is placed on the
path point and propagated BACKWARD under the bench dynamics to t=0, so it
arrives there under the same physics. Labelled stress test, not a seed."""
import math

import numpy as np


def inject(bench, specs, miss=40.0):
    w = bench.world
    shuttle = bench.entities["SpaceBuggy"]
    p0 = w.p[0].copy()
    tgt = np.asarray(bench.entities["AsteroidCycler"].params["Location"], dtype=float)
    los = (tgt - p0) / np.linalg.norm(tgt - p0)
    e1 = np.cross(los, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
    aim = tgt + miss * e1
    debris = sorted(range(1, len(w.mass)), key=lambda i: -w.radius[i])
    out = []
    for k, spec in enumerate(specs):
        i = debris[k]
        q = p0 + float(spec["fraction"]) * (aim - p0)
        across = np.cross(los, e1)
        v = float(spec.get("cross_speed", 0.5)) * across
        p, vel, t, h = q.copy(), v.copy(), float(spec["t"]), 0.05
        while t > 0:                               # backward semi-implicit Euler
            step = min(h, t)
            a = _accel(bench, p, vel)
            p = p - vel * step
            vel = vel - a * step
            t -= step
        w.p[i], w.v[i] = p, vel
        out.append((i, float(w.radius[i]), q.tolist()))
    bench.injected_threats = out
    return out


def _accel(bench, p, v):
    from world import PHOBOS_RATE
    a = bench.world.gravity_field(p)
    if bench.o["frame_terms"]:
        n = PHOBOS_RATE
        a = a + np.array([n * n * p[0] + 2 * n * v[1], n * n * p[1] - 2 * n * v[0], 0.0])
    return a
