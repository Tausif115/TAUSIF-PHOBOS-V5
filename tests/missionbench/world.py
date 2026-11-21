"""PHOBOS V5 Step 0 offline mission bench: world, physics and lock-step scheduler.

The official STU4-Sim system scripts and Controller.py are executed verbatim
(compiled from their files, unmodified) in threads; the mock `spaceteams`
module routes their API calls here. Exactly one system runs at a time; simulated
time advances only through the systems' time.sleep() calls.

Source-backed behaviour (official files run verbatim): radar sampling, noise,
publication and sorting (Radar.py); jet selection LP (SciPy revised simplex, as
STPro's SciPy 1.18.1), per-tick saturation, impulse and fuel (JetSelectionActuator.py);
success latch and score (Scoring.py); debris generation, random start impulses and
round-robin gravity refresh (SpawnDebris.py).

Modelled here (no source available): rigid-body propagation (Unreal Chaos),
ApplyGravity (1 Hz, entities whose Systems list it), the gravity field
(Gravity_Spherical: Phobos point mass, optional Mars tide), rotating-frame terms,
collision detection (spheres; no damage model), system loop overheads.
"""
import builtins
import datetime as dt
import heapq
import json
import math
import os
import pathlib
import sys
import threading
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mock"))
import spaceteams as st  # noqa: E402  (the mock)

STPRO = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "SimDynamX" / "STPro"
ASSETS = STPRO / "Assets"
PHOBOS_MU = 0.0007075e9            # simconf Dynamics.GravitationalParameter_km3_s2
PHOBOS_RADIUS_M = 11100.0          # inferred; reproduces baked seed-100 debris exactly
PHOBOS_RATE = 2.0 * math.pi / (0.31891023 * 86400.0)   # synchronous rotation, rad/s


class BenchStop(BaseException):
    pass


class Ref:
    def __init__(self, name):
        self.name = name


def _native(value):
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], str):
        tag = value[0]
        if tag in ("doubleV2", "doubleV3", "doubleV4"):
            return np.asarray(value[1:], dtype=float)
        if tag == "doubleM3x3":
            return np.asarray(value[1:], dtype=float).reshape(3, 3)
        if tag in ("int32", "int64", "uint32", "int"):
            return int(value[1])
        if tag == "EntityRef":
            return Ref(value[1])
        if tag == "timestamp":
            return st.timestamp(dt.datetime.fromisoformat(value[1].replace("Z", "+00:00")))
        if tag == "array":
            elem, items = value[1], value[2:]
            if elem == "doubleM3x3":
                return [np.asarray(x, dtype=float).reshape(3, 3) for x in items]
            if elem in ("doubleV3", "doubleV4"):
                return [np.asarray(x, dtype=float) for x in items]
            return list(items)
    return value


class BenchEntity(st.Entity):
    def __init__(self, bench, name, params, body_index=None):
        self.bench, self.name, self.params, self.body_index = bench, name, params, body_index
        q = params.get("Rotation")
        self.R = _quat_wlast_to_dcm(q) if q is not None else np.eye(3)
        self.frame = st.Frame(origin_entity=self, rotation=self.R)

    # parameters --------------------------------------------------------
    def GetParam(self, var_type, key):
        value = st._get_path(self.params, key)
        return self.bench.resolve(value)

    GetParamArray = GetParam

    def SetParam(self, var_type, key, value):
        outages = self.bench.o.get("radar_outages") or ()
        if outages and list(key) == ["Radar", "SortedDebrisData"]:
            now = self.bench.world.t
            if any(a <= now < b for a, b in outages):
                return                      # radar publication lost (outage injection)
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, list):
            value = [v.copy() if isinstance(v, np.ndarray) else v for v in value]
        st._set_path(self.params, key, value)

    SetParamArray = AddParamArray = AddOrSetParam = SetParam

    def HasParam(self, key):
        try:
            st._get_path(self.params, key)
            return True
        except KeyError:
            return False

    HasParamArray = HasParam

    # state / frames ----------------------------------------------------
    def _state(self):
        return self.bench.world.entity_state(self)

    def GetBodyFixedFrame(self):
        return self.bench.phobos_frame if self.name == "Phobos" else self.frame

    def getResidentFrame(self):
        return self.bench.phobos_frame

    def getLocVelAcc(self):
        return st._FramedState(*self._state())

    def getLocation(self):
        return st._Location(self._state()[0])

    def getRotation(self):
        return st._Rotation(self.R)

    def getMass(self):
        return float(self.params["Dynamics"]["Mass"])

    def getInertia(self, frame):
        return np.asarray(self.params["Dynamics"]["Inertia_BodyFrame"], dtype=float)

    getInertiaTensor = getInertia            # body frame (attitude constant)

    def setRotation_DCM(self, dcm):          # cameras only; no physical effect
        self.params["_camera_dcm"] = np.asarray(dcm, dtype=float)

    def setPhysEffect(self, name, effect):
        self.bench.world.apply_effect(self, name, effect)


def _quat_wlast_to_dcm(q):
    x, y, z, w = (float(c) for c in np.asarray(q, dtype=float) / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ==========================================================================
class World:
    """Point-mass dynamics in the Phobos body-fixed frame (index 0 = shuttle)."""

    def __init__(self, bench, options):
        self.bench, self.o = bench, options
        self.t = 0.0
        self.p = np.zeros((0, 3)); self.v = np.zeros((0, 3)); self.mass = np.zeros(0)
        self.radius = np.zeros(0)
        self.effects = []                    # per body: {name: world accel}
        self.acc = np.zeros((0, 3))
        self.static_targets = {}
        self.delta_v_shuttle = 0.0
        self.impulse_log = []

    def add_body(self, p, v, mass, radius):
        self.p = np.vstack([self.p, p]); self.v = np.vstack([self.v, v])
        self.mass = np.append(self.mass, mass); self.radius = np.append(self.radius, radius)
        self.effects.append({}); self.acc = np.vstack([self.acc, np.zeros(3)])
        return len(self.mass) - 1

    def finalize(self, target_position, target_radius):
        n = len(self.mass)
        self.target_position = np.asarray(target_position, dtype=float)
        self.target_radius = float(target_radius)
        self.in_contact = np.zeros(n, dtype=bool)
        self.min_clearance = np.full(n, np.inf)
        self.min_clearance_t = np.full(n, np.nan)
        self.contacts = []                   # (t, body, rel_speed)
        self.target_min_range = np.inf
        self.target_contact_t = None
        self.start_r = None                  # pad proxy: max sink toward Phobos from start
        self.max_drop = 0.0
        self.synced_p, self.synced_v = self.p.copy(), self.v.copy()
        self.last_sync = 0.0
        self.trace = []                      # (t, p_shuttle, v_shuttle) at 1 Hz

    # API state ----------------------------------------------------------
    def entity_state(self, entity):
        i = entity.body_index
        if i is None:
            p = np.asarray(entity.params.get("Location", np.zeros(3)), dtype=float)
            return p.copy(), np.zeros(3), np.zeros(3)
        if i > 0 and self.o["debris_sync_hold"]:
            return self.synced_p[i].copy(), self.synced_v[i].copy(), self.acc[i].copy()
        return self.p[i].copy(), self.v[i].copy(), self.acc[i].copy()

    def gravity_field(self, p):
        p = np.asarray(p, dtype=float)
        r = np.linalg.norm(p)
        g = -PHOBOS_MU * p / r ** 3
        if self.o["frame_terms"]:
            n2 = PHOBOS_RATE ** 2           # Mars tide; Mars along body +x, spin +z
            g = g + n2 * np.array([2.0 * p[0], -p[1], -p[2]])
        return g

    def apply_effect(self, entity, name, effect):
        i = entity.body_index
        world_vector = effect.frame.R @ effect.vector
        if i is None:
            return
        if effect.type == "Impulse":
            dv = world_vector / self.mass[i]
            self.v[i] += dv
            if i == 0:
                self.delta_v_shuttle += float(np.linalg.norm(dv))
        elif effect.type == "Acceleration":
            self.effects[i][name] = world_vector
            self.acc[i] = np.sum(list(self.effects[i].values()), axis=0)
        elif effect.type == "Force":
            self.effects[i][name] = world_vector / self.mass[i]
            self.acc[i] = np.sum(list(self.effects[i].values()), axis=0)
        # AngImpulse / Torque: attitude not propagated (net torque ~0 by LP)

    # propagation ------------------------------------------------------------
    def step_to(self, t_new):
        h_max = self.o["substep_s"]
        n = PHOBOS_RATE
        while self.t < t_new - 1e-12:
            h = min(h_max, t_new - self.t)
            a = self.acc
            if self.o["frame_terms"]:
                # centrifugal n^2 (x, y, 0) and Coriolis -2 n z_hat x v (spin +z)
                fict = np.empty_like(self.p)
                fict[:, 0] = n * n * self.p[:, 0] + 2.0 * n * self.v[:, 1]
                fict[:, 1] = n * n * self.p[:, 1] - 2.0 * n * self.v[:, 0]
                fict[:, 2] = 0.0
                a = a + fict
            self.v += a * h
            self.p += self.v * h
            self.t += h
            self._check_contacts()
            if self.o["debris_sync_hold"] and math.floor(self.t) > math.floor(self.last_sync):
                self.synced_p, self.synced_v, self.last_sync = self.p.copy(), self.v.copy(), self.t
            if not self.trace or self.t - self.trace[-1][0] >= 1.0 - 1e-9:
                self.trace.append((self.t, self.p[0].copy(), self.v[0].copy()))

    def _check_contacts(self):
        rel = self.p[1:] - self.p[0]
        clearance = np.linalg.norm(rel, axis=1) - self.radius[1:] - self.radius[0]
        better = clearance < self.min_clearance[1:]
        self.min_clearance[1:][better] = clearance[better]
        self.min_clearance_t[1:][better] = self.t
        touching = clearance < 0.0
        new = touching & ~self.in_contact[1:]
        for j in np.nonzero(new)[0]:
            rel_speed = float(np.linalg.norm(self.v[j + 1] - self.v[0]))
            self.contacts.append((self.t, int(j + 1), rel_speed))
        self.in_contact[1:] = touching
        rng = float(np.linalg.norm(self.target_position - self.p[0]))
        if rng < self.target_min_range:
            self.target_min_range = rng
        r = float(np.linalg.norm(self.p[0]))
        if self.start_r is None:
            self.start_r = r
        self.max_drop = max(self.max_drop, self.start_r - r)
        if rng < self.target_radius + self.radius[0] and self.target_contact_t is None:
            self.target_contact_t = self.t


# ==========================================================================
class System:
    def __init__(self, name, script, params, start_delay_s=0.0, overhead=0.0, seed=0):
        self.name, self.script, self.params = name, script, params
        self.start_delay_s, self.overhead = start_delay_s, overhead
        self.rng = np.random.default_rng(seed)
        self.go = threading.Event()
        self.alive, self.error, self.thread = True, None, None

    # st.GetThisSystem() API
    def GetParam(self, var_type, key):
        return BENCH_RESOLVE(st._get_path(self.params, key))

    GetParamArray = GetParam

    def HasParam(self, key):
        try:
            st._get_path(self.params, key)
            return True
        except KeyError:
            return False


BENCH_RESOLVE = None


class Bench:
    DEFAULTS = dict(
        substep_s=1.0 / 240.0, frame_terms=True, debris_sync_hold=False,
        deterministic_timing=True, shuttle_radius_m=3.0, target_radius_m=10.0,
        t_max_s=900.0, post_success_s=20.0, radar_start_s=0.35,
        radar_overhead=0.26, actuator_overhead=0.03, spawn_overhead=0.03,
        # sensitivity / stress options (defaults = nominal model)
        rotation_transpose=False,       # alternative attitude convention for the shuttle
        radar_outages=(),                # test-only: [(t0, t1), ...] radar publications dropped
        debris_extra_velocity=None,     # None | {"sigma": m/s} | {"profile": "live"}
        debris_extra_seed=0,
    )

    def __init__(self, simconf_path, options=None):
        global BENCH_RESOLVE
        self.o = dict(self.DEFAULTS, **(options or {}))
        st._BENCH = self
        BENCH_RESOLVE = self.resolve
        self.simconf = json.load(open(simconf_path, encoding="utf-8"))
        self.world = World(self, self.o)
        self.logs, self.alerts, self.fatals, self.published = [], [], [], []
        self.entities = {}
        self.systems = []
        self.heap, self.seq = [], 0
        self.turn_done = threading.Event()
        self.stopping = False
        self.success_t = None
        dm = self.simconf["DataManager"]
        self.sim_entity = BenchEntity(self, "SimEntity", _native(dm["SimEntity"]))
        self.start_datetime = self.sim_entity.params["Time"]["StartDateTAI"].as_datetime()
        self.phobos_frame = st.Frame()
        self.entities["Phobos"] = BenchEntity(self, "Phobos", {
            "Dynamics": {"GravitationalParameter_km3_s2": PHOBOS_MU / 1e9},
            "#Planet": {"General": {"Radius_m": PHOBOS_RADIUS_M}},
        })
        self.raw_entities = [e for e in dm["entities"]]
        self.system_params = {
            pathlib.Path(s["Source"]).stem: _native(s.get("Inst_Parameters", {}))
            for s in self.simconf["SystemManager"]["Systems"]
        }

    # entities -------------------------------------------------------------
    def build_entities(self, debris_override=None):
        """debris_override: list of native param dicts replacing simconf debris."""
        shuttle_raw = None
        for raw in self.raw_entities:
            name = raw.get("#Required", {}).get("Name")
            if not name or name.startswith("Debris"):
                continue
            params = _native(raw)
            if name == "SpaceBuggy":
                shuttle_raw = params
                continue
            self.entities[name] = BenchEntity(self, name, params)
        idx = self.world.add_body(shuttle_raw["Location"], shuttle_raw["Velocity"],
                                  shuttle_raw["Dynamics"]["Mass"], self.o["shuttle_radius_m"])
        self.entities["SpaceBuggy"] = BenchEntity(self, "SpaceBuggy", shuttle_raw, idx)
        if self.o["rotation_transpose"]:
            buggy = self.entities["SpaceBuggy"]
            buggy.R = buggy.R.T
            buggy.frame = st.Frame(origin_entity=buggy, rotation=buggy.R)
        if debris_override is None:
            debris = {}
            for raw in self.raw_entities:
                name = raw.get("#Required", {}).get("Name") or ""
                if name.startswith("Debris"):
                    debris[name] = _native(raw)
        else:
            debris = {p["#Required"]["Name"]: p for p in debris_override}
        refs = {}
        for i in range(len(debris)):
            name = f"Debris{i}"
            params = debris[name]
            body = self.world.add_body(params["Location"], params["Velocity"],
                                       params["Dynamics"]["Mass"], 0.5 * float(params["RadarDiameter_m"]))
            self.entities[name] = BenchEntity(self, name, params, body)
            refs[name] = Ref(name)
        self.sim_entity.params["Debris"] = refs
        extra = self.o["debris_extra_velocity"]
        if extra:
            rng = np.random.default_rng(10_000 + int(self.o["debris_extra_seed"]))
            n = len(debris)
            if "sigma" in extra:
                dv = rng.normal(0.0, float(extra["sigma"]), size=(n, 3))
            else:     # live track-speed mix: 75 % 0.5-5 m/s, 25 % 5-10 m/s, random direction
                u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
                fast = rng.random(n) < 0.25
                dv = u * np.where(fast, rng.uniform(5.0, 10.0, n), rng.uniform(0.5, 5.0, n))[:, None]
            self.world.v[1:] += dv
        self.world.finalize(self.entities["AsteroidCycler"].params["Location"], self.o["target_radius_m"])

    def resolve(self, value):
        if isinstance(value, Ref):
            return self.entities[value.name]
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        return value

    def asset_path(self, asset_type, path):
        p = str(path)
        if os.path.isabs(p):
            return pathlib.Path(p)
        if p.startswith("Local/"):
            return ASSETS / p[len("Local/"):]
        return ASSETS / p

    def read_entity_config(self, path):
        data = json.load(open(path, encoding="utf-8"))
        return st.ParamMap(_native(data.get("Entity", data)))

    def add_entity_from_config(self, config):
        params = config.data
        name = params["#Required"]["Name"]
        self.spawned.append(params)
        return Ref(name)

    # records ----------------------------------------------------------------
    def record_log(self, message, tag):
        self.logs.append((self.world.t, tag, message))

    def record_alert(self, message, tag):
        self.alerts.append((self.world.t, tag, message))
        if "Successfully escaped" in message and self.success_t is None:
            self.success_t = self.world.t

    def record_fatal(self, message):
        self.fatals.append((self.world.t, message))

    # systems / scheduler ----------------------------------------------------
    def add_system(self, name, script, params, start_delay_s=0.0, overhead=0.0, seed=0):
        system = System(name, script, params, start_delay_s, overhead, seed)
        self.systems.append(system)
        return system

    def _push(self, wake, system):
        self.seq += 1
        heapq.heappush(self.heap, (wake, self.seq, system))

    def sleep(self, seconds):
        system = getattr(st._local, "system", None)
        if system is None:
            return _REAL_SLEEP(seconds)
        extra = system.rng.uniform(0.0, system.overhead) if system.overhead > 0 else 0.0
        system.go.clear()
        self._push(self.world.t + max(float(seconds), 0.0) * (1.0 + extra), system)
        self.turn_done.set()
        system.go.wait()
        if self.stopping:
            raise BenchStop()

    def _thread_main(self, system):
        st._local.system = system
        system.go.wait()
        try:
            if self.stopping:
                raise BenchStop()
            if callable(system.script):
                system.script()
            else:
                path = str(system.script)
                code = compile(open(path, encoding="utf-8").read(), path, "exec")
                exec(code, {"__name__": "__main__", "__file__": path, "__builtins__": builtins})
        except BenchStop:
            pass
        except BaseException:
            system.error = traceback.format_exc()
        finally:
            system.alive = False
            self.turn_done.set()

    def run(self, stop_condition=None):
        saved_hook = sys.excepthook          # official scripts replace it
        try:
            return self._run(stop_condition)
        finally:
            sys.excepthook = saved_hook

    def _run(self, stop_condition=None):
        time.sleep = self.sleep
        if self.o["deterministic_timing"]:
            time.perf_counter = lambda: self.world.t
        try:
            for system in self.systems:
                system.thread = threading.Thread(target=self._thread_main, args=(system,), daemon=True)
                system.thread.start()
                self._push(system.start_delay_s, system)
            while self.heap:
                wake, _, system = heapq.heappop(self.heap)
                if wake > self.o["t_max_s"]:
                    break
                self.world.step_to(wake)
                if not system.alive:
                    continue
                self.turn_done.clear()
                system.go.set()
                self.turn_done.wait()
                if system.error:
                    raise RuntimeError(f"system {system.name} failed:\n{system.error}")
                if self.fatals:
                    raise RuntimeError(f"logger_fatal: {self.fatals[-1][1][:2000]}")
                if self.success_t is not None and self.world.t >= self.success_t + self.o["post_success_s"]:
                    break
                if stop_condition is not None and stop_condition(self):
                    break
        finally:
            self.stopping = True
            for system in self.systems:
                if system.alive:
                    system.go.set()
            for system in self.systems:
                if system.thread is not None:
                    system.thread.join(timeout=5.0)
            time.sleep = _REAL_SLEEP
            time.perf_counter = _REAL_PERF


_REAL_SLEEP = time.sleep
_REAL_PERF = time.perf_counter


# ==========================================================================
def gravity_system_factory(bench):
    """ApplyGravity (no source available): 1 Hz, entities whose Systems list it."""
    def run():
        targets = [e for e in bench.entities.values()
                   if "ApplyGravity" in (e.params.get("#Required", {}).get("Systems") or [])
                   and e.body_index is not None]
        while True:
            for entity in targets:
                effect = st.PhysEffect("Acceleration", bench.phobos_frame)
                effect.setVector(bench.world.gravity_field(bench.world.p[entity.body_index]))
                entity.setPhysEffect("Gravity", effect)
            time.sleep(1.0)
    return run


def spawn_debris_copy(seed, workdir, hunting=False):
    """Controlled test copy of official SpawnDebris.py: only `seed = 100` (and,
    if requested, the hunting flag) changed; verified line-by-line."""
    src = ASSETS / "Repos" / "STU4-Sim" / "Systems" / "SpawnDebris.py"
    text = open(src, encoding="utf-8").read()
    import re
    found = re.findall(r"\nseed = (\d+)\n", text)   # official: 100 until 2026-11-11, then 8193
    assert len(found) == 1
    official_seed = int(found[0])
    new = text.replace(f"\nseed = {official_seed}\n", f"\nseed = {int(seed)}\n")
    if hunting:
        assert new.count("doTrajectoryAdjustment = False") == 1
        new = new.replace("doTrajectoryAdjustment = False", "doTrajectoryAdjustment = True")
    a, b = text.splitlines(), new.splitlines()
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(changed) == (2 if hunting else (0 if seed == official_seed else 1))
    out = pathlib.Path(workdir) / f"SpawnDebris_seed{int(seed)}{'_hunting' if hunting else ''}.py"
    out.write_text(new, encoding="utf-8")
    return out


def generate_debris(simconf_path, seed, workdir):
    """Run the official spawning branch (DoSpawning=True) of the seed copy."""
    bench = Bench(simconf_path, dict(t_max_s=1.0))
    bench.build_entities(debris_override=[])
    bench.sim_entity.params["DoSpawning"] = True
    bench.spawned = []
    script = spawn_debris_copy(seed, workdir)
    bench.add_system("SpawnDebris", script, bench.system_params["SpawnDebris"])
    bench.run(stop_condition=lambda b: True)      # stop at its first sleep (spawn done)
    assert bench.published == ["SaveSim"], bench.published
    return bench.spawned
