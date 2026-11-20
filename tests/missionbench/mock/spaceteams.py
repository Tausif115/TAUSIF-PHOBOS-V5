"""Offline mock of the SpaceTeams Python API (PHOBOS V5 Step 0 mission bench).

Lets the OFFICIAL STU4-Sim system scripts (Radar.py, JetSelectionActuator.py,
Scoring.py, SpawnDebris.py) and the real Controller.py run unmodified,
in lock-step simulated time. Only the API surface those scripts use is
implemented. Physics, frames and the clock live in `missionbench.world`.

Conventions (documented model assumptions; see tests/missionbench/README.md):
- World frame = Phobos body-fixed frame (positions in m, Phobos-centred).
- Entity body frames: rotation from the simconf w-last quaternion
  (body -> Phobos axes), constant (attitude is not propagated; the actuator
  LP commands zero net torque and the controller zero angular acceleration).
- ExprIn() returns a numpy array (as the official scripts index/dot it).
"""
import datetime as _dt
import threading as _threading

import numpy as _np

_BENCH = None          # set by missionbench.world.Bench


class _Names:
    def __init__(self, label):
        self._label = label

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return name


VarType = _Names("VarType")
Severity = _Names("Severity")
AssetType = _Names("AssetType")
PhysEffectType = _Names("PhysEffectType")


class Entity:                      # used by scripts only as a type annotation
    pass


class timestamp:
    def __init__(self, value):
        self._dt = value

    def as_datetime(self):
        return self._dt


# --------------------------------------------------------------------------
class Frame:
    """origin_entity None -> Phobos body-fixed (world) frame."""

    def __init__(self, origin_entity=None, rotation=None):
        self.origin_entity = origin_entity
        self.R = _np.eye(3) if rotation is None else _np.asarray(rotation, dtype=float)

    def origin_state(self):
        if self.origin_entity is None:
            return _np.zeros(3), _np.zeros(3)
        return self.origin_entity._state()[:2]


class frames:
    Frame = Frame

    class FramedVector:
        def __init__(self, vector, frame):
            self.v = _np.asarray(vector, dtype=float)
            self.frame = frame

        def ExprIn(self, frame):
            return frame.R.T @ (self.frame.R @ self.v)


class _WorldVector:
    """A world-axes vector (relative quantity) awaiting ExprIn()."""

    def __init__(self, world_vector):
        self.w = _np.asarray(world_vector, dtype=float)

    def ExprIn(self, frame):
        return frame.R.T @ self.w


class _FramedState:
    def __init__(self, p, v, a):
        self.p, self.v, self.a = p, v, a

    def loc_WRT(self, frame):
        return _WorldVector(self.p - frame.origin_state()[0])

    def vel_WRT(self, frame):
        return _WorldVector(self.v - frame.origin_state()[1])

    def loc_WRT_ExprIn(self, frame):
        return self.loc_WRT(frame).ExprIn(frame)

    def vel_WRT_ExprIn(self, frame):
        return self.vel_WRT(frame).ExprIn(frame)


class _Location:
    def __init__(self, p):
        self.p = p

    def WRT(self, frame):
        return _WorldVector(self.p - frame.origin_state()[0])

    def WRT_ExprIn(self, frame):
        return self.WRT(frame).ExprIn(frame)


class _Rotation:
    def __init__(self, R):
        self.R = R

    def DCM_WRT(self, frame):
        return frame.R.T @ self.R


class PhysEffect:
    def __init__(self, effect_type, frame):
        self.type = effect_type
        self.frame = frame
        self.vector = _np.zeros(3)

    def setVector(self, vector):
        self.vector = _np.asarray(vector, dtype=float).reshape(3)


# --------------------------------------------------------------------------
class ParamMap:
    def __init__(self, data=None):
        self.data = {} if data is None else data

    def AddParam(self, var_type, key, value):
        _set_path(self.data, key, value)

    AddOrSetParam = AddParam
    SetParam = AddParam

    def GetParam(self, var_type, key):
        return _get_path(self.data, key)

    def HasParam(self, key):
        try:
            _get_path(self.data, key)
            return True
        except KeyError:
            return False


def _keys(key):
    return [key] if isinstance(key, str) else list(key)


def _get_path(store, key):
    node = store
    for k in _keys(key):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(key)
        node = node[k]
    return node


def _set_path(store, key, value):
    ks = _keys(key)
    node = store
    for k in ks[:-1]:
        node = node.setdefault(k, {})
    node[ks[-1]] = value


class file_io:
    @staticmethod
    def ReadEntityConfig(path):
        return _BENCH.read_entity_config(path)


class path_utils:
    @staticmethod
    def AssetPathToReal(asset_type, path):
        return _BENCH.asset_path(asset_type, path)


class PlanetUtils:
    class _NWU:
        def __init__(self, north, west, up):
            self._n, self._w, self._u = north, west, up

        def north(self):
            return self._n

        def west(self):
            return self._w

        def up(self):
            return self._u

    @staticmethod
    def NorthWestUpFromLocation(location, radius):
        loc = _np.asarray(location, dtype=float)
        up = loc / _np.linalg.norm(loc)
        z = _np.array([0.0, 0.0, 1.0])
        north = z - up * float(z @ up)
        north /= _np.linalg.norm(north)
        west = _np.cross(up, north)
        return PlanetUtils._NWU(north, west, up)


class math:
    @staticmethod
    def Euler321_to_DCM(angles):
        a = _np.radians(_np.asarray(angles, dtype=float).reshape(3))
        def rx(t): return _np.array([[1, 0, 0], [0, _np.cos(t), _np.sin(t)], [0, -_np.sin(t), _np.cos(t)]])
        def ry(t): return _np.array([[_np.cos(t), 0, -_np.sin(t)], [0, 1, 0], [_np.sin(t), 0, _np.cos(t)]])
        def rz(t): return _np.array([[_np.cos(t), _np.sin(t), 0], [-_np.sin(t), _np.cos(t), 0], [0, 0, 1]])
        return rx(a[2]) @ ry(a[1]) @ rz(a[0])


class _SimClock:
    @staticmethod
    def GetTimeNow():
        return timestamp(_BENCH.start_datetime + _dt.timedelta(seconds=_BENCH.world.t))

    @staticmethod
    def SetTimescale(value):
        pass


class SimGlobals:
    SimClock = _SimClock

    @staticmethod
    def GetSimEntity():
        return _BENCH.sim_entity

    @staticmethod
    def GetBaseFrame():
        return _BENCH.phobos_frame

    @staticmethod
    def SampleVectorField(name, framed_state):
        if name != "Gravity":
            raise KeyError(name)
        return frames.FramedVector(_BENCH.world.gravity_field(framed_state.p), _BENCH.phobos_frame)

    @staticmethod
    def AddEntity(config):
        return _BENCH.add_entity_from_config(config)

    @staticmethod
    def Publish(topic, payload):
        _BENCH.published.append(topic)


# --------------------------------------------------------------------------
_local = _threading.local()


def connect_to_sim(argv):
    pass


def leave_sim():
    pass


def BeforeInit(callback):
    pass


def BeforeRuntime(callback):
    pass


def GetThisSystem():
    return _local.system


def logger_fatal(message):
    _BENCH.record_fatal(message)


def OnScreenLogMessage(message, tag, severity):
    _BENCH.record_log(message, tag)


def OnScreenAlert(message, tag, severity):
    _BENCH.record_alert(message, tag)
