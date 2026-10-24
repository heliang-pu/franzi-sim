"""Robot configuration: SDK joint order, sign/offset mapping, limits, gripper,
cameras, frames and safety thresholds. Loaded from config/robot/*.yaml.

SDK joint values follow ``q_sdk = sign * q_sim + offset`` (rad). Every number
here is either copied from a named source file or derived by
check_sdk_kinematics.py; none of it is hardware-verified.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from simbridge import ROOT


def urdf_limits(urdf_path, joint_names):
    """(n, 2) lower/upper position limits of the named URDF joints (rad)."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    return np.array(
        [
            [float(joints[n].find("limit").get(k)) for k in ("lower", "upper")]
            for n in joint_names
        ]
    )


@dataclass(frozen=True)
class UnitMap:
    name: str
    side: int
    sim_joints: tuple
    sign: np.ndarray
    offset: np.ndarray
    sim_limits: np.ndarray
    sdk_limits: np.ndarray
    controller_limits: np.ndarray | None
    vmax: np.ndarray
    amax: np.ndarray
    writable: bool
    status: tuple = field(default=())

    def to_sim(self, q_sdk):
        return self.sign * (np.asarray(q_sdk, float) - self.offset)

    def to_sdk(self, q_sim):
        return self.sign * np.asarray(q_sim, float) + self.offset

    def rate_to_sdk(self, values_sim):
        """Velocities and joint torques only flip with the joint direction."""
        return self.sign * np.asarray(values_sim, float)

    def check_model_ranges(self, ranges, engine):
        """An engine's joint ranges (within 1 microradian: build_task writes 9
        significant digits) must lie inside the URDF's and contain every
        target the SDK layer accepts (sdk_limits mapped to simulator joints)."""
        accepted = np.sort(
            np.stack(
                [
                    self.to_sim(self.sdk_limits[:, 0]),
                    self.to_sim(self.sdk_limits[:, 1]),
                ],
                axis=1,
            ),
            axis=1,
        )
        for j, (lo, hi), (ulo, uhi), (alo, ahi) in zip(
            self.sim_joints, ranges, self.sim_limits, accepted
        ):
            if lo < ulo - 1e-6 or hi > uhi + 1e-6:
                raise ValueError(f"{j}: {engine} range exceeds the URDF's")
            if alo < lo - 1e-6 or ahi > hi + 1e-6:
                raise ValueError(
                    f"{j}: the SDK accepts [{alo:.6f}, {ahi:.6f}] rad but {engine} "
                    f"allows only [{lo:.6f}, {hi:.6f}]"
                )


@dataclass(frozen=True)
class GripperMap:
    side: str
    sim_joints: tuple
    finger_links: tuple
    directions: np.ndarray
    stroke_m: float
    max_closure_rate: float
    arrival_tolerance: int

    def finger_targets(self, closure):
        return dict(
            zip(self.sim_joints, self.directions * self.stroke_m * float(closure))
        )

    def closure(self, q_fingers):
        q = np.asarray(q_fingers, float)
        return float(np.dot(q, self.directions) / (2 * self.stroke_m))

    @staticmethod
    def closure_from_position(position):
        """Vendor AG gripper: 0 closed, 1000 fully open."""
        return 1.0 - float(position) / 1000.0

    @staticmethod
    def position_from_closure(closure):
        return int(np.clip(round(1000 * (1 - closure)), 0, 1000))


@dataclass(frozen=True)
class CameraSpec:
    sdk_name: str
    sim_name: str
    device: str
    parent_link: str
    width: int
    height: int


class RobotConfig:
    def __init__(self, path, convention=None):
        self.path = Path(path)
        raw = yaml.safe_load(self.path.read_text())
        self.raw = raw
        self.name = raw["name"]
        self.axis_count = int(raw["sdk"]["axis_count"])
        self.convention = convention or raw["joint_convention"]["active"]
        if self.convention not in raw["joint_convention"]["options"]:
            raise ValueError(f"unknown joint convention {self.convention!r}")
        self.urdf = ROOT / raw["sim_model"]["urdf"]
        planning = yaml.safe_load((ROOT / raw["limits"]["rates_source"]).read_text())
        self._rates = planning["joint_limits"]
        self._margin = float(raw["limits"]["position_margin_rad"])
        self._policy = raw["limits"]["position_policy"]
        self.units = {}
        for name, spec in raw["units"].items():
            self.units[name] = self._unit(name, spec)
        self.arms = {k: self.units[k] for k in ("left", "right")}
        self.grippers = {}
        for side, spec in raw["grippers"].items():
            self.grippers[side] = GripperMap(
                side=side,
                sim_joints=tuple(spec["sim_joints"]),
                finger_links=tuple(spec["finger_links"]),
                directions=np.asarray(spec["directions"], float),
                stroke_m=float(spec["finger_stroke_m"]),
                max_closure_rate=float(spec["max_closure_rate_per_s"]),
                arrival_tolerance=int(spec["arrival_tolerance"]),
            )
        cams = raw["cameras"]
        self.cameras = {
            name: CameraSpec(
                sdk_name=name,
                sim_name=spec["sim"],
                device=spec["device"],
                parent_link=spec["parent_link"],
                width=int(cams["image"]["width"]),
                height=int(cams["image"]["height"]),
            )
            for name, spec in cams["streams"].items()
        }
        self.safety = raw["safety"]
        self.control = raw["control"]
        self.frames = raw["frames"]

    def _unit(self, name, spec):
        joints = tuple(spec["sim_joints"])
        n = len(joints)
        mapping = spec["mapping"][self.convention]
        sign = np.asarray(mapping["sign"], float)
        offset = np.asarray(mapping["offset_rad"], float)
        if sign.shape != (n,) or offset.shape != (n,) or not np.all(abs(sign) == 1):
            raise ValueError(
                f"{name}: mapping must give one +-1 sign and offset per joint"
            )
        sim = urdf_limits(self.urdf, joints)
        mapped = np.sort(sign[:, None] * sim + offset[:, None], axis=1)
        mapped = mapped + np.array([self._margin, -self._margin])
        controller = None
        if spec.get("controller_limits_deg") is not None:
            # Stored in controller (MDH) coordinates; expressed here in the
            # active convention (the same numbers when that is controller_mdh).
            ctrl = spec["mapping"]["controller_mdh"]
            c_sign = np.asarray(ctrl["sign"], float)[:, None]
            c_offset = np.asarray(ctrl["offset_rad"], float)[:, None]
            limits = np.deg2rad(np.asarray(spec["controller_limits_deg"], float))
            controller = np.sort(
                sign[:, None] * c_sign * (limits - c_offset) + offset[:, None], axis=1
            )
        sdk = mapped.copy()
        if self._policy == "intersection" and controller is not None:
            # Same margin on the controller side: the derived MuJoCo scene clips
            # its ranges to exactly this (build_task.align_joint_limits).
            sdk[:, 0] = np.maximum(mapped[:, 0], controller[:, 0] + self._margin)
            sdk[:, 1] = np.minimum(mapped[:, 1], controller[:, 1] - self._margin)
        elif self._policy != "simulation_model":
            raise ValueError(f"unknown position policy {self._policy!r}")
        return UnitMap(
            name=name,
            side=int(spec["side"]),
            sim_joints=joints,
            sign=sign,
            offset=offset,
            sim_limits=sim,
            sdk_limits=sdk,
            controller_limits=controller,
            vmax=np.array([self._rates[j]["max_velocity"] for j in joints], float),
            amax=np.array([self._rates[j]["max_acceleration"] for j in joints], float),
            writable=bool(spec["writable"]),
            status=tuple(mapping.get("status", ())),
        )

    def unit_names(self, side):
        """SDK side value -> controlled arm unit names; None when not an arm."""
        return {0: ("left",), 10: ("right",), 15: ("left", "right")}.get(int(side))

    def unit_by_side(self, side):
        for unit in self.units.values():
            if unit.side == int(side):
                return unit
        return None

    def controlled_sim_joints(self):
        joints = [j for u in self.arms.values() for j in u.sim_joints]
        joints += [j for g in self.grippers.values() for j in g.sim_joints]
        return joints

    def describe(self):
        """Plain-data summary saved with every run."""
        return {
            "config": str(self.path.relative_to(ROOT)),
            "joint_convention": self.convention,
            "hardware_verified": bool(
                self.raw["joint_convention"]["hardware_verified"]
            ),
            "position_policy": self._policy,
            "units": {
                name: {
                    "side": u.side,
                    "sim_joints": list(u.sim_joints),
                    "sign": u.sign.astype(int).tolist(),
                    "offset_rad": u.offset.tolist(),
                    "sdk_limits_rad": u.sdk_limits.round(6).tolist(),
                    "writable": u.writable,
                    "status": list(u.status),
                }
                for name, u in self.units.items()
            },
            "cameras": {k: v.sim_name for k, v in self.cameras.items()},
        }


def load_robot(path=None, convention=None):
    return RobotConfig(path or ROOT / "config/robot/franzi_sdk.yaml", convention)
