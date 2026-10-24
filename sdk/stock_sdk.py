"""Locate and import the unmodified SDK supplied in feimi1010.zip.

``connect(profile)`` is the only switch between the RK3588 controller and a
simulator: a connection profile names a host and the two ports, nothing else.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "sdk_review/feimi1010/sdk"
UPPER = SDK / "上半身运控/SDK_V1R1C9/sdk/sdk_py"
CHASSIS = SDK / "底盘/底盘sdk/python"


def _paths():
    for p in (UPPER, CHASSIS):
        if not p.is_dir():
            raise FileNotFoundError(p)
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_wire():
    """The SDK's bundled MessagePack and rest_rpc route hash, for servers."""
    _paths()
    from core.libs import msgpack
    from core.libs.inbc_rpc.md5_hash import md5_hash32

    return msgpack, md5_hash32


def load():
    msgpack, md5_hash32 = load_wire()
    # The chassis client imports msgpack by its standard package name; use the
    # same bundled implementation as the arm client without editing SDK files.
    sys.modules.setdefault("msgpack", msgpack)
    from core.client import ArmClient
    from module_communication import ModuleCommunicationClient

    return ArmClient, ModuleCommunicationClient, msgpack, md5_hash32


def read_profile(path):
    profile = yaml.safe_load(Path(path).read_text())
    for key in ("host", "arm_port", "sensor_port"):
        if key not in profile:
            raise KeyError(f"{path}: connection profile lacks {key!r}")
    return profile


def connect(profile, **arm_options):
    """Stock ArmClient and connected ModuleCommunicationClient for a profile.

    ``profile`` is a path or an already-read dict. Extra keyword arguments go
    to ArmClient unchanged (timeouts, heartbeat, raise_on_error, ...).
    """
    if not isinstance(profile, dict):
        profile = read_profile(profile)
    ArmClient, ModuleCommunicationClient, _, _ = load()
    arm = ArmClient(profile["host"], int(profile["arm_port"]), **arm_options)
    if not arm.is_connected():
        arm.shutdown()
        raise ConnectionError(
            f"no SDK server at {profile['host']}:{profile['arm_port']}"
        )
    sensors = ModuleCommunicationClient(profile["host"], int(profile["sensor_port"]))
    sensors.connect(timeout_sec=float(profile.get("connect_timeout_s", 5.0)))
    return arm, sensors
