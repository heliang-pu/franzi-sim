"""Adapter registry. Engines are imported only when their backend is selected."""

import importlib

# name -> (module, class, implementation status, verification status)
REGISTRY = {
    "mujoco": (
        "simbridge.backends.mujoco_backend",
        "MujocoBackend",
        "implemented",
        (
            "verified on the pro6000 (Linux): stock-SDK acceptance, framework "
            "and simulator-owned loops, wheel_robot_26.8.16_3 "
            "(outputs/sdk_backends_urdf_v5)"
        ),
    ),
    "isaacsim": (
        "simbridge.backends.isaacsim_backend",
        "IsaacSimBackend",
        (
            "draft: API calls checked against the Isaac Sim 5.1.0 sources on "
            "the 4090, never executed"
        ),
        (
            "NOT VERIFIED: the 4090 run was deferred (no env_isaaclab; the 5.1 "
            "installs there need an EULA decision)"
        ),
    ),
}


def status_table():
    return {
        name: {"implementation": impl, "verification": verified}
        for name, (_, _, impl, verified) in REGISTRY.items()
    }


def create(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(REGISTRY)}")
    module, cls, _, _ = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)()
