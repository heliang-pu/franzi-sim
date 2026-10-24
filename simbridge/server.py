"""Unified simulation server for the unmodified V1R1C9 SDK clients.

    python -m simbridge.server --backend config/backends/mujoco.yaml \
        --output outputs/<new_run>

The backend file names the engine, the interpreter that has it, the robot
config and the connection profile (host and ports) the server listens on.
Clients switch between simulators, or the RK3588 controller, only by choosing
a different connection profile. Listens on loopback unless the profile says
otherwise; exits after ``idle_timeout_s`` without RPC traffic.
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import yaml

from simbridge import ROOT
from simbridge.backends import REGISTRY, create, status_table
from simbridge.protocol import ARM_PREFIX, SENSOR_ROUTES, Endpoint, arm_catalog
from simbridge.robot import load_robot
from simbridge.runtime import Runtime
from simbridge.services import ArmService, SensorService


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_backend(path):
    cfg = yaml.safe_load(resolve(path).read_text())
    if cfg["backend"] not in REGISTRY:
        raise KeyError(f"{path}: unknown backend {cfg['backend']!r}")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, help="config/backends/<name>.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-port", type=int, help="override the profile (0 = any)")
    parser.add_argument("--sensor-port", type=int, help="override the profile")
    parser.add_argument("--record", action="store_true", help="write session.mp4")
    parser.add_argument("--loop", choices=["framework", "simulator"])
    parser.add_argument("--idle-timeout", type=float)
    parser.add_argument(
        "--joint-convention",
        choices=["controller_mdh", "urdf_passthrough"],
        help="override the robot config (urdf_passthrough is for debugging only)",
    )
    args = parser.parse_args(argv)
    cfg = read_backend(args.backend)
    profile = yaml.safe_load(resolve(cfg["connection"]).read_text())
    options = dict(cfg.get("runtime", {}))
    options.update(cfg.get("options", {}))
    if args.record:
        options["record"] = True
    if args.loop:
        options["loop"] = args.loop
    if args.idle_timeout is not None:
        options["idle_timeout_s"] = args.idle_timeout
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    robot = load_robot(resolve(cfg["robot"]), args.joint_convention)
    backend = create(cfg["backend"])
    backend.load(robot, options, output)
    caps = backend.capabilities()
    if options.get("loop", "framework") not in caps.loop_modes:
        raise SystemExit(
            f"{backend.name} does not support loop mode {options['loop']!r}"
        )
    runtime = Runtime(robot, backend, options, output)
    host = profile["host"]
    arm_port = profile["arm_port"] if args.arm_port is None else args.arm_port
    sensor_port = (
        profile["sensor_port"] if args.sensor_port is None else args.sensor_port
    )
    from sdk.stock_sdk import UPPER

    arm = Endpoint(
        "arm",
        host,
        arm_port,
        arm_catalog(UPPER / "core/client"),
        ArmService(runtime, robot, backend).dispatch,
        prefix=ARM_PREFIX,
    ).start()
    sensor = Endpoint(
        "sensor",
        host,
        sensor_port,
        SENSOR_ROUTES,
        SensorService(runtime, robot, backend).dispatch,
    ).start()
    ready = {
        "host": host,
        "arm_port": arm.port,
        "sensor_port": sensor.port,
        "backend": backend.name,
        "backend_label": backend.label,
        "backend_status": status_table()[backend.name],
        "backend_config": str(resolve(args.backend).relative_to(ROOT)),
        "python": sys.executable,
        "loop_mode": options.get("loop", "framework"),
        "physics_dt_s": backend.dt,
        "capabilities": caps.as_dict(),
        "robot": robot.describe(),
        "sdk": "unmodified V1R1C9 clients; routes from the vendor client sources",
        "arm_routes": len(arm.routes),
        "implemented_arm_methods": sorted(ArmService(runtime, robot, backend).handlers),
        "clocks": {
            "sim_time": "physics clock in seconds from scene start",
            "control": "setpoints update every physics step; trace at trace_hz",
            "image_timestamp": "UTC milliseconds when the frame was rendered",
        },
    }
    (output / "ready.json").write_text(json.dumps(ready, indent=2))
    print(
        json.dumps(
            {k: ready[k] for k in ("host", "arm_port", "sensor_port", "backend")}
        ),
        flush=True,
    )
    signal.signal(signal.SIGTERM, lambda *_: runtime.stop_event.set())
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()
        sensor.close()
        result = runtime.result()
        result["rpc_calls"] = arm.calls + sensor.calls
        runtime.close()
        # Write before closing the engine: some (Isaac Sim's app) may end the
        # interpreter on close.
        (output / "server_result.json").write_text(json.dumps(result, indent=2))
        backend.close()


if __name__ == "__main__":
    main()
