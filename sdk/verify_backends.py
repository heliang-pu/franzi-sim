"""Run one unmodified-SDK acceptance program against several simulator backends.

    python -m sdk.verify_backends --output outputs/<new_run> \
        --backends mujoco,mujoco_owned_loop --task --fault-probe --record

For each backend config (config/backends/<name>.yaml) this starts the unified
server under the interpreter named there, runs sdk_acceptance.py with that
backend's connection profile (the only thing that differs between runs), stops
the server and scores the run. Results are kept apart by what they establish:
communication compatibility, control semantics, kinematic mapping, physical
behaviour, fault handling and full-task success. Nothing is averaged into a
single pass rate. Task success uses the simulator's ground truth from the
server trace, which never reaches the SDK client.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

from robot_kinematics import Kinematics
from scenes.pick_place.task import GOALS
from sdk.stock_sdk import SDK
from sim.build_task import TABLE_Z
from simbridge import ROOT
from simbridge.robot import load_robot

PROGRAM = ROOT / "sdk/sdk_acceptance.py"  # run as module sdk.sdk_acceptance


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sdk_integrity():
    """Every file of the original sdk.zip, byte-compared with the extracted copy."""
    archive = SDK.parent / "sdk.zip"
    same, differ, missing = 0, [], []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            local = SDK.parent / info.filename
            if not local.exists():
                missing.append(info.filename)
            elif local.read_bytes() != z.read(info):
                differ.append(info.filename)
            else:
                same += 1
    py = [p for p in SDK.rglob("*.py")]
    return {
        "archive": str(archive.relative_to(ROOT)),
        "identical_files": same,
        "different_files": differ,
        "missing_files": missing,
        "python_sources": len(py),
        "unchanged": not differ and not missing,
    }


def wait_for(path, process, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def settled_sample(trace, after, units=("left", "right")):
    """First trace sample at or after sim time ``after`` with both arms still."""
    for sample in trace:
        if sample["t"] >= after and all(sample["motion_status"][u] == 0 for u in units):
            return sample
    return None


def score_mapping(trace, robot):
    """SDK values must equal sign * sim + offset in every trace sample."""
    worst = 0.0
    for sample in trace:
        for name, unit in robot.arms.items():
            sim = np.array([sample["sim_joint_q"][j] for j in unit.sim_joints])
            worst = max(
                worst, float(np.max(np.abs(unit.to_sdk(sim) - sample["state"][name])))
            )
    return worst


def score_kinematics(trace, robot):
    """The shared kinematics (robot_kinematics, which the SDK's Fk/IK endpoints
    use) at the measured SDK joints and waist, against the engine's own
    {side}_grasp positions."""
    kin = Kinematics()
    waist = robot.units["waist"]
    # A backend that tracks no grasp sites (SimBackend's default) is not scored.
    samples = [
        s
        for s in trace[::10]
        if all(f"{side}_grasp" in s["frames"] for side in robot.arms)
    ]
    worst = None
    if samples:
        worst = max(
            float(
                np.linalg.norm(
                    kin.fk(
                        side,
                        robot.arms[side].to_sim(s["state"][side]),
                        dict(zip(waist.sim_joints, waist.to_sim(s["state"]["waist"]))),
                    )[0]
                    - s["frames"][f"{side}_grasp"]
                )
            )
            for s in samples
            for side in robot.arms
        )
    return {
        "shared_kinematics_vs_sim_grasp_site_max_error_m": worst,
        "samples": len(samples),
    }


def score_task(trace, events):
    stops = [e["sim_time"] for e in events if e["event"] == "TeleStop"]
    if not stops:
        return None
    end = stops[-1]
    start = max((e["sim_time"] for e in events if e["event"] == "TeleStart"), default=0)
    window = [s for s in trace if start <= s["t"] <= end]
    if not window:
        return None
    final = window[-1]["objects"]
    result = {"evaluated_at_sim_time": window[-1]["t"], "parts": {}}
    ok = True
    for side, goal in GOALS.items():
        o = final[side]
        airborne = sum(
            1
            for s in window
            if s["objects"][side]["xyz"][2] > TABLE_Z + 0.10
            and len(s["objects"][side]["jaw_contacts"]) == 2
        )
        part = {
            "xy_error_m": float(np.linalg.norm(np.array(o["xyz"][:2]) - goal)),
            "z_error_m": float(abs(o["xyz"][2] - (TABLE_Z + 0.006 + 0.0275))),
            "speed_m_s": o["velocity_m_s"],
            "touching_robot": o["touching_robot"],
            "airborne_bilateral_samples_25hz": airborne,
        }
        part["success"] = bool(
            part["xy_error_m"] < 0.025
            and part["z_error_m"] < 0.004
            and part["speed_m_s"] < 0.025
            and not part["touching_robot"]
            and airborne >= 10
        )
        ok &= part["success"]
        result["parts"][side] = part
    result["success"] = bool(ok)
    result["criteria"] = (
        "same thresholds as PickPlaceScene.complete, plus >= 0.4 s airborne "
        "with both jaws in contact; evaluated at the end of the task replay"
    )
    return result


def video_info(path):
    if not path.exists():
        return None
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    stream = json.loads(out.stdout or "{}").get("streams", [{}])[0]
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"file": str(shown), **stream}


def run_backend(name, out, args):
    cfg_path = ROOT / "config/backends" / f"{name}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    python = os.path.expandvars(cfg["python"])  # e.g. ${ISAAC_PYTHON}
    if "$" in python:
        raise SystemExit(f"{name}: set the environment variable in {cfg['python']}")
    out.mkdir(parents=True)
    server_out, client_out = out / "server", out / "client"
    command = [
        python, "-m", "simbridge.server", "--backend", str(cfg_path),
        "--output", str(server_out), "--idle-timeout", "60",
    ]  # fmt: skip
    if args.record:
        command.append("--record")
    print(f"== {name}: {' '.join(command)}", flush=True)
    # Snapshot of every input that defines this run.
    (out / "config").mkdir()
    for source in (cfg_path, ROOT / cfg["connection"], ROOT / cfg["robot"]):
        shutil.copy2(source, out / "config" / source.name)
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            if not wait_for(server_out / "ready.json", server, 300):
                return {"error": "server did not start; see server.log"}
            client = [
                sys.executable, "-m", "sdk.sdk_acceptance", "--connection", cfg["connection"],
                "--output", str(client_out),
            ]  # fmt: skip
            if args.task:
                client.append("--task")
            if args.fault_probe:
                client.append("--fault-probe")
            with (out / "client.log").open("w") as clog:
                code = subprocess.run(
                    client, cwd=ROOT, stdout=clog, stderr=subprocess.STDOUT, check=False
                ).returncode
            (out / "client_exit.json").write_text(json.dumps({"code": code}))
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=600)
            except subprocess.TimeoutExpired:
                server.kill()
    return score(name, cfg, out, code)


def score(name, cfg, out, client_code):
    robot = load_robot()
    ready = json.loads((out / "server/ready.json").read_text())
    result = json.loads((out / "server/server_result.json").read_text())
    trace = json.loads((out / "server/trace.json").read_text())
    verification = json.loads((out / "client/verification.json").read_text())
    events = result["controller_events"]
    probe_start = None
    if "fault_probe" in verification:
        moves = [
            e["sim_time"]
            for e in events
            if e["event"] == "MoveJ" and e["units"] == ["left"]
        ]
        probe_start = moves[-2] if len(moves) >= 2 else None
    faults_outside_probe = [
        f
        for f in result["faults"]
        if probe_start is None or f["sim_time"] < probe_start
    ]
    first_move = next(e["sim_time"] for e in events if e["event"] == "MoveJ")
    pose = settled_sample(trace, first_move + 0.1)
    outcomes = {}
    for call in result["rpc_calls"]:
        outcomes[call["outcome"]] = outcomes.get(call["outcome"], 0) + 1
    by_category = verification["summary_by_category"]
    report = {
        "backend": ready["backend"],
        "label": ready["backend_label"],
        "registry_status": ready["backend_status"],
        "loop_mode": ready["loop_mode"],
        "python": ready["python"],
        "connection_profile": cfg["connection"],
        "client_exit_code": client_code,
        "client_error": verification["error"],
        "communication": by_category.get("communication"),
        "control_semantics": by_category.get("control_semantics"),
        "kinematic_mapping": {
            "joint_convention": ready["robot"]["joint_convention"],
            # The client program's targets were written in this convention.
            "program_convention_matches_server": verification["joint_convention"]
            == ready["robot"]["joint_convention"],
            "sdk_equals_mapped_sim_max_error_rad": score_mapping(trace, robot),
            "settled_pose_a": None
            if pose is None
            else {
                "sim_time": pose["t"],
                "sim_joint_q": pose["sim_joint_q"],
                "frames": pose["frames"],
            },
            "hardware_verified": False,
        },
        "kinematics": {
            "client_checks": by_category.get("kinematics"),
            **score_kinematics(trace, robot),
        },
        "physical_behaviour": {
            "client_checks": by_category.get("physical_behaviour"),
            "max_penetration_mm": result["max_penetration_mm"],
            "faults_before_fault_probe": faults_outside_probe,
            "realtime_factor": result["realtime_factor_achieved"],
            "physics_dt_s": result["physics_dt_s"],
        },
        "fault_handling": {
            "client_checks": by_category.get("fault_handling"),
            "probe": verification.get("fault_probe", {}).get("result"),
            "faults": result["faults"],
        },
        "task": {
            "sdk_side_checks": by_category.get("task_execution"),
            "ground_truth": score_task(trace, events)
            if "task_log" in verification
            else None,
        },
        "rpc_calls_by_outcome": outcomes,
        "camera_frames_served": result["camera_frames"],
        "video": video_info(out / "server/session.mp4"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def cross_backend(reports):
    poses = {
        n: r["kinematic_mapping"]["settled_pose_a"]
        for n, r in reports.items()
        if r.get("kinematic_mapping", {}).get("settled_pose_a")
    }
    names = sorted(poses)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            fa, fb = poses[a]["frames"], poses[b]["frames"]
            qa, qb = poses[a]["sim_joint_q"], poses[b]["sim_joint_q"]
            rows.append(
                {
                    "pair": [a, b],
                    "max_frame_difference_m": max(
                        float(np.linalg.norm(np.subtract(fa[k], fb[k])))
                        for k in fa
                        if k in fb
                    ),
                    "max_sim_joint_difference_rad": max(
                        abs(qa[j] - qb[j]) for j in qa if j in qb
                    ),
                }
            )
    return {
        "same_sdk_target": "pose_a from config/sdk_acceptance/program.json, settled",
        "pairs": rows,
        "meaning": "both engines received the same SDK joint target through the same "
        "mapping; differences combine tracking error and engine FK differences",
    }


def earlier_reports(out, requested):
    """Per-backend reports already in the run folder from earlier invocations."""
    return {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(out.glob("*/report.json"))
        if path.parent.name not in requested
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="mujoco,mujoco_owned_loop")
    parser.add_argument("--task", action="store_true")
    parser.add_argument("--fault-probe", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--rescore", action="store_true", help="score an existing --output again"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    # A run folder may grow: backends already in it are kept (and re-scored
    # with --rescore); a requested backend must not exist yet unless rescoring.
    out.mkdir(parents=True, exist_ok=True)
    before = sdk_integrity()
    reports = {}
    requested = args.backends.split(",")
    for name in requested:
        if (out / name).exists() and not args.rescore:
            raise SystemExit(f"{out / name} exists; pick another --output or --rescore")
    for name in requested:
        try:
            if args.rescore:
                cfg = yaml.safe_load(
                    (out / name / "config" / f"{name}.yaml").read_text()
                )
                code = json.loads((out / name / "client_exit.json").read_text())["code"]
                reports[name] = score(name, cfg, out / name, code)
            else:
                reports[name] = run_backend(name, out / name, args)
        except Exception as exc:  # noqa: BLE001 - one failure must not hide the rest
            reports[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(
            json.dumps(
                {
                    name: {
                        k: reports[name].get(k)
                        for k in (
                            "communication",
                            "control_semantics",
                            "client_error",
                            "error",
                        )
                    }
                }
            ),
            flush=True,
        )
    after = sdk_integrity()
    summary = {
        "acceptance_program": {
            "file": str(PROGRAM.relative_to(ROOT)),
            "sha256": sha256(PROGRAM),
        },
        "program_data_sha256": sha256(ROOT / "config/sdk_acceptance/program.json"),
        "robot_config_sha256": sha256(ROOT / "config/robot/franzi_sdk.yaml"),
        "vendor_sdk": {"before": before, "after": after},
        "backends": {**earlier_reports(out, requested), **reports},
        "cross_backend": cross_backend({**earlier_reports(out, requested), **reports}),
        "not_established": [
            "sim-to-real kinematic equivalence (no live controller readback)",
            "real servo/trajectory algorithms, gripper force, camera intrinsics",
            "Isaac Sim (draft adapter, never executed on this machine)",
        ],
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
