"""The thread that owns the simulator.

RPC worker threads never touch the backend. They submit closures through a
queue (``call``) or a render request (``render``) and wait for the physics
thread to run them between steps; state queries read the latest immutable
snapshot (``snapshot``), published once per physics step after the audit.
The loop runs on the process main thread, which keeps OpenGL/Metal contexts
and simulators that insist on the main thread (Isaac Sim, macOS GUI) happy.
"""

import json
import queue
import threading
import time

import numpy as np

from simbridge.backends.base import EFFORT_UNAVAILABLE, OwnedLoopHooks
from simbridge.control import Controller
from simbridge.recorder import SessionVideo
from simbridge.status import PhysicsFault


class _Request:
    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.cancelled = False
        self.result = None
        self.error = None


class Runtime:
    def __init__(self, robot, backend, options, output):
        self.robot = robot
        self.backend = backend
        self.options = options
        self.output = output
        self.controller = Controller(robot)
        self.commands = queue.Queue()
        self.renders = queue.Queue()
        self.stop_event = threading.Event()
        self.last_activity = time.monotonic()
        self.dt = backend.dt
        self.trace_every = max(1, round(1 / (robot.control["trace_hz"] * self.dt)))
        self.trace = []
        self.camera_log = []
        self.frame_seq = {name: 0 for name in robot.cameras}
        self.faults = []
        self.last_command = "start"
        self.wall_start = time.monotonic()
        self.loop_mode = options.get("loop", "framework")
        self.video = None
        self.record_every = int(options.get("record_every_samples", 1))
        if options.get("record"):
            # One frame per recorded sample, so playback runs at sim speed.
            self.video = SessionVideo(
                output / "session.mp4",
                f"{backend.label} | stock V1R1C9 SDK session",
                list(robot.cameras),
                fps=f"{robot.control['trace_hz']}/{self.record_every}",
            )
        self._snapshot = None
        self._snapshot_lock = threading.Lock()
        self._answered = []
        state = backend.read()
        measured = self._measure(state)
        self.controller.adopt(measured)
        self._publish(state, measured)

    # ----- called from RPC threads -------------------------------------------
    def touch(self, name):
        self.last_activity = time.monotonic()
        self.last_command = name

    def call(self, fn, timeout=None):
        """Run ``fn(controller, sim_time)`` between physics steps and return it."""
        request = _Request(fn)
        self.commands.put(request)
        if not request.done.wait(timeout or self.options.get("command_timeout_s", 2.0)):
            request.cancelled = True
            raise TimeoutError("simulation loop did not take the command (paused?)")
        if request.error:
            raise request.error
        return request.result

    def submit(self, fn):
        """Queue ``fn(controller, sim_time)`` for the physics thread; no reply."""
        request = _Request(fn)
        request.fire_and_forget = True
        self.commands.put(request)

    def render(self, camera, timeout):
        request = _Request(camera)
        self.renders.put(request)
        if not request.done.wait(timeout):
            request.cancelled = True
            raise TimeoutError("render request not served")
        if request.error:
            raise request.error
        return request.result

    def snapshot(self):
        with self._snapshot_lock:
            return self._snapshot

    # ----- physics thread ------------------------------------------------------
    def _drain_commands(self):
        while True:
            try:
                request = self.commands.get_nowait()
            except queue.Empty:
                return
            if request.cancelled:
                continue
            try:
                request.result = request.fn(self.controller, self._sim_time())
            except Exception as exc:  # noqa: BLE001 - returned to the RPC thread
                request.error = exc
            # Replied only after the next snapshot, so a client that polls
            # GetMotionStatus right after "accepted" already sees the motion.
            if not getattr(request, "fire_and_forget", False):
                self._answered.append(request)

    def _release(self):
        for request in self._answered:
            request.done.set()
        self._answered.clear()

    def _drain_renders(self):
        while True:
            try:
                request = self.renders.get_nowait()
            except queue.Empty:
                return
            if request.cancelled:
                continue
            try:
                spec = self.robot.cameras[request.fn]
                rgb = self.backend.render(spec.sim_name, spec.width, spec.height)
                self.frame_seq[request.fn] += 1
                stamp = time.time_ns() // 1_000_000
                snap = self.snapshot()
                entry = {
                    "camera": request.fn,
                    "seq": self.frame_seq[request.fn],
                    "utc_ms": stamp,
                    "sim_time": snap["sim_time"],
                    "step": snap["step"],
                }
                self.camera_log.append(entry)
                request.result = (rgb, entry)
            except Exception as exc:  # noqa: BLE001 - returned to the RPC thread
                request.error = exc
            request.done.set()

    def _sim_time(self):
        snap = self.snapshot()
        return snap["sim_time"] if snap else 0.0

    def pre_step(self):
        self._drain_commands()
        self.backend.apply(self.controller.tick(self.dt))

    def post_step(self):
        state = self.backend.read()
        measured = self._measure(state)
        self._publish(state, measured)
        self._release()
        if state.step % self.trace_every == 0:
            self._sample(state)
        self._drain_renders()

    def on_fault(self, exc):
        fault = {
            "kind": exc.kind,
            "message": str(exc),
            "sim_time": self._sim_time(),
            "utc_ms": time.time_ns() // 1_000_000,
            **exc.details,
        }
        self.faults.append(fault)
        self.controller.latch_fault(fault, fault["sim_time"])
        print(
            f"PHYSICS FAULT at {fault['sim_time']:.3f}s: {fault['message']}", flush=True
        )

    def halted(self):
        return self.controller.fault is not None

    def idle(self):
        """Physics frozen after a fault: keep answering commands and renders."""
        self._drain_commands()
        snap = dict(self.snapshot())
        snap["physics_running"] = False
        snap["estop"] = self.controller.estop
        snap["fault"] = self.controller.fault
        with self._snapshot_lock:
            self._snapshot = snap
        self._release()
        self._drain_renders()
        time.sleep(0.002)

    def should_stop(self):
        idle = time.monotonic() - self.last_activity
        return self.stop_event.is_set() or idle > self.options.get("idle_timeout_s", 60)

    def run(self):
        if self.loop_mode == "simulator":
            hooks = OwnedLoopHooks(self.pre_step, self.post_step, self.should_stop)
            hooks.on_fault = self.on_fault
            hooks.halted = self.halted
            hooks.idle = self.idle
            self.backend.run_owned_loop(hooks)
            return
        rtf = float(self.options.get("realtime_factor", 1.0))
        tick = time.monotonic()
        while not self.should_stop():
            if self.halted():
                self.idle()
                continue
            self.pre_step()
            try:
                self.backend.step()
            except PhysicsFault as exc:
                self.on_fault(exc)
            self.post_step()
            if rtf > 0:
                tick += self.dt / rtf
                delay = tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    tick = time.monotonic()

    # ----- measurement and snapshots ------------------------------------------
    def _measure(self, state):
        units = {}
        for name, unit in self.robot.units.items():
            q = unit.to_sdk([state.q[j] for j in unit.sim_joints])
            dq = unit.rate_to_sdk([state.dq[j] for j in unit.sim_joints])
            effort = None
            if state.effort_kind != EFFORT_UNAVAILABLE:
                effort = unit.rate_to_sdk([state.effort[j] for j in unit.sim_joints])
            units[name] = {"q": q, "dq": dq, "effort": effort}
        grippers = {}
        for side, g in self.robot.grippers.items():
            q = [state.q[j] for j in g.sim_joints]
            dq = [state.dq[j] for j in g.sim_joints]
            jaws = set()
            for c in state.contacts:
                for mine, other, other_robot in (
                    (c["a"], c["b"], c["b_robot"]),
                    (c["b"], c["a"], c["a_robot"]),
                ):
                    if mine in g.finger_links and not other_robot:
                        jaws.add(mine)
            grippers[side] = {
                "closure": g.closure(q),
                "closure_rate": g.closure(dq),
                "external_contact_jaws": len(jaws),
            }
        return {"units": units, "grippers": grippers}

    def _publish(self, state, measured):
        ctl = self.controller
        now = state.sim_time
        arms = {}
        for name, arm in ctl.arms.items():
            arms[name] = {
                "setpoint": arm.setpoint.tolist(),
                "velocity": arm.velocity.tolist(),
                "final": arm.final.tolist(),
                "mode": arm.mode,
                "enabled": arm.enabled,
                "ratio": arm.ratio,
                "teleop_queue": len(arm.stream.points),
                "waypoints": len(arm.waypoints) + (arm.segment is not None),
                "motion_status": ctl.motion_status(name, measured, now),
            }
        grippers = {}
        for side, g in self.robot.grippers.items():
            m = measured["grippers"][side]
            grippers[side] = {
                "position": g.position_from_closure(m["closure"]),
                "target": g.position_from_closure(ctl.grip_target[side]),
                "closure": m["closure"],
                "grip_state": ctl.grip_state(side, measured),
                "external_contact_jaws": m["external_contact_jaws"],
            }
        snap = {
            "sim_time": now,
            "step": state.step,
            "wall_time": time.time(),
            "physics_running": ctl.fault is None,
            "units": {
                n: {
                    "q": u["q"].tolist(),
                    "dq": u["dq"].tolist(),
                    "effort": None if u["effort"] is None else u["effort"].tolist(),
                }
                for n, u in measured["units"].items()
            },
            "effort_kind": state.effort_kind,
            "arms": arms,
            "grippers": grippers,
            "estop": ctl.estop,
            "fault": ctl.fault,
            "max_penetration_mm": state.max_penetration_m * 1000,
        }
        with self._snapshot_lock:
            self._snapshot = snap

    def _sample(self, state):
        snap = self.snapshot()
        sample = {
            "t": snap["sim_time"],
            "wall_s": time.monotonic() - self.wall_start,
            "last_command": self.last_command,
            "state": {n: u["q"] for n, u in snap["units"].items()},
            "velocity": {n: snap["units"][n]["dq"] for n in self.robot.arms},
            "effort": {n: snap["units"][n]["effort"] for n in self.robot.arms},
            "setpoint": {n: a["setpoint"] for n, a in snap["arms"].items()},
            "mode": {n: a["mode"] for n, a in snap["arms"].items()},
            "motion_status": {n: a["motion_status"] for n, a in snap["arms"].items()},
            "gripper": snap["grippers"],
            "estop": snap["estop"],
            "fault": snap["fault"] is not None,
            "max_penetration_mm": snap["max_penetration_mm"],
            "sim_joint_q": {
                j: float(state.q[j]) for j in self.robot.controlled_sim_joints()
            },
            "frames": self.backend.tracked_frames(),
            "objects": self.backend.task_state(),
        }
        self.trace.append(sample)
        if self.video and (len(self.trace) - 1) % self.record_every == 0:
            self._record(snap)

    def _record(self, snap):
        size = self.options.get("record_inset_size", [320, 240])
        overview = self.backend.render(self.options["display_camera"], 960, 720)
        insets = {
            name: self.backend.render(spec.sim_name, *size)
            for name, spec in self.robot.cameras.items()
        }
        modes = " ".join(f"{n[0].upper()}:{a['mode']}" for n, a in snap["arms"].items())
        flags = "SOFT STOP " if snap["estop"] else ""
        flags += "FAULT " if snap["fault"] else ""
        self.video.frame(
            overview,
            insets,
            f"sim {snap['sim_time']:6.2f}s | {modes} | last RPC {self.last_command} "
            f"| {flags}peak depth {snap['max_penetration_mm']:.3f} mm",
        )

    def result(self):
        wall = time.monotonic() - self.wall_start
        sim = self.snapshot()["sim_time"]
        running = self.trace[-1]["wall_s"] if self.trace else wall
        return {
            "backend": self.backend.name,
            "loop_mode": self.loop_mode,
            "sim_time_s": sim,
            "wall_time_s": wall,
            # Over the stepping period only; after a fault physics is frozen.
            "realtime_factor_achieved": self.trace[-1]["t"] / running
            if running
            else None,
            "physics_dt_s": self.dt,
            "trace_hz": self.robot.control["trace_hz"],
            "faults": self.faults,
            "max_penetration_mm": self.snapshot()["max_penetration_mm"],
            "controller_events": self.controller.events,
            "camera_frames": len(self.camera_log),
            "video_frames": self.video.video.frames if self.video else 0,
        }

    def close(self):
        if self.video:
            self.video.close()
        (self.output / "trace.json").write_text(json.dumps(self.trace))
        (self.output / "camera_log.json").write_text(
            json.dumps(self.camera_log, indent=1)
        )
        if self.trace:
            np.savez_compressed(
                self.output / "trace_arrays.npz",
                t=[s["t"] for s in self.trace],
                left=[s["state"]["left"] for s in self.trace],
                right=[s["state"]["right"] for s in self.trace],
                left_setpoint=[s["setpoint"]["left"] for s in self.trace],
                right_setpoint=[s["setpoint"]["right"] for s in self.trace],
            )
