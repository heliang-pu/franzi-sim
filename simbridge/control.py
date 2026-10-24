"""Engine-independent control semantics in SDK joint coordinates.

Runs only on the thread that owns the simulator (see runtime.py), so its state
needs no locks. Every command returns an SDK status; 0 means *accepted*, never
*arrived*. Arrival is reported by ``motion_status`` (GetMotionStatus).

Simulation policies that the vendor documents do not pin down are marked
"policy" below and listed in the README.
"""

from collections import deque

import numpy as np

from simbridge.status import Sdk

IDLE, TRAJECTORY, TELEOP, BRAKING = "idle", "trajectory", "teleop", "braking"


class Segment:
    """Synchronised trapezoidal move: every joint starts and stops together.

    The normalised profile s(t) in [0, 1] uses the tightest per-joint ratio of
    limit to distance, so no joint exceeds its own velocity or acceleration.
    """

    def __init__(self, start, goal, vmax, amax):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        dist = np.abs(self.goal - self.start)
        moving = dist > 1e-12
        if not moving.any():
            self.duration = 0.0
            return
        v = float(np.min(vmax[moving] / dist[moving]))
        a = float(np.min(amax[moving] / dist[moving]))
        if v * v / a <= 1.0:
            self.t_acc = v / a
            self.duration = 1.0 / v + self.t_acc
            self.v_peak = v
        else:
            self.t_acc = np.sqrt(1.0 / a)
            self.duration = 2 * self.t_acc
            self.v_peak = a * self.t_acc
        self.a = a

    def sample(self, t):
        """Position and velocity at time t after the segment started."""
        if self.duration == 0.0 or t >= self.duration:
            return self.goal.copy(), np.zeros_like(self.goal), True
        a, ta, vp, T = self.a, self.t_acc, self.v_peak, self.duration
        if t < ta:
            s, ds = 0.5 * a * t * t, a * t
        elif t <= T - ta:
            s, ds = 0.5 * a * ta * ta + vp * (t - ta), vp
        else:
            r = T - t
            s, ds = 1.0 - 0.5 * a * r * r, a * r
        delta = self.goal - self.start
        return self.start + delta * s, delta * ds, False


class Stream:
    """Teleop playback at the client's own pacing (policy).

    Each pushed point is played over the stream's period: the median wall-clock
    gap between pushes over the session's last 50 gaps, the default period until
    8 are known. Gaps longer than the pause limit are pauses, not pacing: they
    are left out and do not clear the history. The client's timing jitter is
    therefore not replayed as velocity jitter, nor are catch-up bursts: a slow
    reply stalls the client, whose absolute schedule then sends the missed
    points back to back. A median of the last 8 gaps, cleared at every pause,
    collapsed in those bursts (PyBullet's steps hold the GIL, so they are
    common there) and drove the playback into the velocity clamps. Playback
    interpolates in simulated time, one point behind the newest, so a steady
    stream moves steadily also when the simulator runs slower than real time.
    """

    WINDOW = 50  # gaps in the median: two seconds of a 25 Hz stream
    WARMUP = 8  # gaps needed before the median replaces the default period

    def __init__(self, start, control):
        self.points = deque()
        self.reference = np.array(start, float)
        self.velocity = np.zeros_like(self.reference)
        self.active = None
        self.elapsed = 0.0
        self.waited = 0.0
        self.last_stamp = None
        self.gaps = deque(maxlen=self.WINDOW)
        self.default = float(control["teleop_default_period_s"])
        self.max_gap = float(control["teleop_max_gap_s"])

    def push(self, q, stamp):
        if (
            self.last_stamp is not None
            and 0.0 <= stamp - self.last_stamp <= self.max_gap
        ):
            self.gaps.append(stamp - self.last_stamp)
        self.last_stamp = stamp
        period = self.default
        if len(self.gaps) >= self.WARMUP:
            period = float(np.median(self.gaps))
        self.points.append((q, max(period, 1e-4)))

    def busy(self):
        return bool(self.points) or self.active is not None

    def _start(self, carry):
        q, duration = self.points.popleft()
        self.active = (self.reference.copy(), q, duration)
        self.elapsed = carry
        self.waited = 0.0

    def advance(self, dt):
        """Reference position and velocity after dt of simulated time."""
        if self.active is None and self.points:
            self.waited += dt
            if len(self.points) >= 2 or self.waited >= 2 * self.points[0][1]:
                self._start(0.0)
                self.elapsed = dt
        elif self.active is not None:
            self.elapsed += dt
        while self.active is not None and self.elapsed >= self.active[2]:
            carry = self.elapsed - self.active[2]
            self.reference = self.active[1].copy()
            self.active = None
            if self.points:
                self._start(carry)
        if self.active is None:
            self.velocity = np.zeros_like(self.reference)
        else:
            start, goal, duration = self.active
            self.reference = start + (goal - start) * (self.elapsed / duration)
            self.velocity = (goal - start) / duration
        return self.reference, self.velocity


class Arm:
    def __init__(self, unit, control):
        self.unit = unit
        self.enabled = True
        self.mode = IDLE
        self.ratio = 100.0
        self.setpoint = np.zeros(len(unit.sim_joints))
        self.velocity = np.zeros_like(self.setpoint)
        self.final = self.setpoint.copy()
        self.control = control
        self.waypoints = deque()
        self.stream = Stream(self.setpoint, control)
        self.track_gain = float(control["teleop_track_gain_per_s"])
        self.segment = None
        self.segment_time = 0.0
        self.capacity = int(control["teleop_queue_capacity"])
        self.stalled_since = None
        self.motion_started = None

    @property
    def vmax(self):
        return self.unit.vmax * self.ratio / 100

    @property
    def amax(self):
        return self.unit.amax * self.ratio / 100

    def pending(self):
        return bool(
            self.segment or self.waypoints or self.stream.busy() or self.mode == BRAKING
        )

    def restart_stream(self):
        self.stream = Stream(self.setpoint, self.control)

    def hold(self):
        """Policy: stop with the acceleration limit, then hold the setpoint."""
        self.waypoints.clear()
        self.restart_stream()
        self.segment = None
        self.mode = BRAKING if np.any(self.velocity) else IDLE
        self.final = self.setpoint.copy()

    def advance(self, dt):
        if self.mode == BRAKING:
            dv = np.minimum(np.abs(self.velocity), self.amax * dt)
            self.velocity -= np.sign(self.velocity) * dv
            self.setpoint += self.velocity * dt
            self.final = self.setpoint.copy()
            if not np.any(self.velocity):
                self.mode = IDLE
            return
        if self.mode == TRAJECTORY:
            if self.segment is None and self.waypoints:
                self.segment = Segment(
                    self.setpoint, self.waypoints.popleft(), self.vmax, self.amax
                )
                self.segment_time = 0.0
            if self.segment is not None:
                self.segment_time += dt
                q, dq, done = self.segment.sample(self.segment_time)
                self.setpoint, self.velocity = q, dq
                if done:
                    self.segment = None
            if self.segment is None and not self.waypoints:
                self.mode = IDLE
            return
        if self.mode == TELEOP:
            reference, feedforward = self.stream.advance(dt)
            # Safety clamp around the playback: follow the stream's velocity,
            # remove any remaining error without overshooting, and never exceed
            # the joint velocity and acceleration limits.
            error = reference - self.setpoint
            correction = np.sign(error) * np.minimum(
                self.track_gain * np.abs(error), np.sqrt(2 * self.amax * np.abs(error))
            )
            desired = np.clip(feedforward + correction, -self.vmax, self.vmax)
            self.velocity += np.clip(
                desired - self.velocity, -self.amax * dt, self.amax * dt
            )
            self.setpoint = self.setpoint + self.velocity * dt
            settled = (
                np.max(np.abs(error)) < 1e-6 and np.max(np.abs(self.velocity)) < 1e-3
            )
            if settled and not self.stream.busy():
                self.setpoint = reference.copy()
                self.velocity = np.zeros_like(reference)


def check_target(robot, unit, values):
    """(joint vector, None) or (None, reason): shape, finiteness, SDK limits."""
    q = np.asarray(values, float)
    lim = robot.arms[unit].sdk_limits
    if q.shape != (len(lim),) or not np.all(np.isfinite(q)):
        return None, f"{unit}: expected {len(lim)} finite joint angles (rad)"
    bad = np.flatnonzero((q < lim[:, 0]) | (q > lim[:, 1]))
    if bad.size:
        j = int(bad[0])
        policy = robot.raw["limits"]["position_policy"]
        return None, (
            f"{unit} j{j + 1} target {q[j]:.4f} rad outside "
            f"[{lim[j, 0]:.4f}, {lim[j, 1]:.4f}] ({policy})"
        )
    return q, None


def precheck_push(robot, points, snapshot):
    """Teleop point checks an RPC thread can run against the latest snapshot.

    Streaming replies must not wait for the physics thread: when the simulator
    runs slower than real time that wait would distort the push pacing that
    playback follows. tele_push repeats every check on the physics thread and
    drops (and logs) a point that a state change made invalid meanwhile.
    """
    if snapshot["fault"]:
        return {"status": int(Sdk.UNIT_FAULT), "err_msg": "physics fault"}
    if snapshot["estop"]:
        return {"status": int(Sdk.NOT_READY), "err_msg": "soft stop active"}
    capacity = int(robot.control["teleop_queue_capacity"])
    for unit, values in points.items():
        arm = snapshot["arms"][unit]
        if not arm["enabled"]:
            return {"status": int(Sdk.NOT_READY), "err_msg": f"unit disabled: {unit}"}
        if arm["mode"] != TELEOP:
            return {
                "status": int(Sdk.MODE_CONFLICT),
                "err_msg": "no teleop session (call TeleJoint first)",
            }
        _, err = check_target(robot, unit, values)
        if err:
            return {"status": int(Sdk.PARAM_INVALID), "err_msg": err}
        if arm["teleop_queue"] >= capacity:
            return {
                "status": int(Sdk.PARAM_INVALID),
                "err_msg": f"{unit} teleop queue full",
            }
    return None


class Controller:
    def __init__(self, robot):
        self.robot = robot
        self.control = robot.control
        self.arms = {name: Arm(unit, self.control) for name, unit in robot.arms.items()}
        self.grip_target = {s: 0.0 for s in robot.grippers}
        self.grip_setpoint = {s: 0.0 for s in robot.grippers}
        self.grip_history = {
            s: {"held": None, "dropped": False} for s in robot.grippers
        }
        self.estop = False
        self.fault = None
        self.events = []

    # ----- initial state -------------------------------------------------
    def adopt(self, measured):
        """Start from the measured pose so the first command does not jump."""
        for name, arm in self.arms.items():
            arm.setpoint = measured["units"][name]["q"].copy()
            arm.final = arm.setpoint.copy()
        for side in self.grip_setpoint:
            closure = measured["grippers"][side]["closure"]
            self.grip_setpoint[side] = self.grip_target[side] = closure

    # ----- helpers -------------------------------------------------------
    @staticmethod
    def _reject(code, message):
        return {"status": int(code), "err_msg": message}

    def _gate(self, units, now):
        if self.fault:
            return self._reject(
                Sdk.UNIT_FAULT, f"physics fault: {self.fault['message']}"
            )
        if self.estop:
            return self._reject(Sdk.NOT_READY, "soft stop active")
        disabled = [u for u in units if not self.arms[u].enabled]
        if disabled:
            return self._reject(Sdk.NOT_READY, f"unit disabled: {disabled}")
        return None

    def _target(self, unit, values):
        return check_target(self.robot, unit, values)

    def _log(self, now, kind, **data):
        self.events.append({"sim_time": now, "event": kind, **data})

    # ----- commands (all return SDK status payloads) ---------------------
    def move_j(self, units, path, now, home=False):
        gate = self._gate(units, now)
        if gate:
            return gate
        if any(self.arms[u].mode == TELEOP for u in units):
            return self._reject(Sdk.MODE_CONFLICT, "stop the teleop session first")
        if not path:
            return self._reject(Sdk.PARAM_INVALID, "empty joint_path")
        prepared = {u: [] for u in units}
        width = 7 * len(units)
        for point in path:
            if len(point) != width:
                return self._reject(
                    Sdk.PARAM_INVALID, f"each point needs {width} values"
                )
            for i, u in enumerate(units):
                q, err = self._target(u, point[7 * i : 7 * (i + 1)])
                if err:
                    return self._reject(Sdk.PARAM_INVALID, err)
                prepared[u].append(q)
        for u in units:
            arm = self.arms[u]
            # Policy: a MoveJ during another MoveJ queues behind it.
            arm.waypoints.extend(prepared[u])
            arm.final = prepared[u][-1].copy()
            arm.mode = TRAJECTORY
            arm.motion_started = now
            arm.stalled_since = None
        self._log(now, "Home" if home else "MoveJ", units=list(units), points=len(path))
        return {"status": 0}

    def tele_start(self, units, now):
        gate = self._gate(units, now)
        if gate:
            return gate
        if any(self.arms[u].mode in (TRAJECTORY, BRAKING) for u in units):
            return self._reject(Sdk.MODE_CONFLICT, "trajectory still executing")
        for u in units:
            arm = self.arms[u]
            arm.mode = TELEOP
            arm.final = arm.setpoint.copy()
            arm.restart_stream()
        self._log(now, "TeleStart", units=list(units))
        return {"status": 0}

    def tele_stop(self, units, now):
        for u in units:
            if self.arms[u].mode == TELEOP:
                self.arms[u].hold()
        self._log(now, "TeleStop", units=list(units))
        return {"status": 0}

    def tele_push(self, points, now, stamp, replied=False):
        units = list(points)
        gate = self._gate(units, now)
        if gate:
            if replied:
                self._log(now, "TeleopPointDropped", reason=gate["err_msg"])
            return gate
        reason = None
        if any(self.arms[u].mode != TELEOP for u in units):
            reason = (Sdk.MODE_CONFLICT, "no teleop session (call TeleJoint first)")
        prepared = {}
        for u in units if reason is None else ():
            q, err = self._target(u, points[u])
            if err:
                reason = (Sdk.PARAM_INVALID, err)
                break
            if len(self.arms[u].stream.points) >= self.arms[u].capacity:
                reason = (Sdk.PARAM_INVALID, f"{u} teleop queue full")
                break
            prepared[u] = q
        if reason:
            if replied:
                self._log(now, "TeleopPointDropped", reason=reason[1])
            return self._reject(*reason)
        for u, q in prepared.items():
            arm = self.arms[u]
            arm.stream.push(q, stamp)
            arm.final = q.copy()
            arm.motion_started = now
            arm.stalled_since = None
        return {"status": 0}

    def lifecycle(self, units, cmd, now):
        if cmd not in (1, 2, 3):
            return self._reject(Sdk.PARAM_INVALID, f"lifecycle_cmd {cmd} not in 1/2/3")
        if self.fault:
            return self._reject(
                Sdk.UNIT_FAULT,
                "physics fault is terminal for this simulation session; restart it",
            )
        if cmd == 1 and self.estop:
            return self._reject(Sdk.ESTOP_ACTIVE, "release the soft stop first")
        for u in units:
            arm = self.arms[u]
            if cmd == 1:
                arm.enabled = True
            else:
                # Policy: Disable freezes commands and keeps the simulated servo
                # holding; it does not model brakes or a power-off drop.
                arm.hold()
                if cmd == 2:
                    arm.enabled = False
        self._log(now, "Lifecycle", units=list(units), cmd=cmd)
        return {"status": 0}

    def soft_stop(self, active, now):
        self.estop = bool(active)
        if self.estop:
            for arm in self.arms.values():
                arm.hold()
            for side in self.grip_target:
                self.grip_target[side] = self.grip_setpoint[side]
        self._log(now, "SoftStop", active=self.estop)
        return {"status": 0}

    def set_ratio(self, units, ratios, now):
        if len(ratios) != len(units) or not all(1 <= float(r) <= 100 for r in ratios):
            return self._reject(Sdk.PARAM_INVALID, "one ratio per unit, each in 1..100")
        for u, r in zip(units, ratios):
            self.arms[u].ratio = float(r)
        return {"status": 0}

    def gripper(self, side, position, now):
        if self.fault:
            return self._reject(
                Sdk.UNIT_FAULT, f"physics fault: {self.fault['message']}"
            )
        if self.estop:
            return self._reject(Sdk.NOT_READY, "soft stop active")
        if not self.arms[side].enabled:
            return self._reject(Sdk.NOT_READY, f"unit disabled: {side}")
        if not 0 <= position <= 1000:
            return self._reject(Sdk.PARAM_INVALID, "gripper position must be 0..1000")
        self.grip_target[side] = self.robot.grippers[side].closure_from_position(
            position
        )
        self.grip_history[side] = {"held": None, "dropped": False}
        self._log(now, "Gripper", side=side, position=int(position))
        return {"status": 0}

    def latch_fault(self, fault, now):
        self.fault = fault
        self.estop = True
        for arm in self.arms.values():
            arm.waypoints.clear()
            arm.restart_stream()
            arm.segment = None
            arm.velocity[:] = 0
            arm.mode = IDLE
        self._log(now, "Fault", fault=fault)

    # ----- per control tick ----------------------------------------------
    def tick(self, dt):
        """Advance setpoints by dt and return simulator joint targets by name."""
        targets = {}
        if self.fault is None:
            for arm in self.arms.values():
                arm.advance(dt)
            for side, gripper in self.robot.grippers.items():
                if not self.estop:
                    diff = self.grip_target[side] - self.grip_setpoint[side]
                    step = gripper.max_closure_rate * dt
                    self.grip_setpoint[side] += float(np.clip(diff, -step, step))
        for arm in self.arms.values():
            targets.update(zip(arm.unit.sim_joints, arm.unit.to_sim(arm.setpoint)))
        for side, gripper in self.robot.grippers.items():
            targets.update(gripper.finger_targets(self.grip_setpoint[side]))
        return targets

    # ----- status derived from measurements --------------------------------
    def motion_status(self, unit, measured, now):
        """GetMotionStatus codes: 0 still at target, 1 moving or converging,
        -1 fault, -3 stopped short of the target for longer than the timeout."""
        if self.fault:
            return -1
        arm = self.arms[unit]
        q = measured["units"][unit]["q"]
        dq = measured["units"][unit]["dq"]
        tol = float(self.control["arrival_tolerance_rad"])
        still = float(self.control["still_velocity_rad_s"])
        error = float(np.max(np.abs(q - arm.final)))
        if arm.pending() or np.max(np.abs(arm.setpoint - arm.final)) > 1e-9:
            arm.stalled_since = None
            return 1
        if error <= tol and np.max(np.abs(dq)) <= still:
            arm.stalled_since = None
            return 0
        if np.max(np.abs(dq)) <= still:
            if arm.stalled_since is None:
                arm.stalled_since = now
            if now - arm.stalled_since >= float(self.control["stall_timeout_s"]):
                return -3
        else:
            arm.stalled_since = None
        return 1

    def grip_state(self, side, measured):
        """EeGripperReadGripState: 0 moving, 1 arrived, 2 holding, 3 dropped.

        Holding: the fingers stopped short of a more-closed target while both
        jaws touch something that is not the robot (backend contact query).
        Dropped: after holding, the fingers closed further, i.e. the object
        left the jaws; latched until the next gripper command.
        """
        g = measured["grippers"][side]
        history = self.grip_history[side]
        target = self.grip_target[side]
        moving = (
            abs(g["closure_rate"]) > 0.02
            or abs(self.grip_setpoint[side] - target) > 1e-9
        )
        if history["dropped"]:
            return 3
        if moving:
            return 0
        if g["external_contact_jaws"] >= 2 and target - g["closure"] > 0.005:
            if history["held"] is None:
                history["held"] = g["closure"]
            return 2
        if history["held"] is not None and g["closure"] > history["held"] + 0.01:
            history["dropped"] = True
            return 3
        return 1
