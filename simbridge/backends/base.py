"""Simulator adapter interface.

An adapter speaks simulator joint names and SI units only; the SDK mapping is
done above it. All methods are called from one thread (the runtime's physics
thread, or the simulator's own loop through the hooks), never from RPC workers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Kinds of joint effort an adapter can report. ReadJointState labels its torque
# with one of these so simulated, measured and missing data never look alike.
EFFORT_SIMULATED = "simulated_actuator_effort"
EFFORT_MEASURED = "measured_joint_torque_sensor"
EFFORT_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Capabilities:
    """What an adapter can do. Anything False/empty must fail explicitly."""

    joint_position_targets: bool
    joint_velocity_feedback: bool
    joint_effort: str
    rgb_cameras: tuple
    contact_query: bool
    penetration_audit: bool
    loop_modes: tuple
    render_thread: str = "physics thread only"
    depth_cameras: tuple = ()
    wrist_force_torque: bool = False
    notes: tuple = field(default=())

    def as_dict(self):
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()
        }


@dataclass
class BackendState:
    """One consistent read after a physics step, in simulator joint space."""

    sim_time: float
    step: int
    q: dict
    dq: dict
    effort: dict
    effort_kind: str
    contacts: list
    max_penetration_m: float
    extra: dict = field(default_factory=dict)


class OwnedLoopHooks:
    """Callbacks a simulator-owned loop calls around each physics step."""

    def __init__(self, pre_step, post_step, should_stop):
        self.pre_step = pre_step
        self.post_step = post_step
        self.should_stop = should_stop


class SimBackend(ABC):
    name = "abstract"
    implementation_status = "abstract"
    verification_status = "none"

    @abstractmethod
    def load(self, robot, options, output):
        """Build the scene from the robot config; called once on the physics thread."""

    @abstractmethod
    def capabilities(self):
        """Return Capabilities."""

    @property
    @abstractmethod
    def dt(self):
        """Physics step in seconds."""

    @abstractmethod
    def reset(self):
        """Back to the initial scene state (only before the session starts)."""

    @abstractmethod
    def apply(self, targets):
        """Hold these joint position targets (sim joint name -> rad or m)."""

    @abstractmethod
    def step(self):
        """Advance one physics step; raise simbridge.status.PhysicsFault on audit failure."""

    @abstractmethod
    def read(self):
        """Return BackendState for the current step."""

    @abstractmethod
    def render(self, camera, width, height):
        """RGB uint8 (height, width, 3) from a camera named in the robot config."""

    def run_owned_loop(self, hooks):
        """Simulator-owned stepping. Adapters that support it override this."""
        raise NotImplementedError(f"{self.name} has no simulator-owned loop")

    def task_state(self):
        """Ground-truth task objects for scoring only, never sent through the SDK."""
        return {}

    def tracked_frames(self):
        """World poses of named frames for cross-backend kinematic comparison."""
        return {}

    @abstractmethod
    def close(self):
        """Release simulator resources."""
