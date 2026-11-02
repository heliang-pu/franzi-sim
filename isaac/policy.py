"""Run a learned policy closed-loop on Isaac Sim: one episode of a benchmark
scene, from the default posture until the scene's success test holds or a
time limit.

    ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh isaac.policy <scene> --output outputs/<new_run> --seed 10000 [--server 127.0.0.1:18780] [--limit 90]
    ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh isaac.policy <scene> --output outputs/<new_run> --seed 0 --replay outputs/<run>/trajectory.npz

The scene's own run.run() runs with its expert replaced (this process wraps
sim.tabletop.run_episode), so the layout, the success test and the run files
are those of every expert episode:
- the episode starts as every expert's does, in tabletop.start_home before the
  recording;
- at each control tick (25 Hz) the policy is shown the four policy cameras'
  frames of that tick, the very frames the run's videos record (one render per
  tick), and the measured 16-D state in the datasets' SDK units (joints, then
  gripper positions 0 closed .. 1000 open, rounded as convert_expert_run
  rounds them). Its chunk's first --horizon actions are applied one per tick,
  as the arms' joint targets and the grippers' closures;
- the subtask is generated again every --subtask-every ticks, and between
  those it conditions the actions unchanged. Each query also carries the
  previous query's subtask (None at first), which a checkpoint with the memory
  recipe reads (fineart.recipe);
- it ends when the scene's complete() has held for 15 ticks, or at --limit
  seconds of recording. Success then means what it means for the expert: no
  error (penetration over 1 mm, joint limits), complete() held, the grasp
  evidence and no audited collision. The carried-object check (CARRY_PHASES)
  is keyed to the expert's phase names, so it doesn't apply here.
--lenient logs penetrations and limit errors instead of ending the episode
(Scene.stop_on_violation), so the summary's "completed" (the task done,
whatever was touched) can be counted too; "success" means the same either way.
The termination is success, completed (the task done, with collisions or
violations), error, void (the simulation broke down, before any violation),
time_limit or incomplete.
--replay serves a recorded run's own commands instead of a policy: the check
of this harness, since a replay of an expert run of the same seed must succeed
again.
"""

import argparse
import importlib
import json
import traceback
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import yaml

from fineart.protocol import AUTHKEY, PORT, SUBTASK_EVERY, address, pack, unpack
from isaac import runtime

ROOT = Path(__file__).resolve().parents[1]
STREAMS = yaml.safe_load((ROOT / "config/robot/franzi_sdk.yaml").read_text())[
    "cameras"
]["streams"]
CAMERAS = ("head", "waist", "wrist_left", "wrist_right")  # the datasets' names
SIDES = ("left", "right")


def to_sdk(kin, state):
    """A 16-D state (URDF joints, closures) in the datasets' SDK units: the
    grippers as simbridge.robot.GripperMap.position_from_closure."""
    grips = [float(np.clip(round(1000 * (1 - c)), 0, 1000)) for c in state[14:]]
    return np.r_[kin.to_sdk("left", state[:7]), kin.to_sdk("right", state[7:14]), grips]


def apply(s, action):
    """Sets one tick's targets from a 16-D action in SDK units."""
    action = np.asarray(action, float)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid policy action {action.tolist()}")
    s.goal["left"] = s.kin.from_sdk("left", action[:7])
    s.goal["right"] = s.kin.from_sdk("right", action[7:14])
    for side, position in zip(SIDES, action[14:], strict=True):
        s.grip[side] = float(np.clip(1 - position / 1000, 0, 1))


class Remote:
    """A policy behind fineart.serve."""

    def __init__(self, server, seed):
        self.connection = Client(address(server), authkey=AUTHKEY)
        self.seed = seed
        self.calls = []

    def __call__(self, s, tick, state, frames, subtask, memory):
        request = {
            "op": "infer",
            "task": s.spec["instruction"],
            "state": pack(state.astype(np.float32)),
            "images": {c: pack(frames[STREAMS[c]["sim"]]) for c in CAMERAS},
            "subtask": subtask,
            "memory": memory,
            "seed": self.seed * 100000 + tick,
        }
        self.connection.send(request)
        reply = self.connection.recv()
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        self.calls.append(
            {
                "tick": tick,
                "subtask": reply["subtask"],
                "generated": reply["generated"],
                "memory": memory if reply["generated"] else None,
                "seconds": reply["seconds"],
            }
        )
        return unpack(reply["actions"]), reply["subtask"]


class Replay:
    """A recorded run's commands, converted to SDK units and back as the
    datasets' are."""

    def __init__(self, trajectory, kin):
        with np.load(trajectory) as data:
            self.actions = np.array([to_sdk(kin, a) for a in data["action"]])
        self.calls = []

    def __call__(self, s, tick, state, frames, subtask, memory):
        ahead = np.minimum(np.arange(tick, tick + 50), len(self.actions) - 1)
        self.calls.append({"tick": tick})
        return self.actions[ahead], "replay"


def drive(s, policy, limit, horizon, subtask_every):
    """The policy loop in place of a scene's expert."""
    from sim import tabletop as tt

    tt.start_home(s)
    s.stage("policy", "")
    queue, subtask, memory, asked, tick = [], None, None, None, 0
    while s.stable_frames < 15 and s.clock() < limit:
        if not queue:
            frames = s.capture()
            state = to_sdk(s.kin, s.state())
            if asked is None or tick - asked >= subtask_every:
                memory, subtask, asked = subtask, None, tick
            actions, subtask = policy(s, tick, state, frames, subtask, memory)
            queue = list(actions[:horizon])
            s.instruction = subtask
        apply(s, queue.pop(0))
        for _ in range(20):
            s.physics_step()
        tick += 1


class Frames:
    """One render per tick: the policy's observation and the recording share
    the frames captured at a physics step."""

    captured_at = None

    def capture(self):
        if self.captured_at != self.step_number:
            self.captured = super().capture()
            self.captured_at = self.step_number
        return self.captured


def main():
    parser = argparse.ArgumentParser(
        description="Run a policy on a scene in Isaac Sim."
    )
    parser.add_argument("scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--server", default=str(PORT), help="host:port of fineart.serve"
    )
    parser.add_argument("--replay", type=Path, help="a run's trajectory.npz instead")
    parser.add_argument("--limit", type=float, default=120.0, help="recorded seconds")
    parser.add_argument(
        "--horizon", type=int, default=25, help="actions applied per chunk"
    )
    parser.add_argument(
        "--subtask-every", type=int, default=SUBTASK_EVERY, help="ticks"
    )
    parser.add_argument(
        "--lenient", action="store_true", help="log violations instead of stopping"
    )
    parser.add_argument(
        "--display", action="store_true", help="also expert.mp4 and cameras.mp4"
    )
    args = parser.parse_args()
    app = runtime.start_app()
    try:
        from isaac import engine
        from isaac.engine import isaac_scene
        from sim import tabletop as tt

        engine.DISPLAY = args.display
        module = importlib.import_module(f"scenes.{args.scene}.run")
        run_episode = tt.run_episode
        used = {}

        def policy_episode(
            scene_cls, expert, spec, output, record=False, extra=None, engine=None
        ):
            def run(s):
                if args.replay:
                    used["policy"] = Replay(args.replay, s.kin)
                else:
                    used["policy"] = Remote(args.server, args.seed)
                drive(s, used["policy"], args.limit, args.horizon, args.subtask_every)

            def fields(s):
                # A scene's extra fields can read attributes only its expert sets.
                try:
                    return extra(s) if extra else {}
                except Exception as e:  # noqa: BLE001
                    return {"extra_error": repr(e)}

            def scene(cls):
                return type(
                    f"Policy{cls.__name__}",
                    (Frames, engine(cls)),
                    {"stop_on_violation": not args.lenient},
                )

            return run_episode(
                scene_cls, run, spec, output, True, extra=fields, engine=scene
            )

        tt.run_episode = policy_episode
        result = module.run(args.output.resolve(), args.seed, True, engine=isaac_scene)
        policy = used.get("policy")
        if result["success"]:
            termination = "success"
        elif result["completed"]:
            termination = "completed"
        elif result["error"]:
            # A breakdown after a violation stays a failure, as the violation
            # would have ended the episode without --lenient.
            violated = result["physics"]["violations"]
            termination = "void" if result["void"] and not violated else "error"
        else:
            termination = (
                "time_limit" if result["duration_s"] >= args.limit else "incomplete"
            )
        result.update(
            controller="replay of a recorded run"
            if args.replay
            else "FineART-VLA via fineart.serve",
            termination=termination,
            policy={
                "server": None if args.replay else args.server,
                "replay": str(args.replay) if args.replay else None,
                "limit_s": args.limit,
                "horizon": args.horizon,
                "subtask_every": args.subtask_every,
                "lenient": args.lenient,
                "calls": policy.calls if policy else [],
            },
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        summary = {
            k: result[k]
            for k in (
                "success",
                "completed",
                "termination",
                "error",
                "progress",
                "duration_s",
            )
        }
        print("ISAAC_POLICY", json.dumps(summary), flush=True)
    except Exception:
        traceback.print_exc()
        print("ISAAC_POLICY_FAILED", flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
