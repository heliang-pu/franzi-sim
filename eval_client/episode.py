"""Run a learned policy closed-loop on one benchmark scene episode, from the
default posture until the scene's success test holds or a time limit:

    ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh eval_client.episode <scene> --policy <name> --output outputs/<new_run> --seed 10000 [--server ws://127.0.0.1:18780] [--limit 90] [--set horizon=50]
    python -m eval_client.episode <scene> --engine mujoco --policy <name> --output outputs/<new_run> --seed 10000   (the project env)

The policy is served by policy.serve (policy/<name>/, XPolicyLab's websocket
protocol); here runs its deploy.eval_one_episode, with this module's TaskEnv as
XPolicyLab's TASK_ENV. The settings are policy/<name>/deploy.yml's with each
--set applied (horizon: actions applied per chunk). The scene's own run.run()
runs with its expert replaced (this process wraps sim.tabletop.run_episode), so
the layout, the success test and the run files are those of every expert
episode, and the episode starts as every expert's does, in tabletop.start_home
before the recording. --replay's counterpart is the replay policy
(policy/replay/): a recorded run's commands, the check of this path.

It ends when the scene's complete() has held for 15 ticks, or at --limit
seconds of recording. Success then means what it means for the expert: no
error (penetration over 1 mm, joint limits), complete() held, the grasp
evidence and no audited collision. The carried-object check (CARRY_PHASES) is
keyed to the expert's phase names, so it doesn't apply here. --lenient logs
penetrations and limit errors instead of ending the episode
(Scene.stop_on_violation), so the summary's "completed" (the task done,
whatever was touched) can be counted too; "success" means the same either way.
The termination is success, completed (the task done, with collisions or
violations), error, void (the simulation broke down, before any violation),
time_limit or incomplete. A policy server's error, or a lost server, ends the
episode as an error.
"""

import argparse
import importlib
import json
import traceback
from pathlib import Path

from eval_client.env import Frames, TaskEnv
from policy.serve import PORT, load_config

ROOT = Path(__file__).resolve().parents[1]


class PolicyClient:
    """XPolicyLab's websocket client, its failures raised as RuntimeError so the
    episode records them as an expert's errors are recorded."""

    def __init__(self, url, episode):
        from client_server.ws import WsModelClient

        self.client = WsModelClient(
            url=url, evaluation_id=episode, trial_id=episode, action_case_id=episode
        )

    def call(self, func_name=None, obs=None):
        try:
            return self.client.call(func_name=func_name, obs=obs)
        except RuntimeError:
            raise
        except Exception as e:  # WsError, ConnectionError, ServerRestartedError
            raise RuntimeError(f"policy server: {e}") from e

    def close(self):
        self.client.close()


def termination(result, limit):
    if result["success"]:
        return "success"
    if result["completed"]:
        return "completed"
    if result["error"]:
        # A breakdown after a violation stays a failure, as the violation
        # would have ended the episode without --lenient.
        violated = result["physics"]["violations"]
        return "void" if result["void"] and not violated else "error"
    return "time_limit" if result["duration_s"] >= limit else "incomplete"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene")
    parser.add_argument("--policy", required=True, help="policy/<name>/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--server", default=f"ws://127.0.0.1:{PORT}")
    parser.add_argument("--engine", choices=("isaac", "mujoco"), default="isaac")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--limit", type=float, default=120.0, help="recorded seconds")
    parser.add_argument(
        "--lenient", action="store_true", help="log violations instead of stopping"
    )
    parser.add_argument(
        "--display", action="store_true", help="also expert.mp4 and cameras.mp4"
    )
    args = parser.parse_args()
    config = load_config(ROOT / "policy" / args.policy / "deploy.yml", args.set)
    isaac = args.engine == "isaac"
    if isaac:
        from isaac import runtime

        app = runtime.start_app()
    client = None
    try:
        from sim import tabletop as tt

        if isaac:
            from isaac import engine as isaac_engine
            from isaac.engine import isaac_scene

            isaac_engine.DISPLAY = args.display
        deploy = importlib.import_module(f"policy.{args.policy}.deploy")
        module = importlib.import_module(f"scenes.{args.scene}.run")
        client = PolicyClient(args.server, f"{args.scene}/{args.seed}")
        policy_info = client.call(func_name="describe")
        run_episode = tt.run_episode
        used = {}

        def policy_episode(
            scene_cls, expert, spec, output, record=False, extra=None, engine=None
        ):
            def run(s):
                tt.start_home(s)
                s.stage("policy", "")
                used["env"] = TaskEnv(
                    s, args.scene, args.seed, args.limit, config, policy_info
                )
                deploy.eval_one_episode(TASK_ENV=used["env"], model_client=client)

            def fields(s):
                # A scene's extra fields can read attributes only its expert sets.
                try:
                    return extra(s) if extra else {}
                except Exception as e:  # noqa: BLE001
                    return {"extra_error": repr(e)}

            def scene(cls):
                return type(
                    f"Policy{cls.__name__}",
                    (Frames, engine(cls) if engine else cls),
                    {"stop_on_violation": not args.lenient},
                )

            return run_episode(
                scene_cls, run, spec, output, True, extra=fields, engine=scene
            )

        tt.run_episode = policy_episode
        result = module.run(
            args.output.resolve(),
            args.seed,
            True,
            engine=isaac_scene if isaac else None,
        )
        env = used.get("env")
        result.update(
            controller=f"{policy_info.get('policy', args.policy)} via the policy server",
            termination=termination(result, args.limit),
            policy={
                "name": args.policy,
                "server": args.server,
                "engine": args.engine,
                "config": config,
                "describe": policy_info,
                "limit_s": args.limit,
                "lenient": args.lenient,
                "calls": env.calls if env else [],
            },
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        keys = ("success", "completed", "termination", "error", "progress")
        summary = {k: result[k] for k in (*keys, "duration_s")}
        print("POLICY_EPISODE", json.dumps(summary), flush=True)
    except Exception:
        traceback.print_exc()
        print("POLICY_EPISODE_FAILED", flush=True)
        raise
    finally:
        if client is not None:
            client.close()
        if isaac:
            app.close()


if __name__ == "__main__":
    main()
