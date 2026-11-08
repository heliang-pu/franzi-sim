"""The closed loop on the simulator's side, which the deploy.py files here run.
It imports nothing heavy: it runs in the simulator's Python.

The episode opens with prepare_case and closes with trial_end, so a server
that several episodes share keeps their states apart. When the queue of
actions runs out the model is queried with the tick's observation; it returns
a chunk, whose first `horizon` actions are applied one per control tick. A
model that declares observe_every is also shown every such tick it isn't
queried on (a video memory needs frames at a fixed cadence whatever the
horizon).
"""

import contextlib


def run(TASK_ENV, model_client, horizon):
    meta = TASK_ENV.meta()
    every = TASK_ENV.policy_info.get("observe_every")
    model_client.call(func_name="prepare_case", obs=meta)
    end = {"episode": meta["episode"]}
    queue = []
    try:
        while not TASK_ENV.is_episode_end():
            if not queue:
                reply = model_client.call(func_name="query", obs=TASK_ENV.get_obs())
                queue = list(reply["actions"][:horizon])
                if not queue:
                    raise RuntimeError("the policy returned no actions")
                TASK_ENV.note(reply)
            elif every and TASK_ENV.tick % every == 0:
                model_client.call(func_name="observe", obs=TASK_ENV.get_obs())
            TASK_ENV.take_action(queue.pop(0))
    except BaseException:
        # Free the episode's state, but report what ended the episode.
        with contextlib.suppress(Exception):
            model_client.call(func_name="trial_end", obs=end)
        raise
    model_client.call(func_name="trial_end", obs=end)
