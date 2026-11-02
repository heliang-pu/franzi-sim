"""Serve a FineART checkpoint to closed-loop episodes (isaac.policy) over a
local socket (fineart.protocol). Any number of episodes connect; their requests
take turns on the GPU.

    python -m fineart.serve --model <checkpoint> --tokenizer <paligemma_tokenizer> [--port 18780]

A request {"op": "infer", "task", "state", "images", "subtask", "memory",
"seed"} carries the episode's instruction, the measured 16-D state in the
datasets' SDK units (packed float32), the four policy cameras' frames by
dataset camera name (packed uint8 [480, 640, 3]), the current subtask and the
memory (the previous query's subtask, None at first). With subtask None the
server first generates one from the task (select_message, on the prompt the
checkpoint's recipe trains it on; the memory recipe's also holds the memory,
fineart.recipe), then predicts the action chunk on the prompt the recipe's
low-level stream trained: the subtask alone for the bimanual base's recipe,
"<task> Subtask: <subtask>" for super_poulain's. seed seeds the flow
matching's noise, so a reply depends on its request alone. The reply holds the
chunk (packed float32 [50, 16], SDK units), the subtask, whether it was
generated, and the seconds the GPU took. {"op": "info"} describes the policy.
"""

import argparse
import json
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from lerobot.utils.constants import MESSAGES_RENDERED

from fineart import recipe
from fineart.data import CAMERAS, ROBOT
from fineart.protocol import AUTHKEY, HOST, PORT, pack, unpack
from fineart.train import load

SUBTASK_TOKENS = 40


def action_prompt(model):
    """The checkpoint's low-level prompt template (its recipe's user message
    for the action stream), with ${task} and ${subtask} in it."""
    processor = json.loads((Path(model) / "policy_preprocessor.json").read_text())
    step = next(
        s
        for s in processor["steps"]
        if s["registry_name"] == "render_training_messages_processor"
    )
    messages = step["config"]["recipe"]["blend"]["low_level_execution"]["messages"]
    (message,) = [m for m in messages if m["role"] == "user"]
    return message["content"]


class Server:
    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model
        self.policy, self.pre, self.post = load(model, tokenizer, device, False)
        self.policy.reset()
        self.lock = threading.Lock()
        self.prompt = action_prompt(model)
        self.recipe = recipe.load(model)
        self.requests = 0

    def info(self):
        config = self.policy.config
        return {
            "model": str(self.model.resolve()),
            "chunk": config.chunk_size,
            "inputs": sorted(config.input_features),
            "action_names": config.action_feature_names,
            "action_prompt": self.prompt,
            "memory": recipe.uses_memory(self.recipe),
            "requests": self.requests,
        }

    def infer(self, request):
        state = torch.from_numpy(unpack(request["state"]).astype("float32"))
        obs = {"observation.state": state, "task": request["task"], "robot_type": ROBOT}
        for camera in CAMERAS:
            frame = torch.from_numpy(unpack(request["images"][camera]).copy())
            obs[f"observation.images.{camera}"] = frame.permute(2, 0, 1).float() / 255
        subtask, generated = request.get("subtask"), False
        with self.lock, torch.inference_mode():
            start = time.monotonic()
            torch.manual_seed(int(request.get("seed", 0)))
            if not subtask:
                messages = recipe.subtask_messages(
                    self.recipe, request["task"], request.get("memory")
                )
                query = obs | {MESSAGES_RENDERED: messages}
                subtask = self.policy.select_message(
                    self.pre(query),
                    max_new_tokens=SUBTASK_TOKENS,
                    suppress_loc_tokens=True,
                ).strip()
                generated = True
            prompt = self.prompt.replace("${task}", request["task"])
            batch = self.pre(obs | {"task": prompt.replace("${subtask}", subtask)})
            actions = self.post(self.policy.predict_action_chunk(batch))
            actions = actions.float().cpu().numpy()[0]
            seconds = time.monotonic() - start
            self.requests += 1
        return {
            "actions": pack(actions),
            "subtask": subtask,
            "generated": generated,
            "seconds": seconds,
        }

    def handle(self, connection):
        with connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    return
                try:
                    if request["op"] == "infer":
                        reply = self.infer(request)
                    elif request["op"] == "info":
                        reply = self.info()
                    else:
                        reply = {"error": f"unknown op {request['op']!r}"}
                except Exception as e:  # noqa: BLE001, the episode reports it
                    reply = {"error": repr(e)}
                connection.send(reply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    server = Server(args.model, args.tokenizer)
    with Listener((HOST, args.port), authkey=AUTHKEY) as listener:
        print("FINEART_SERVER", json.dumps(server.info()), flush=True)
        while True:
            connection = listener.accept()
            threading.Thread(
                target=server.handle, args=(connection,), daemon=True
            ).start()


if __name__ == "__main__":
    main()
