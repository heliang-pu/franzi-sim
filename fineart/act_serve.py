"""Serve an ACT checkpoint (fineart.act_train) to closed-loop episodes
(isaac.policy) with fineart.serve's requests and replies. Any number of episodes
connect; their requests take turns on the GPU.

    python -m fineart.act_serve --model <checkpoint> [--port 18780] [--ensemble 0.01]

ACT reads no language: the subtask is ignored, the task only for the checkpoint's
--variants word (fineart.act_train), and the reply's subtask is the task, shown
in the run's videos. Its chunk is deterministic (zero latent), so the seed is
ignored too.

--ensemble is the paper's temporal ensemble with that coefficient (LeRobot's
ACTTemporalEnsembler, one per connection): each tick's action averages what
every chunk so far said about that tick, the oldest weighing most. It needs one
query per tick, isaac.policy --horizon 1, and refuses anything else (a tick's
request has the previous one's seed plus 1). The reply's first action is then
the ensembled one.
"""

import argparse
import json
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

from fineart.act_train import IMAGES, VARIANT, load, one_hot, prepare, tensors
from fineart.data import CAMERAS, CHUNK
from fineart.protocol import AUTHKEY, HOST, PORT, pack, unpack


class Server:
    def __init__(self, model, ensemble, device="cuda:0"):
        self.model = model
        self.ensemble = ensemble
        self.device = device
        self.policy, stats = load(model, device)
        self.policy.eval()
        self.scales = tensors(stats, device)
        self.variants = stats.get("variants")
        self.lock = threading.Lock()
        self.requests = 0

    def info(self):
        config = self.policy.config
        return {
            "policy": "ACT",
            "model": str(self.model.resolve()),
            "chunk": config.chunk_size,
            "inputs": sorted(config.input_features),
            "variants": self.variants,
            "ensemble": self.ensemble,
            "requests": self.requests,
        }

    def infer(self, request, ensembler):
        state = torch.from_numpy(unpack(request["state"]).astype("float32"))
        raw = {"observation.state": state.unsqueeze(0)}
        for camera, key in zip(CAMERAS, IMAGES, strict=True):
            frame = torch.from_numpy(unpack(request["images"][camera]).copy())
            raw[key] = frame.permute(2, 0, 1).unsqueeze(0)
        if self.variants:
            raw[VARIANT] = one_hot(request["task"], self.variants).unsqueeze(0)
        with self.lock, torch.inference_mode():
            start = time.monotonic()
            batch = prepare(raw, self.scales, self.device)
            actions = self.policy.predict_action_chunk(batch)
            mean, std = self.scales["action"]
            actions = actions * std + mean
            if ensembler is not None:
                actions[:, 0] = ensembler.update(actions)
            actions = actions.float().cpu().numpy()[0]
            seconds = time.monotonic() - start
            self.requests += 1
        return {
            "actions": pack(actions),
            "subtask": request["task"],
            "generated": False,
            "seconds": seconds,
        }

    def handle(self, connection):
        ensembler = (
            None
            if self.ensemble is None
            else ACTTemporalEnsembler(self.ensemble, CHUNK)
        )
        previous = None
        with connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    return
                try:
                    if request["op"] == "infer":
                        seed = request["seed"]
                        if ensembler and previous is not None and seed != previous + 1:
                            raise ValueError("--ensemble needs one query per tick")
                        previous = seed
                        reply = self.infer(request, ensembler)
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
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--ensemble", type=float, help="the temporal ensemble's coefficient (0.01)"
    )
    args = parser.parse_args()
    server = Server(args.model, args.ensemble)
    with Listener((HOST, args.port), authkey=AUTHKEY) as listener:
        print("ACT_SERVER", json.dumps(server.info()), flush=True)
        while True:
            connection = listener.accept()
            threading.Thread(
                target=server.handle, args=(connection,), daemon=True
            ).start()


if __name__ == "__main__":
    main()
