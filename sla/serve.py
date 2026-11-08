"""Serve an SLA checkpoint to closed-loop episodes (isaac.policy) with fineart.serve's
requests and replies, so isaac.evaluate runs it unchanged:

    python -m sla.serve --model <checkpoint> --assets <assets>/<asset id> \\
        [--scheduler <scheduler checkpoint>] [--port 18780] [--zero-noise]

A request carries the measured 16-D joint state and the four cameras; the reply's
chunk holds 16-D joint targets. Between them SLA sees the hand poses and outputs
relative ones, which its transforms make absolute and sla.eef solves into joints
through the shared kinematics, each row from the one before.

The prompt: in a short scene the instruction. In a long scene the scheduler's coarse
step, asked whenever a request comes without a subtask (every isaac.policy
--subtask-every ticks) and kept per connection, an episode being one connection.

--zero-noise starts the flow at zero instead of at the request's seeded noise, so
consecutive chunks differ only by what the model saw.

--memory-frames > 1 serves a checkpoint trained with MEM (sla.mem): the info reply
says how many frames how many ticks apart, isaac.policy sends that history with each
request, and the frames before the episode's start are its first frame.
"""

import argparse
import json
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.shared import normalize

from fineart.protocol import AUTHKEY, HOST, PORT, pack, unpack
from sla import mem
from sla.eef import Hands
from sla.policy import CAMERAS
from sla.train import inference_config, model


class Server:
    def __init__(self, checkpoint, assets, scheduler, zero_noise, memory):
        self.checkpoint = checkpoint
        self.zero_noise = zero_noise
        self.memory = memory  # (frames, stride) or None
        self.policy = policy_config.create_trained_policy(
            inference_config(assets), checkpoint, norm_stats=normalize.load(assets)
        )
        self.scheduler = None
        if scheduler:
            from sla.scheduler import load

            self.scheduler = load(scheduler)
        self.lock = threading.Lock()
        self.requests = 0

    def info(self):
        return {
            "policy": "SLA",
            "model": str(self.checkpoint),
            "scheduler": self.scheduler is not None,
            "zero_noise": self.zero_noise,
            "memory": None
            if self.memory is None
            else {
                "frames": self.memory[0],
                "stride": self.memory[1],
                "cameras": list(CAMERAS.values()),
            },
            "requests": self.requests,
        }

    def infer(self, request, hands, episode):
        state = unpack(request["state"]).astype(np.float64)
        frames = {c: unpack(request["images"][c]) for c in request["images"]}
        subtask, generated = request.get("subtask"), False
        if not subtask:
            generated = True
            subtask = request["task"]
            if self.scheduler and self.scheduler.handles(subtask):
                with self.lock:
                    if "schedule" not in episode:
                        episode["schedule"] = self.scheduler.episode(subtask)
                    subtask = episode["schedule"].step(frames)
        images = {camera: frames[camera] for camera in CAMERAS.values()}
        if self.memory is not None:
            # The history, oldest first; the current frame repeated if none came.
            past = request.get("history") or {}
            images = {
                camera: np.stack(
                    [
                        *(unpack(f) for f in past.get(camera, [])),
                        frame,
                    ][-self.memory[0] :]
                    if past.get(camera)
                    else [frame] * self.memory[0]
                )
                for camera, frame in images.items()
            }
        obs = {"images": images, "state": state, "prompt": subtask}
        config = model()
        noise = (
            np.zeros((config.action_horizon, config.action_dim), np.float32)
            if self.zero_noise
            else None
        )
        with self.lock:
            start = time.monotonic()
            # The reply depends on the request alone.
            self.policy._rng = jax.random.key(int(request.get("seed", 0)))
            poses = self.policy.infer(obs, noise=noise)["actions"]
            seconds = time.monotonic() - start
            self.requests += 1
        rows, errors = hands.joints(poses, state)
        return {
            "actions": pack(rows.astype(np.float32)),
            "subtask": subtask,
            "generated": generated,
            "seconds": seconds,
            "ik_errors": errors,
        }

    def handle(self, connection):
        hands, episode = Hands(), {}
        with connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    return
                try:
                    if request["op"] == "infer":
                        reply = self.infer(request, hands, episode)
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
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument(
        "--memory-frames", type=int, default=1, help="MEM's frames (sla.mem); 1 is off"
    )
    parser.add_argument("--memory-stride", type=int, default=mem.STRIDE, help="ticks")
    args = parser.parse_args()
    if args.memory_frames > 1:
        mem.install(args.memory_frames, args.memory_stride)
    memory = (
        (args.memory_frames, args.memory_stride) if args.memory_frames > 1 else None
    )
    server = Server(args.model, args.assets, args.scheduler, args.zero_noise, memory)
    with Listener((HOST, args.port), authkey=AUTHKEY) as listener:
        print("SLA_SERVER", json.dumps(server.info()), flush=True)
        while True:
            connection = listener.accept()
            threading.Thread(
                target=server.handle, args=(connection,), daemon=True
            ).start()


if __name__ == "__main__":
    main()
