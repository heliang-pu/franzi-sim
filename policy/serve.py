"""Serve a policy over XPolicyLab's websocket protocol:

    python -m policy.serve --config policy/<name>/deploy.yml [--set key=value ...] [--host 127.0.0.1] [--port 18780]

Runs in the policy's own env, which needs XPolicyLab at commit 6baccd5
(pip install --no-deps) and its transport's packages (websockets, msgpack,
msgpack-numpy, pydantic, and opencv and h5py, which its server imports). The
config is deploy.yml with each --set applied as a YAML value (--set
checkpoint=/path --set ensemble=0.01); the model is
policy.<policy_name>.model.Model(config). It loads before the port opens, which
XPolicyLab's clients wait for. The server prints POLICY_SERVER and the model's
describe() once it is about to listen, and binds 127.0.0.1 unless told
otherwise: the protocol has no authentication.

XPolicyLab's server runs one model call at a time (a server-wide lock); this
one runs the calls of the episodes sharing it at once, and each model locks
its GPU work itself (policy/session.py).
"""

import argparse
import asyncio
import importlib
import json
import logging
from pathlib import Path

import yaml

PORT = 18780


class Unlocked:
    """An async context manager that holds nothing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def policy_server(model, host, port, ping_interval=20.0, ping_timeout=20.0):
    """XPolicyLab's PolicyServer for model, its server-wide lock lifted."""
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=host,
            port=int(port),
            ws_ping_interval_s=ping_interval,
            ws_ping_timeout_s=ping_timeout,
        ),
    )
    # The lock _call_model_method holds around every model call (XPolicyLab 6baccd5).
    assert isinstance(server._model_lock, asyncio.Lock), "XPolicyLab changed"
    server._model_lock = Unlocked()
    return server


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set wants key=value, not {item!r}")
        config[key] = yaml.safe_load(value)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config, args.set)
    host = args.host or config.get("host") or "127.0.0.1"
    port = args.port or config.get("port") or PORT
    module = importlib.import_module(f"policy.{config['policy_name']}.model")
    model = module.Model(config)
    print("POLICY_SERVER", json.dumps(model.describe()), flush=True)
    server = policy_server(
        model,
        host,
        port,
        config.get("ws_ping_interval_s", 20.0),
        config.get("ws_ping_timeout_s", 20.0),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
