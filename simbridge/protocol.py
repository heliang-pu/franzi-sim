"""rest_rpc wire layer for the stock V1R1C9 clients. No simulator imports.

Frame: 20-byte little-endian header ``<BB2xIQI`` (magic 39, request type, body
length, request id, route id) followed by a MessagePack argument tuple. The
reply body is ``(0, value)`` on success or ``(1, message)`` on a protocol-level
error, which the stock clients always raise as an exception.
"""

import ast
import logging
import socketserver
import struct
import threading
import time

from simbridge.status import InvalidRequest, Unsupported

HEADER = struct.Struct("<BB2xIQI")
MAGIC = 39
MAX_BODY = 10 * 1024 * 1024
ARM_PREFIX = "arm_sdk."
# Route names of module_communication/client.py FUNC_IDS (sensor/chassis port).
SENSOR_ROUTES = (
    "publish_vehicle_control",
    "start_localization",
    "send_nav_goal",
    "send_move_goal_simple",
    "cancel_nav_goal",
    "get_navigate_state",
    "get_chassis_state",
    "get_image",
)
log = logging.getLogger("simbridge.protocol")


def wire():
    """MessagePack and route hashing bundled with the vendor SDK (read-only)."""
    from sdk.stock_sdk import load_wire

    return load_wire()


def arm_catalog(sdk_client_dir):
    """Every route the stock ArmClient can call: ``self.rpc("Name", ...)``, and
    names handed to a helper that forwards a parameter to ``self.rpc`` (the
    vendor's ``_fk_impl("Fk", ...)``)."""
    trees = [
        ast.parse(p.read_text(encoding="utf-8"))
        for p in sorted(sdk_client_dir.glob("*.py"))
    ]

    def calls(tree, method):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
                and node.args
            ):
                yield node

    forwarders = {}  # helper name -> index of the argument it sends as the route
    for tree in trees:
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                params = [a.arg for a in fn.args.args[1:]]
                for call in calls(fn, "rpc"):
                    first = call.args[0]
                    if isinstance(first, ast.Name) and first.id in params:
                        forwarders[fn.name] = params.index(first.id)
    names = set()
    for tree in trees:
        for method, index in [("rpc", 0), *forwarders.items()]:
            for call in calls(tree, method):
                arg = call.args[index] if index < len(call.args) else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
    return sorted(names)


class Endpoint:
    """One TCP port. ``dispatch(name, args)`` runs on the connection's thread."""

    def __init__(self, name, host, port, routes, dispatch, prefix=""):
        _, md5_hash32 = wire()
        self.name = name
        self.prefix = prefix
        self.routes = {md5_hash32(prefix + r): r for r in routes}
        self.dispatch = dispatch
        self.calls = []
        self.calls_lock = threading.Lock()
        self.server = _Server((host, port), _Handler)
        self.server.endpoint = self
        self.thread = threading.Thread(
            target=self.server.serve_forever, name=f"rpc-{name}", daemon=True
        )

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def record(self, name, outcome, started):
        with self.calls_lock:
            self.calls.append(
                {
                    "endpoint": self.name,
                    "method": name,
                    "outcome": outcome,
                    "wall_time": started,
                    "latency_ms": round((time.time() - started) * 1000, 3),
                }
            )

    def answer(self, kind, funcid, body):
        msgpack, _ = wire()
        started = time.time()
        if kind != 0:
            self.record(f"subscribe:0x{funcid:08x}", "unsupported", started)
            return None
        name = self.routes.get(funcid)
        try:
            if name is None:
                raise Unsupported(f"unknown route 0x{funcid:08x} on {self.name} port")
            args = msgpack.unpackb(body, raw=False, strict_map_key=False)
            if not isinstance(args, (list, tuple)):
                raise InvalidRequest("arguments must be a MessagePack tuple")
            value = self.dispatch(name, tuple(args))
            outcome = "ok"
            payload = (0, value)
        except Unsupported as exc:
            outcome, payload = "unsupported", (1, f"UNSUPPORTED: {exc}")
        except (InvalidRequest, KeyError, TypeError, ValueError, IndexError) as exc:
            outcome = "invalid"
            payload = (1, f"INVALID_REQUEST {self.prefix}{name}: {exc!r}")
        except TimeoutError as exc:
            outcome, payload = "timeout", (1, f"TIMEOUT {self.prefix}{name}: {exc}")
        except Exception as exc:  # reported to the client, not hidden
            log.exception("handler failed: %s", name)
            outcome = "internal_error"
            payload = (1, f"INTERNAL {self.prefix}{name}: {type(exc).__name__}: {exc}")
        self.record(name or f"0x{funcid:08x}", outcome, started)
        return msgpack.packb(payload, use_bin_type=True)


class _Handler(socketserver.BaseRequestHandler):
    def exact(self, n):
        data = bytearray()
        while len(data) < n:
            part = self.request.recv(n - len(data))
            if not part:
                raise EOFError
            data.extend(part)
        return bytes(data)

    def handle(self):
        endpoint = self.server.endpoint
        while True:
            try:
                header = self.exact(HEADER.size)
                magic, kind, length, reqid, funcid = HEADER.unpack(header)
                if magic != MAGIC or length > MAX_BODY:
                    return
                if length == 0:
                    # Heartbeat frame: the stock client expects the header back.
                    self.request.sendall(header)
                    continue
                payload = endpoint.answer(kind, funcid, self.exact(length))
                if payload is not None:
                    self.request.sendall(
                        HEADER.pack(MAGIC, 0, len(payload), reqid, funcid) + payload
                    )
            except (EOFError, ConnectionError, OSError):
                return


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
