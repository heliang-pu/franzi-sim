"""One stock-SDK interface in front of several simulators.

Layers, top to bottom (each imports only the ones below it):

- ``protocol``: rest_rpc framing and routing for the unmodified V1R1C9 clients.
- ``services``: SDK request validation and response shapes, per RPC method.
- ``control``: engine-independent control semantics (modes, queues, limits,
  enable gate, soft stop, faults) in SDK joint coordinates.
- ``robot``: the robot configuration and the SDK <-> simulator joint mapping.
- ``runtime``: the thread that owns the simulator; commands arrive through a
  queue, state leaves as immutable snapshots.
- ``backends``: one adapter per engine, loaded lazily by name.

Only ``backends/<engine>_backend.py`` may import a physics engine.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
