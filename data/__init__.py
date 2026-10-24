"""VLA training data: SDK-side episode recording, expert-run conversion and the
LeRobot export. Import-free on purpose: export_lerobot runs in the lerobot env,
which has no stock SDK, simbridge or Pinocchio.

Run the scripts from the project root as modules, e.g.
``python -m data.record_sdk_episode ...``.
"""
