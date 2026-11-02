"""VLA training data: SDK-side episode recording, expert-run conversion, the
LeRobot export (v2.1 or v3.0) and the conversion between the two. Import-free on
purpose: export_lerobot and convert_lerobot run in the lerobot env, which has no
stock SDK, simbridge or Pinocchio, and check_lerobot_v21 in openpi's env.

Run the scripts from the project root as modules, e.g.
``python -m data.record_sdk_episode ...``.
"""
