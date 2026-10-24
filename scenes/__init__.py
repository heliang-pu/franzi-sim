"""Simulation scenes, one package each. A scene keeps what is its own: the
layout and objects, the scripted expert that plans its episodes, its acceptance
check and the assets only it uses. What every scene shares stays in sim/: the
derived robot with its table and lights (build_task), the policy cameras
(camera_rig) and the episode framework (episode.Scene: model overrides, audits,
recordings, moves through the shared IK).

A scene package holds:

- ``task.py``: the layout and ``build()``, which adds the scene's objects to
  ``sim.build_task.build()`` and returns the tree and the build report. Object
  bodies are named ``<name>_part``: ``Scene.robot_bodies`` excludes that suffix.
- ``run.py``: the ``Scene`` subclass (``build_root``, ``object_info``,
  ``sample``, ``track_objects``, ``complete``), the expert and the CLI.
- ``check.py``: the scene's acceptance.
- ``assets/``: meshes, textures and maps that only this scene uses, referred to
  by absolute path (``Path(__file__).parent / "assets"``), since the compiler's
  meshdir is the robot's.

Scenes import sim and robot_kinematics, never each other. Run them from the
project root as modules, e.g.
``python -m scenes.kitting.run --output outputs/<new_run> --seed 0``.
"""
