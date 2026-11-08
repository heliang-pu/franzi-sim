"""Simulation scenes, one package each. A scene keeps what is its own: the
layout and objects, the scripted expert that plans its episodes, its acceptance
check and the assets only it uses. The robot comes from robots/ (robots.franzi:
the derived robot with its table and lights, the policy cameras, the shared IK),
and what every scene shares stays in sim/: the episode framework (episode.Scene:
model overrides, audits, recordings, moves through the shared IK), the tabletop
helpers, the prop loaders, the work cell and the checks.

A scene package holds:

- ``task.py``: the layout and ``build()``, which adds the scene's objects to
  ``robots.franzi.scene.build()`` and returns the tree and the build report.
  Object bodies are named ``<name>_part``: ``Scene.robot_bodies`` excludes that
  suffix.
- ``run.py``: the ``Scene`` subclass (``build_root``, ``object_info``,
  ``sample``, ``track_objects``, ``complete``), the expert and the CLI.
- ``check.py``: the scene's acceptance.
- ``assets/``: meshes, textures and maps that only this scene uses, referred to
  by absolute path (``Path(__file__).parent / "assets"``), since the compiler's
  meshdir is the robot's. A Blender prop only this scene uses is a folder
  ``assets/<id>/``, which ``sim.prop_assets`` finds by its id; props several
  scenes share are in the library, the top-level ``assets/props/``.

Scenes import sim and robots.franzi, never each other. Run them from the
project root as modules, e.g.
``python -m scenes.kitting.run --output outputs/<new_run> --seed 0``.
"""
