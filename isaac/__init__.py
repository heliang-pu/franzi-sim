"""The Isaac Sim port of the scenes (user request 2026-10-25: the industrial
benchmark first).

A scene is ported from its compiled MuJoCo model, so the layout, colliders,
masses, joints and cameras are the ones every MuJoCo run uses:
- export.py builds the scene and writes the model as plain data (model.json,
  arrays.npz, textures/); it runs in the project env;
- stage.py authors the USD stage from that data, runtime.py runs it on PhysX
  and check.py checks it against MuJoCo;
- engine.py runs a scene's expert closed-loop on Isaac, with MuJoCo's data as a
  kinematic mirror of Isaac's state, and episode.py is its command line.
All but export.py run in Isaac Sim's own Python, which isaac/run.sh starts.
"""
