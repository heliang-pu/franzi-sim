"""What every simulation scene shares: the episode framework (episode.Scene), the
tabletop helpers and object scenes (tabletop), the props (prop_assets for the
Blender-built ones, props and robosyn for the RoboSyn ones), the industrial
work cell (workcell) and the scene checks (bench_check, scene_check).

The robot comes from robots/ (robots.franzi: the derived robot scene, its
kinematics, cameras and appearance); the framework is written for that one robot
for now. The scenes themselves, each with its layout, expert, acceptance and
assets, live in scenes/. Run the scripts from the project root as modules, e.g.
``python -m sim.bench_check``.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The plain table every robot scene stands at (robots.franzi.scene.build); the
# industrial scenes put the work bench (sim.workcell.BENCH_Z) in its place.
TABLE_Z = 0.80
# Part colours shared by the scenes (scenes/pick_place, scenes/kitting).
PART_RGBA = {"blue": "0.12 0.48 0.88 1", "orange": "0.95 0.48 0.12 1"}


def fmt(a):
    return " ".join(f"{v:.9g}" for v in np.asarray(a).reshape(-1))
