"""Restore source housing points before the final one-pass bounded rounding.

    An idempotency regression caught a second rounding pass after whole-piece
    precision promotion. Recover the source points without remeshing; retain
    every existing face and index, then apply the bounded compactor once.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy
import numpy as np
import prop_lib as props

QA = HERE / "_qa" / "task13"
ASSETS = props.ROOT / "assets" / "props"
rows = json.loads((QA / "obj.json").read_text())["obj"]
changed = {row["file"] for row in rows if row["max_vertex_change_mm"] > 1e-9}
restored = []
for name in ("bearing_housing_c010", "bearing_housing_c020", "bearing_housing_c030"):
    folder = ASSETS / name
    data = json.loads((folder / "prop.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder / "source.blend"))
    for part in data["parts"]:
        for reference, source_name in zip(part["collision"], part["source_collision_objects"], strict=True):
            path = folder / reference
            if f"{name}/{reference}" not in changed:
                continue
            mesh = props._world_mesh(bpy.data.objects[source_name])
            source = np.asarray([tuple(vertex.co) for vertex in mesh.vertices])
            bpy.data.meshes.remove(mesh)
            lines = path.read_text().splitlines()
            positions = np.asarray([[float(value) for value in line.split()[1:4]] for line in lines if line.startswith("v ")])
            distances = np.linalg.norm(positions[:, None, :] - source[None, :, :], axis=2)
            nearest = distances.argmin(axis=1)
            assert len(set(nearest)) == len(positions) and distances.min(axis=1).max() < .000012, path
            replacements = iter(source[nearest])
            output = ["v " + " ".join(f"{value:.12f}" for value in next(replacements))
                      if line.startswith("v ") else line for line in lines]
            path.write_text("\n".join(output) + "\n")
            restored.append(f"{name}/{reference}")
assert len(restored) == len(changed), (len(restored), len(changed))
(QA / "restored_precision_sources.json").write_text(json.dumps(restored, indent=2) + "\n")
print("RESTORED_SOURCE_POINTS", len(restored), "unchanged faces/indices")
