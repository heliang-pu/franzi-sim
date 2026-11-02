"""Sort wheel_robot_26.8.16_3's gripper surface into the linkage's moving parts.

wheel_robot_26.8.16_3 exports each gripper's body, links, jaw blocks, pins and
screws as one surface. wheel_robot_4.0's wrist mesh kept them as separate CAD
components: sim.gripper_linkage.group_of picks the moving ones by bounding box,
and that model painted its jaw blocks and pins aluminium and its screws red by
size (FINISHES). This gives every face of the new gripper mesh the moving part
and the finish of the wheel_robot_4.0 component whose surface it lies on and
writes sim/gripper_faces.json, which gripper_linkage.face_table() reads for the
linkage's visuals and colliders.

In wheel_robot_4.0's wrist frame (franzi_model.GRIPPERS, which goes through the
identical fingers) the new body, links and jaw blocks sit 3.5 mm off the old
ones along y, the hinge axes: the pads moved on their jaw blocks. The fit finds
that offset. A face then takes the component whose shifted surface is nearest
and faces the same way (normals within 60 degrees), up to 1 mm away; any other
face stays on the gripper body, black.

    python -m tools.gripper_faces [--repo PATH]

The wheel_robot_4.0 wrist meshes are read from the git history (OLD_COMMIT) of
the repository at --repo (default: this checkout).
"""

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from robot_kinematics import ROOT, SOURCE
from sim import franzi_model
from sim.gripper_linkage import FACES, group_of

OLD_COMMIT = "7f1fe94"  # the initial import, wheel_robot_4.0
OLD_MESH = "assets/franzi_source/meshes/{side}_wrist_roll_Link.STL"
LINKS = {"left": "left_gripper_link", "right": "right_gripper_Link"}
# wheel_robot_4.0's finishes by component size (m), as its paint recipe had them.
FINISHES = (
    ((0.0317, 0.0175, 0.028), "fa_silver"),  # jaw blocks
    ((0.0075, 0.029, 0.0075), "fa_silver"),  # pins
    ((0.008, 0.0055, 0.0055), "fa_estop_red"),  # screws on the jaw blocks
)
DENSITY = 20  # surface samples per mm2
GATE = 0.001  # m: farthest a face may lie from the component it takes after
AGREE = 0.5  # cosine: normals within 60 degrees
NEAREST = 32  # samples compared per face


def old_components(repo, side):
    """wheel_robot_4.0's wrist mesh as [(component, moving group or None,
    finish or None)], each component wound outward."""
    path = f"{OLD_COMMIT}:{OLD_MESH.format(side=side)}"
    data = subprocess.run(
        ["git", "-C", str(repo), "show", path], capture_output=True, check=True
    ).stdout
    mesh = trimesh.load(io.BytesIO(data), file_type="stl", process=True, force="mesh")
    result = []
    for component in mesh.split(only_watertight=False):
        size = np.ptp(component.bounds, axis=0)
        finish = next((f for s, f in FINISHES if np.abs(size - s).max() < 0.0015), None)
        result.append((outward(component), group_of(component.bounds), finish))
    return result


def outward(mesh):
    """The mesh with every body's faces wound outward (in place; face order
    kept): 21 of the old wrist mesh's components mix both windings."""
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def samples(meshes):
    """Points on the meshes' surfaces, their face normals and mesh indices."""
    points, normals, owner = [], [], []
    for k, mesh in enumerate(meshes):
        count = int(mesh.area * 1e6 * DENSITY)
        p, face = trimesh.sample.sample_surface(mesh, count, seed=0)
        points += [p, mesh.triangles_center]
        normals += [mesh.face_normals[face], mesh.face_normals]
        owner.append(np.full(len(p) + len(mesh.faces), k))
    return np.concatenate(points), np.concatenate(normals), np.concatenate(owner)


def fit_offset(points, normals, new):
    """Translation (m) from the old moving parts onto the new surface:
    point-to-plane least squares over faces that match within 5 mm."""
    tree = cKDTree(points)
    offset = np.zeros(3)
    for _ in range(20):
        dist, index = tree.query(
            new.triangles_center - offset, distance_upper_bound=0.005
        )
        ok = np.isfinite(dist)
        ok[ok] &= (normals[index[ok]] * new.face_normals[ok]).sum(1) > AGREE
        n = normals[index[ok]]
        residual = ((new.triangles_center[ok] - offset - points[index[ok]]) * n).sum(1)
        step = np.linalg.lstsq(n, residual, rcond=None)[0]
        offset += step
        if np.abs(step).max() < 1e-7:
            break
    return offset


def nearest(points, normals, owner, new):
    """Each new face's nearest old component facing its way (-1: none within
    GATE) and its distance (m)."""
    dist, index = cKDTree(points).query(
        new.triangles_center, k=NEAREST, distance_upper_bound=GATE
    )
    found = np.isfinite(dist)
    index = np.where(found, index, 0)
    agree = (normals[index] * new.face_normals[:, None]).sum(2) > AGREE
    dist = np.where(found & agree, dist, np.inf)
    best = dist.argmin(1)
    rows = np.arange(len(dist))
    hit = np.isfinite(dist[rows, best])
    return np.where(hit, owner[index[rows, best]], -1), dist[rows, best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    table = {
        "_": (
            "The wheel_robot_4.0 moving part and finish of each face of "
            "wheel_robot_26.8.16_3's gripper meshes (faces in file order; '.' the "
            "gripper body, black), and the offset (m) of the new parts from the "
            "old in wheel_robot_4.0's wrist frame. Made by tools/gripper_faces.py."
        )
    }
    for side, link in LINKS.items():
        path = SOURCE / "meshes" / f"{link}.STL"
        new = outward(trimesh.load(path, process=True, force="mesh"))
        new.apply_transform(franzi_model.to_previous(link))
        old = old_components(args.repo, side)
        moving = [k for k, (_, group, _) in enumerate(old) if group]
        points, normals, owner = samples([m for m, _, _ in old])
        on_moving = np.isin(owner, moving)
        offset = fit_offset(points[on_moving], normals[on_moving], new)
        component, dist = nearest(points + offset, normals, owner, new)
        group = np.array([old[k][1] if k >= 0 else None for k in component])
        finish = np.array([old[k][2] if k >= 0 else None for k in component])
        names = sorted({g for _, g, _ in old if g})
        code = {None: "."} | {g: chr(ord("a") + k) for k, g in enumerate(names)}
        paints = sorted({f for _, _, f in old if f})
        paint = {None: "."} | {f: str(k) for k, f in enumerate(paints)}
        table[link] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "faces": len(new.faces),
            "offset_m": np.round(offset, 6).tolist(),
            "groups": {code[g]: g for g in names},
            "labels": "".join(code[g] for g in group),
            "finishes": {paint[f]: f for f in paints},
            "paint": "".join(paint[f] for f in finish),
        }
        area = new.area_faces * 1e6
        print(f"{link}: offset {np.round(offset * 1000, 2)} mm")
        for g in names:
            on = group == g
            was = sum(m.area for m, og, _ in old if og == g) * 1e6
            print(
                f"  {g:9s} {on.sum():5d} faces {area[on].sum():6.0f} mm2 "
                f"(old part {was:6.0f} mm2), p99 {np.percentile(dist[on], 99) * 1000:.2f} mm"
            )
        far = component < 0
        print(
            f"  body      {(group == None).sum():5d} faces, of which {far.sum()} "
            f"({area[far].sum():.0f} mm2) lie on no old part"
        )
        for f in paints:
            print(f"  {f:13s} {area[finish == f].sum():6.0f} mm2")
    FACES.write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
