"""Derive a fixed-base dual-arm scene without modifying the source model."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from robot_kinematics import ARM_JOINTS, controller_ranges
from sim.appearance import apply_appearance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/franzi_source"
__all__ = ["ARM_JOINTS"]  # re-exported: one definition, in robot_kinematics
TABLE_Z = 0.80
# Far enough out for the head and body cameras to see the parts and pads.
SOURCES = {"left": [0.38, 0.34], "right": [0.38, -0.34]}
GOALS = {"left": [0.40, 0.15], "right": [0.40, -0.15]}
# The two part colours, shared with kitting_task.py.
PART_RGBA = {"blue": "0.12 0.48 0.88 1", "orange": "0.95 0.48 0.12 1"}


def fmt(a):
    return " ".join(f"{v:.9g}" for v in np.asarray(a).reshape(-1))


def align_joint_limits(root):
    """Set joint ranges, and the matching actuator ctrlranges, to the controller-
    aligned ranges of robot_kinematics.controller_ranges (the IK's bounds)."""
    ranges = controller_ranges()
    joints = {j.get("name"): j for j in root.iter("joint")}
    actuators = {a.get("joint"): a for a in root.find("actuator") if a.get("joint")}
    for name, r in ranges.items():
        joints[name].set("range", fmt(r["aligned_rad"]))
        if name in actuators and actuators[name].get("ctrlrange"):
            actuators[name].set("ctrlrange", fmt(r["aligned_rad"]))
    return ranges


def build(task_objects=True):
    """Derived scene tree and report. task_objects=False leaves out the two parts
    and their pads, for a task module that brings its own (kitting_task.py)."""
    root = ET.parse(SOURCE / "franzi.xml").getroot()
    root.set("model", "franzi_dual_arm_kitting")
    compiler = root.find("compiler")
    compiler.set("meshdir", str(SOURCE / "meshes"))
    compiler.set("texturedir", str(SOURCE / "assets"))
    root.find("option").set("timestep", str(1 / 500))
    root.find("option").set("iterations", "100")
    root.find("option").set("cone", "elliptic")
    root.remove(root.find("keyframe"))
    base = root.find(".//body[@name='base_link']")
    base.remove(base.find("freejoint"))
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="studio_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.55 0.62 0.72",
        rgb2="0.85 0.89 0.94",
        width="256",
        height="1536",
    )
    report = {
        "source": "wheel_robot_4.0.urdf",
        "base": "fixed to floor for arm-only experiment",
        "objects_pose_written_during_episode": False,
        "attachment_constraints": False,
        "wrist_collision": {},
        "finger_collision": {},
    }
    report["joint_limits"] = align_joint_limits(root)
    report["appearance"] = apply_appearance(root, SOURCE)
    for side in ["left", "right"]:
        # A single wrist AABB fills the real gripper opening. Each disconnected
        # CAD part gets its own convex collider, preserving the free jaw space.
        name = side + "_wrist_roll_Link"
        body = root.find(f".//body[@name='{name}']")
        body.remove(body.find(f"geom[@name='{name}']"))
        mesh = trimesh.load(
            SOURCE / "meshes" / (name + ".STL"), process=True, force="mesh"
        )
        edges = mesh.edges_unique
        graph = coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(len(mesh.vertices),) * 2,
        )
        count, labels = connected_components(graph, directed=False)
        parts = 0
        for c in range(count):
            vertices = mesh.vertices[labels == c]
            if (
                len(vertices) < 4
                or np.linalg.matrix_rank(vertices - vertices.mean(0)) < 3
            ):
                continue
            hull = trimesh.convex.convex_hull(vertices)
            if hull.volume < 1e-12:
                continue
            part = f"{side}_wrist_part_{parts:03d}"
            ET.SubElement(
                asset,
                "mesh",
                name=part,
                vertex=fmt(hull.vertices),
                face=fmt(hull.faces),
            )
            ET.SubElement(
                body,
                "geom",
                name=part,
                type="mesh",
                mesh=part,
                **{"class": "collision"},
            )
            parts += 1
        report["wrist_collision"][side] = {
            "cad_components": int(count),
            "convex_colliders": parts,
        }
        for finger in [1, 2]:
            name = f"{side}finger{finger}_Link"
            body = root.find(f".//body[@name='{name}']")
            geom = body.find(f"geom[@name='{name}']")
            geom.attrib.clear()
            geom.attrib.update(
                name=name,
                type="mesh",
                mesh=name,
                friction="1.2 0.01 0.0001",
                condim="4",
                solref="0.004 1",
                solimp="0.95 0.99 0.001",
                **{"class": "collision"},
            )
            mesh = trimesh.load(
                SOURCE / "meshes" / (name + ".STL"), process=True, force="mesh"
            )
            report["finger_collision"][name] = {
                "method": "convex hull of original finger STL; flat grasp face preserved",
                "bounds_m": mesh.bounds.tolist(),
                "vertices": len(mesh.vertices),
                "hull_to_mesh_volume_ratio": float(
                    mesh.convex_hull.volume / abs(mesh.volume)
                ),
            }
        # Grasp center is midway between the two prismatic finger origins.
        body = root.find(f".//body[@name='{side}_wrist_roll_Link']")
        ET.SubElement(
            body,
            "site",
            name=f"{side}_grasp",
            pos=f"-0.03045 {'-0.00175' if side == 'left' else '0.00175'} -0.245",
            size="0.004",
            group="4",
        )
    # Preserve robot-world contact, additionally enable left-vs-right arm
    # contacts. Adjacent links within an arm keep the source model's exclusions.
    for body in root.findall(".//body"):
        name = body.get("name", "")
        side = (
            "left"
            if name.startswith("left") and "front" not in name
            else "right"
            if name.startswith("right") and "front" not in name
            else None
        )
        if side:
            for geom in body.findall("geom"):
                if geom.get("class") == "collision":
                    geom.set("contype", "5" if side == "left" else "9")
                    geom.set("conaffinity", "10" if side == "left" else "6")
    world = root.find("worldbody")
    # MuJoCo centres a directional light's shadow box on the beam through its pos,
    # so the key's pos puts that axis on the robot and table (its direction, and so
    # the lighting, is unchanged); the box then only spans what it shades.
    ET.SubElement(
        world,
        "light",
        name="studio_key",
        pos="1.09 -0.68 3.48",
        dir="-0.4 0.3 -1",
        ambient="0 0 0",
        diffuse="0.40 0.40 0.40",
        specular="0.12 0.12 0.12",
        directional="true",
        castshadow="true",
    )
    ET.SubElement(
        world,
        "light",
        name="studio_fill",
        pos="1 2 2.5",
        dir="-0.3 -0.5 -1",
        ambient="0 0 0",
        diffuse="0.25 0.25 0.25",
        specular="0.05 0.05 0.05",
        directional="true",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="3 3 0.05",
        rgba="0.42 0.46 0.52 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="table_top",
        type="box",
        pos=f"0.42 0 {TABLE_Z - 0.025}",
        size="0.24 0.65 0.025",
        rgba="0.62 0.66 0.70 1",
        friction="0.7 0.005 0.0001",
        solref="0.004 1",
        solimp="0.95 0.99 0.001",
    )
    for x in [0.23, 0.61]:
        for y in [-0.57, 0.57]:
            ET.SubElement(
                world,
                "geom",
                name=f"leg_{x}_{y}",
                type="box",
                pos=f"{x} {y} {TABLE_Z / 2 - 0.025}",
                size=f".025 .025 {TABLE_Z / 2 - 0.025}",
                rgba=".16 .19 .23 1",
            )
    parts = [("left", PART_RGBA["blue"]), ("right", PART_RGBA["orange"])]
    for side, color in parts if task_objects else []:
        x, y = SOURCES[side]
        body = ET.SubElement(
            world,
            "body",
            name=side + "_part",
            pos=f"{x} {y} {TABLE_Z + 0.0275 + 0.001}",
        )
        ET.SubElement(body, "freejoint", name=side + "_part_joint")
        ET.SubElement(
            body,
            "geom",
            name=side + "_part",
            type="box",
            size=".0225 .0225 .0275",
            mass="0.07",
            rgba=color,
            friction="1.2 .01 .0001",
            condim="4",
            solref=".004 1",
            solimp=".95 .99 .001",
        )
        # Wide, separated destination pads rather than a tight insertion task.
        x, y = GOALS[side]
        ET.SubElement(
            world,
            "geom",
            name=side + "_target",
            type="box",
            pos=f"{x} {y} {TABLE_Z + 0.003}",
            size=".065 .055 .003",
            rgba=color,
            solref=".004 1",
            solimp=".95 .99 .001",
        )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    # 0.35 mm shadow texels over a box (1.44 m half-size) around the robot, the
    # table and the floor they shade. The default 4096 map over a 4.6 m box gave
    # sawtooth shadow edges, worst under the chest slot; a tighter box with the old
    # key position cut off up to half of the shadows.
    ET.SubElement(visual, "quality", shadowsize="8192")
    ET.SubElement(visual, "map", shadowclip="0.63")
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.30 0.30 0.30",
        diffuse="0.35 0.35 0.35",
        specular="0.1 0.1 0.1",
    )
    return root, report


if __name__ == "__main__":
    root, report = build()
    (ROOT / "outputs").mkdir(exist_ok=True)
    ET.ElementTree(root).write(ROOT / "outputs/task.xml", encoding="unicode")
    (ROOT / "outputs/collision_setup.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
