"""Derive the dual-arm robot scene (the robot, its table and the lights) without
modifying the source model: the base fixed, or on planar joints (mobile=True).
Each scene in scenes/ adds its own objects to it."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.transform import Rotation

from robots.franzi import model as franzi_model
from robots.franzi.appearance import apply_appearance
from robots.franzi.gripper_linkage import (
    HAND,
    add_linkage,
    body_name,
    face_table,
    offset_in_wrist,
)
from robots.franzi.kinematics import (
    ARM_JOINTS,
    ROBOT_CONFIG,
    SOURCE,
    URDF,
    controller_ranges,
)
from robots.franzi.wrist_mount import PAD, block_pieces, sensor_pose
from robots.franzi.wrist_mount import P as MOUNT
from sim import ROOT, TABLE_Z, fmt

__all__ = ["ARM_JOINTS", "FINGERS"]  # re-exported: one definition, in kinematics
# The optional mobile base (build(mobile=True); navigation plan, user decision
# 2026-10-24): x and y slides and a yaw hinge on base_link, in that order, so x
# and y are world axes and the yaw turns about the base's centre. Each is a
# position servo; the chassis SDK takes velocities (publish_vehicle_control),
# which Scene.drive() keeps within. The base rides BASE_LIFT up, so its
# colliders clear the floor: the servos move it, not wheel friction.
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
BASE_LIFT = 0.002  # m
BASE_KP = {"base_x": 30000.0, "base_y": 30000.0, "base_yaw": 3000.0}  # N/m, N m/rad
# Each hand's fingers: the one that closes with a positive slide, then the
# other (robots/franzi/config/franzi_sdk.yaml grippers); a grip g commands +-0.0475 g.
FINGERS = {
    side: tuple(unit["finger_links"])
    for side, unit in yaml.safe_load(ROBOT_CONFIG.read_text())["grippers"].items()
}
FINGER_JOINTS = {
    side: tuple(unit["sim_joints"])
    for side, unit in yaml.safe_load(ROBOT_CONFIG.read_text())["grippers"].items()
}
# The space between the open jaws that no gripper collider may hold, in the
# tool frame (m): along the jaws, across them, and from below the pads to the
# palm.
JAW_CLEAR = ((-0.035, 0.035), (-0.012, 0.012), (-0.02, 0.035))
# Arm-trunk pairs left out of the self-collision (build). The shoulder roll
# links sit in the torso's shoulder recess, which its hull fills: over random
# arm postures their hulls overlapped in 12% by up to 35 mm, their meshes by
# about 2 mm, at the inward roll limit. wheel_robot_26.8.16_3 adds the shoulder
# mounts (j0) between them, whose hulls overlap the roll links' in 7% of random
# postures by up to 13 mm while the meshes stay 10 mm apart. The upper arm's
# overlaps with the torso are real (meshes 30-57 mm deep) and stay.
SELF_EXCLUDES = (
    ("left_shoulder_roll_Link", "torso_Link"),
    ("right_shoulder_roll_Link", "torso_Link"),
    ("left_shoulder_roll_Link", "left_j0_Link"),
    ("right_shoulder_roll_Link", "right_j0_Link"),
)


def align_joint_limits(root):
    """Set joint ranges, and the matching actuator ctrlranges, to the controller-
    aligned ranges of robots.franzi.kinematics.controller_ranges (the IK's bounds)."""
    ranges = controller_ranges()
    joints = {j.get("name"): j for j in root.iter("joint")}
    actuators = {a.get("joint"): a for a in root.find("actuator") if a.get("joint")}
    for name, r in ranges.items():
        joints[name].set("range", fmt(r["aligned_rad"]))
        if name in actuators and actuators[name].get("ctrlrange"):
            actuators[name].set("ctrlrange", fmt(r["aligned_rad"]))
    return ranges


def add_mobile_base(root, base):
    """Give base_link the planar holonomic joints and their servos (BASE_JOINTS).
    Joint axes are in the body's frame, which wheel_robot_26.8.16_3 turns
    (BASE_ROT): written as 1 0 0 there, base_x ran vertically and the robot
    sank 37 mm on its servo onto the floor."""
    base.set("pos", fmt(np.array(base.get("pos").split(), float) + [0, 0, BASE_LIFT]))
    world = {"base_x": [1, 0, 0], "base_y": [0, 1, 0], "base_yaw": [0, 0, 1]}
    for k, name in enumerate(BASE_JOINTS):
        kind = "hinge" if name == "base_yaw" else "slide"
        axis = np.round(franzi_model.BASE_ROT.T @ world[name], 9) + 0.0
        joint = ET.Element(
            "joint",
            name=name,
            type=kind,
            axis=fmt(axis),
            limited="false",
            damping="0",
            armature="0",
            frictionloss="0",
            stiffness="0",
        )
        base.insert(k, joint)
    actuators = root.find("actuator")
    for name in BASE_JOINTS:
        ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=str(BASE_KP[name]),
            dampratio="1",
            ctrllimited="false",
            forcelimited="false",
        )


def part_hulls(asset, body, stl, prefix, faces=None, clear=None, shift=None):
    """Convex colliders in body for stl (a trimesh in body's frame, less shift),
    or for its faces: one per connected CAD part, as a single hull would fill a
    gripper's jaw opening. A part whose hull holds any of the points clear is
    cut in two at the median of its longest side, again until none does: the
    gripper's linkage arms and body are one connected surface across the jaws.
    Returns the count."""
    if faces is not None:
        stl = stl.submesh([faces], append=True)
    if shift is not None:
        stl = stl.copy()
        stl.apply_translation(-np.asarray(shift, float))
    edges = stl.edges_unique
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(stl.vertices),) * 2,
    )
    count, labels = connected_components(graph, directed=False)
    pieces = []

    def holds(vertices):
        if clear is None:
            return False
        try:
            return bool((Delaunay(vertices).find_simplex(clear) >= 0).any())
        except QhullError:
            return False

    def cut(faces, depth=0):
        vertices = stl.vertices[np.unique(stl.faces[faces])]
        if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices.mean(0)) < 3:
            return
        if depth < 12 and len(faces) > 1 and holds(vertices):
            centres = stl.triangles_center[faces]
            axis = int(np.argmax(np.ptp(vertices, axis=0)))
            low = centres[:, axis] <= np.median(centres[:, axis])
            if low.any() and not low.all():
                cut(faces[low], depth + 1)
                cut(faces[~low], depth + 1)
                return
        pieces.append(vertices)

    by_face = labels[stl.faces[:, 0]]
    for c in range(count):
        cut(np.flatnonzero(by_face == c))
    parts = 0
    for vertices in pieces:
        hull = trimesh.convex.convex_hull(vertices)
        if hull.volume < 1e-12:
            continue
        name = f"{prefix}_{parts:03d}"
        ET.SubElement(
            asset, "mesh", name=name, vertex=fmt(hull.vertices), face=fmt(hull.faces)
        )
        ET.SubElement(
            body, "geom", name=name, type="mesh", mesh=name, **{"class": "collision"}
        )
        parts += 1
    return parts


def load_stl(name):
    return trimesh.load(SOURCE / "meshes" / name, process=True, force="mesh")


def jaw_clearance(position, rotation):
    """Grid points of the space the open jaws close on (JAW_CLEAR, the tool
    frame), in the gripper link frame where the tool point sits at position
    with rotation."""
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(JAW_CLEAR, (15, 7, 12))]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    with np.errstate(all="ignore"):  # macOS Accelerate's matmul warns spuriously
        return np.asarray(position) + grid @ np.asarray(rotation).T


def build(mobile=False):
    """Derived scene tree and report: the robot (robots/franzi/model.py), the
    table and the lights. A scene's task.py adds its objects
    (scenes/<name>/task.py). mobile=True gives the base its planar joints
    (add_mobile_base); the default keeps it fixed."""
    root, model = franzi_model.model()
    option = root.find("option")
    option.set("timestep", str(1 / 500))
    option.set("iterations", "100")
    option.set("cone", "elliptic")
    base = root.find(".//body[@name='base_link']")
    if mobile:
        add_mobile_base(root, base)
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
        "source": URDF.name,
        "base": (
            "planar holonomic base: x, y slides and yaw hinge, position servos, "
            f"{BASE_LIFT * 1000:g} mm above the floor"
            if mobile
            else "fixed to floor for arm-only experiment"
        ),
        "objects_pose_written_during_episode": False,
        "attachment_constraints": False,
        "link_collision": model["link_collision"],
        "wrist_collision": {},
        "finger_collision": {},
    }
    report["joint_limits"] = align_joint_limits(root)
    # Before the appearance and the colliders, which put the links and jaw
    # blocks into the bodies this adds.
    report["gripper_linkage"] = add_linkage(root, fmt)
    report["appearance"] = apply_appearance(root, SOURCE)
    tcp = yaml.safe_load(ROBOT_CONFIG.read_text())["frames"]["tcp"]
    for side in ["left", "right"]:
        gripper = tcp["parent_link"][side]
        # The wrist flange and the gripper: one convex collider per CAD part,
        # which keeps the jaw opening clear. The gripper's links and jaw blocks
        # collide in the bodies that move them (gripper_linkage.face_table),
        # the rest in its hand body, in wheel_robot_4.0's wrist frame.
        wrist = root.find(f".//body[@name='{side}_wrist_roll_Link']")
        hand_name = HAND.format(side=side)
        hand = root.find(f".//body[@name='{hand_name}']")
        stl = load_stl(f"{gripper}.STL")
        stl.apply_transform(franzi_model.to_previous(gripper))
        moving = face_table(SOURCE / "meshes" / f"{gripper}.STL")[0]
        assert len(moving) == len(stl.faces), gripper
        Rg, tg = franzi_model.GRIPPERS[gripper]
        clear = jaw_clearance(
            Rg @ tcp["position_m"][side] + tg, Rg @ np.array(tcp["rotation"][side])
        )
        parts = {
            "wrist_parts": part_hulls(
                asset,
                wrist,
                load_stl(f"{side}_wrist_roll_Link.STL"),
                f"{side}_wrist_part",
            ),
            "gripper_parts": part_hulls(
                asset,
                hand,
                stl,
                f"{side}_gripper_part",
                faces=np.flatnonzero(moving == None),
                clear=clear,
            ),
            "on_moving_links_and_jaws": 0,
        }
        for group in sorted(set(moving) - {None}):
            target = body_name(side, group)
            parts["on_moving_links_and_jaws"] += part_hulls(
                asset,
                root.find(f".//body[@name='{target}']"),
                stl,
                f"{side}_gripper_{group}_part",
                faces=np.flatnonzero(moving == group),
                shift=offset_in_wrist(root, side, target),
            )
        report["wrist_collision"][side] = parts
        for finger in FINGERS[side]:
            body = root.find(f".//body[@name='{finger}']")
            # The finger body is in its wheel_robot_4.0 frame, as the mesh asset
            # (robots/franzi/model.py).
            ET.SubElement(
                body,
                "geom",
                name=finger,
                type="mesh",
                mesh=finger,
                friction="1.2 0.01 0.0001",
                condim="4",
                solref="0.004 1",
                solimp="0.95 0.99 0.001",
                **{"class": "collision"},
            )
            mesh = trimesh.load(
                SOURCE / "meshes" / f"{finger}.STL", process=True, force="mesh"
            )
            report["finger_collision"][finger] = {
                "method": "convex hull of the finger STL; flat grasp face preserved",
                "bounds_m": mesh.bounds.tolist(),
                "vertices": len(mesh.vertices),
                "hull_to_mesh_volume_ratio": float(
                    mesh.convex_hull.volume / abs(mesh.volume)
                ),
            }
        # The wrist D405 adapter (robots/franzi/wrist_mount.py) on the gripper's camera
        # strip: three convex pieces, and a box for the D405 it holds, where the
        # camera and the D405 mesh are (camera_rig, appearance).
        body = root.find(f".//body[@name='{gripper}']")
        for k, piece in enumerate(block_pieces()):
            hull = trimesh.convex.convex_hull((piece + PAD) / 1000)
            block = f"{side}_wrist_d405_block_{k}"
            ET.SubElement(
                asset,
                "mesh",
                name=block,
                vertex=fmt(hull.vertices),
                face=fmt(hull.faces),
            )
            ET.SubElement(
                body,
                "geom",
                name=block,
                type="mesh",
                mesh=block,
                **{"class": "collision"},
            )
        site = body.find(f".//site[@name='{side}_wrist_d405']")
        w, x, y, z = map(float, site.get("quat").split())
        at, turned = sensor_pose(side, Rotation.from_quat([x, y, z, w]).as_matrix())
        half = np.array(MOUNT["d405"]["housing_mm"]) / 2000
        x, y, z, w = Rotation.from_matrix(turned).as_quat()
        ET.SubElement(
            body,
            "geom",
            name=f"{side}_wrist_d405_housing",
            type="box",
            size=fmt(half),
            pos=fmt(at - turned @ [0, 0, half[2]]),  # behind the glass
            quat=fmt([w, x, y, z]),
            **{"class": "collision"},
        )
        parts["d405_adapter_pieces"] = len(block_pieces())
        # The tool point: the open pads' centre, x along the jaws' closing, z
        # toward the wrist (config/franzi_sdk.yaml, as the kinematics).
        R = np.array(tcp["rotation"][side], float)
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        ET.SubElement(
            root.find(f".//body[@name='{gripper}']"),
            "site",
            name=f"{side}_grasp",
            pos=fmt(tcp["position_m"][side]),
            quat=fmt([w, x, y, z]),
            size="0.004",
            group="4",
        )
    # Contact bits: the world collides with everything (contype and conaffinity
    # 1); each arm with the other arm (bits 2 and 3: left 5/10, right 9/6) and,
    # since 2026-10-25 (self-collision, as RoboDojo's enabled_self_collisions),
    # with the trunk: base, legs, waist, torso and its shoulder mounts (j0), head
    # and wheel units, whose conaffinity takes both arm bits (14). Contacts within
    # one arm and within the trunk stay off; SELF_EXCLUDES drops the pairs that
    # meet at the joints. The wheels never touch the world: the fixed base is
    # welded, and the mobile one rides BASE_LIFT up.
    for body in root.findall(".//body"):
        name = body.get("name", "")
        side = None
        if "front" not in name and "j0" not in name:
            side = (
                "left"
                if name.startswith("left")
                else "right"
                if name.startswith("right")
                else None
            )
        for geom in body.findall("geom"):
            if geom.get("class") not in ("collision", "wheel"):
                continue
            if side:
                geom.set("contype", "5" if side == "left" else "9")
                geom.set("conaffinity", "10" if side == "left" else "6")
            else:
                geom.set("conaffinity", "14")
                if geom.get("class") == "wheel":
                    geom.set("contype", "0")
    contact = ET.SubElement(root, "contact")
    for a, b in SELF_EXCLUDES:
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    report["self_collision"] = {
        "pairs": "left arm-right arm, each arm-trunk; none within an arm or the trunk",
        "excluded": [list(pair) for pair in SELF_EXCLUDES],
    }
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
    ET.ElementTree(root).write(ROOT / "outputs/robot_scene.xml", encoding="unicode")
    (ROOT / "outputs/collision_setup.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
