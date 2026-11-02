"""The robot's MuJoCo model from its URDF (wheel_robot_26.8.16_3, user decision
2026-10-25), as the source project's make_model.py made franzi.xml from the
previous URDF (wheel_robot_4.0): a body per link with the URDF's joint,
inertial and visual mesh, the classes, materials and servos franzi.xml had, a
site and a visual geom per fixed sensor link, and the colliders: each link's
convex hull (sim/link_hulls.py), the wheels' cylinders. build_task.build()
adds the fingers', wrist's and gripper's colliders, the gripper linkage and
the appearance.

The two URDFs share the geometry and the zero posture of the body, legs, arms,
head and wheels (identical meshes; every link within 0.013 mm once the roots
are aligned). The new one re-exports every joint about z, which flips the
sign of five arm joints, and mounts the gripper on its own link, turned 90
degrees about the hand's axis and 12 mm further out. Its root frame lies on
its back (its x points up): BASE_POSE stands it where franzi.xml's base
stood, upright, facing +x, wheels on the floor. FRAMES (franzi_frames.json)
maps each shared link's frame onto the previous one, in which sim/appearance's
recipe is written.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_kinematics import BASE_HEIGHT, ROOT_IN_BASE, SOURCE, URDF
from sim.gripper_linkage import FINGER_ORIGIN, HAND
from sim.link_hulls import hulls as link_hulls

MESHES = SOURCE / "meshes"
# The URDF root in the world, as robot_kinematics places it: the base frame
# BASE_HEIGHT up (franzi.xml's base origin; the wheels stand on the floor), the
# root on the calf axis above it, its x pointing up and its y forward.
BASE_ROT = ROOT_IN_BASE[0]
BASE_POS = np.array([0.0, 0.0, BASE_HEIGHT]) + ROOT_IN_BASE[1]
TABLE = json.loads(Path(__file__).with_name("franzi_frames.json").read_text())


def _maps(group):
    return {
        link: (np.array(v["rotation"], float), np.array(v["translation"], float))
        for link, v in TABLE[group].items()
    }


# previous = R new + t. FRAMES: the links whose meshes are wheel_robot_4.0's,
# in a new frame; GRIPPERS: each new gripper link onto the previous wrist_roll
# frame, through the identical fingers (the gripper is turned 90 degrees and
# 12 mm further out); WRISTS: each new wrist_roll link onto the previous one.
FRAMES, GRIPPERS, WRISTS = _maps("links"), _maps("gripper"), _maps("wrist")
MATERIALS = {
    "franzi_white": {"rgba": "0.85 0.86 0.88 1", "specular": "0.5", "shininess": "0.6"},
    "franzi_joint": {
        "rgba": "0.09 0.095 0.105 1",
        "specular": "0.3",
        "shininess": "0.4",
    },
    "franzi_visor": {
        "rgba": "0.02 0.02 0.025 1",
        "specular": "0.8",
        "shininess": "0.9",
    },
    "franzi_tire": {
        "rgba": "0.04 0.04 0.045 1",
        "specular": "0.05",
        "shininess": "0.05",
    },
}
# franzi.xml's material per visual mesh; links it did not have are dark.
WHITE = {
    "base_link",
    "calf_Link",
    "thigh_Link",
    "torso_Link",
    *(
        f"{s}_{j}_Link"
        for s in ("left", "right")
        for j in ("shoulder_roll", "shoulder_yaw", "wrist_yaw", "wrist_pitch")
    ),
}
TIRES = {f"{u}_wheel_Link" for u in ("left_front", "right_front", "rear")}
# Position servos (kp, force limit) as franzi.xml had them; the wheels are
# velocity servos.
SERVOS = {
    **{
        j: (6000, 800)
        for j in (
            "calf_pitch_joint",
            "thigh_pitch_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        )
    },
    **{j: (400, 100) for j in ("head_yaw_joint", "head_pitch_joint")},
    **{
        f"{s}_{j}_joint": (1500, 100)
        for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")
    },
    **{
        f"{s}_{j}_joint": (400, 100)
        for s in ("left", "right")
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll")
    },
    **{f"{s}_finger0{i}_joint": (2000, 100) for s in ("left", "right") for i in (1, 2)},
    **{
        f"{u}_steering_joint": (400, 100) for u in ("left_front", "right_front", "rear")
    },
}
WHEEL_KV = 20
# Fixed sensor links: a site of franzi.xml's name and a visual geom in the
# parent body, the URDF mass in a child body.
SENSORS = {
    "head_d435_Link": "head_d435",
    "body_d435_Link": "body_d435",
    "left_D405_Link": "left_wrist_d405",
    "right_D405_Link": "right_wrist_d405",
    "MID360_Link": "MID360",
    "2Dlidar_Link": "2Dlidar",
}
# Colliders build_task adds itself: the fingers' meshes, the wrist's and
# gripper's CAD parts.
OWN_COLLIDERS = {
    f"{s}_{p}"
    for s in ("left", "right")
    for p in ("wrist_roll_Link", "finger01_Link", "finger02_Link")
} | {"left_gripper_link", "right_gripper_Link"}
# wheel_robot_4.0's D405 site axes in its wrist_roll frame (franzi.xml): 180
# degrees about x.
OLD_D405_SITE_ROT = np.diag([1.0, -1.0, -1.0])
# Links the URDF gives a mass but no inertia, and the radius they get.
POINT_RADIUS = 0.03
POINT_MASSES = []
# franzi.xml's FMC3 logo plate on the chest, in the previous torso frame.
CHEST_LOGO = ((0.131328, 0.0, 0.294728), (0.0005, 0.08, 0.08))


def to_previous(link):
    """4x4 from link's frame to wheel_robot_4.0's (FRAMES; for a gripper
    link, the previous wrist frame, GRIPPERS), or None for a new link."""
    if link not in FRAMES and link not in GRIPPERS:
        return None
    R, t = FRAMES[link] if link in FRAMES else GRIPPERS[link]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def placement(link, pos=None, quat_wxyz=None):
    """pos and quat attributes that put something given in link's previous
    frame (at pos, quat_wxyz there; default its origin) into link's body."""
    R, t = FRAMES[link]
    Rp = np.eye(3)
    if quat_wxyz is not None:
        w, x, y, z = quat_wxyz
        Rp = Rotation.from_quat([x, y, z, w]).as_matrix()
    p = np.zeros(3) if pos is None else np.asarray(pos, float)
    return {"pos": fmt(R.T @ (p - t)), "quat": fmt(quat(R.T @ Rp))}


def fmt(a):
    return " ".join(f"{v:.9g}" for v in np.asarray(a, float).reshape(-1))


def quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def origin(element):
    """(xyz, rotation) of a URDF origin element (rpy: fixed X, Y, Z)."""
    if element is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array(element.get("xyz", "0 0 0").split(), float)
    rpy = np.array(element.get("rpy", "0 0 0").split(), float)
    return xyz, Rotation.from_euler("xyz", rpy).as_matrix()


def parse():
    """(links, joints by child link) of the URDF."""
    urdf = ET.parse(URDF).getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {joint.find("child").get("link"): joint for joint in urdf.findall("joint")}
    return links, joints


def mesh_file(link):
    mesh = link.find("visual/geometry/mesh")
    return None if mesh is None else Path(mesh.get("filename")).name


def add_inertial(body, link, pos=None, R=None):
    """The URDF inertial of link in body, optionally placed at (pos, R)."""
    inertial = link.find("inertial")
    if inertial is None:
        return
    xyz, rot = origin(inertial.find("origin"))
    i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    tensor = np.array(
        [
            [i["ixx"], i["ixy"], i["ixz"]],
            [i["ixy"], i["iyy"], i["iyz"]],
            [i["ixz"], i["iyz"], i["izz"]],
        ]
    )
    if R is not None:
        xyz, rot = pos + R @ xyz, R @ rot
    t = rot @ tensor @ rot.T
    mass = float(inertial.find("mass").get("value"))
    if np.linalg.eigvalsh(t).min() <= 0:
        # The URDF gives the 2D lidar no inertia: a 3 cm solid sphere's, about
        # what wheel_robot_4.0 gave the same part.
        t = np.eye(3) * 0.4 * mass * POINT_RADIUS**2
        POINT_MASSES.append(link.get("name"))
    ET.SubElement(
        body,
        "inertial",
        pos=fmt(xyz),
        mass=fmt(mass),
        fullinertia=fmt([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]]),
    )


def d405_map(link, xyz, R):
    """The map from a D405 link's frame (at xyz, R in its gripper link) to the
    frame its site and geom take: wheel_robot_4.0's site convention (x along
    the jaws, +z down the hand) at the new link origin. That origin is on the
    face of the gripper opposite the previous mount, so the axes are the
    previous site's carried over by GRIPPERS and turned 180 degrees about the
    hand."""
    side = "left" if link.startswith("left") else "right"
    Rg, _ = next(m for g, m in GRIPPERS.items() if g.startswith(side))
    site = np.diag([-1.0, -1.0, 1.0]) @ Rg.T @ OLD_D405_SITE_ROT
    return site.T @ R, np.zeros(3)


def model():
    """The robot's MJCF tree and a report: what franzi.xml was, from the URDF."""
    links, joints = parse()
    children = {}
    for child, joint in joints.items():
        children.setdefault(joint.find("parent").get("link"), []).append(child)
    hulled = sorted(
        name
        for name in links
        if name not in SENSORS and name not in OWN_COLLIDERS and name not in TIRES
    )
    hulls = link_hulls(hulled)
    root = ET.Element("mujoco", model="franzi")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        balanceinertia="true",
        autolimits="true",
        meshdir=str(MESHES),
        texturedir=str(SOURCE / "assets"),
    )
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    asset = ET.SubElement(root, "asset")
    for name, attrs in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **attrs)
    ET.SubElement(asset, "texture", name="fmc3_logo", type="cube", file="fmc3_logo.png")
    ET.SubElement(
        asset,
        "material",
        name="fmc3_logo",
        texture="fmc3_logo",
        specular="0.3",
        shininess="0.4",
    )
    default = ET.SubElement(
        ET.SubElement(root, "default"), "default", {"class": "franzi"}
    )
    ET.SubElement(
        default, "geom", contype="1", conaffinity="2", friction="0.9 0.005 0.0001"
    )
    ET.SubElement(default, "joint", damping="2", armature="0.1")
    for cls, geom, joint in (
        ("visual", {"contype": "0", "conaffinity": "0", "group": "1"}, None),
        ("collision", {"group": "3", "rgba": "0.7 0.9 0.4 0.4"}, None),
        ("servo", None, {"damping": "5"}),
        (
            "wheel",
            {"friction": "1.2 0.01 0.0001", "group": "3", "rgba": "0.7 0.9 0.4 0.4"},
            {"damping": "0.5", "armature": "0.02"},
        ),
    ):
        node = ET.SubElement(default, "default", {"class": cls})
        if geom:
            ET.SubElement(node, "geom", **geom)
        if joint:
            ET.SubElement(node, "joint", **joint)
    actuators = []

    def visual(body, name, frame=None, geom_name=None, to=None):
        """name's visual mesh in body, whose coordinates sit at frame (R, p) in
        it. Meshes shared with wheel_robot_4.0 (FRAMES), or any given to (R,
        t), are expressed in that frame (refpos, refquat) so that the
        appearance recipe sees the coordinates it was written in; the geom
        frame keeps them in place. Returns the geom frame."""
        frame = frame or (np.eye(3), np.zeros(3))
        mesh = mesh_file(links[name])
        if mesh is None:
            return frame
        mapping = to or FRAMES.get(name)
        attrs = {"name": name, "file": mesh}
        if mapping is not None:
            Rm, tm = mapping
            attrs.update(refpos=fmt(-Rm.T @ tm), refquat=fmt(quat(Rm.T)))
            frame = (frame[0] @ Rm.T, frame[1] - frame[0] @ Rm.T @ tm)
        if asset.find(f"mesh[@name='{name}']") is None:
            ET.SubElement(asset, "mesh", attrs)
        material = (
            "franzi_visor"
            if name == "head_pitch_Link"
            else "franzi_white"
            if name in WHITE
            else "franzi_tire"
            if name in TIRES
            else "franzi_joint"
        )
        R, pos = frame
        placed = {}
        if not (np.allclose(R, np.eye(3)) and np.allclose(pos, 0)):
            placed = {"pos": fmt(pos), "quat": fmt(quat(R))}
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name or f"{name}_visual",
                "type": "mesh",
                "mesh": name,
                "material": material,
                "class": "visual",
                **placed,
            },
        )
        return frame

    def collider(body, name):
        if name in hulls:
            hull = f"{name}_hull"
            ET.SubElement(asset, "mesh", name=hull, vertex=fmt(hulls[name][0]))
            ET.SubElement(
                body,
                "geom",
                name=name,
                type="mesh",
                mesh=hull,
                **{"class": "collision"},
            )

    def build(body, name):
        hand = None
        if name in GRIPPERS:
            # The gripper in a welded body in wheel_robot_4.0's wrist frame,
            # where the fingers and the linkage (sim/gripper_linkage.py) live.
            side = "left" if name.startswith("left") else "right"
            Rg, tg = GRIPPERS[name]
            hand = ET.SubElement(
                body,
                "body",
                name=HAND.format(side=side),
                pos=fmt(-Rg.T @ tg),
                quat=fmt(quat(Rg.T)),
            )
            visual(hand, name, (Rg, tg), to=GRIPPERS[name])
        else:
            visual(body, name)
            collider(body, name)
        if name == "torso_Link":
            R, t = FRAMES[name]  # previous = R new + t
            pos = R.T @ (np.array(CHEST_LOGO[0]) - t)
            ET.SubElement(
                body,
                "geom",
                {"class": "visual"},
                name="chest_logo",
                type="box",
                size=fmt(CHEST_LOGO[1]),
                pos=fmt(pos),
                quat=fmt(quat(R.T)),
                material="fmc3_logo",
            )
        for child in children.get(name, []):
            joint = joints[child]
            xyz, R = origin(joint.find("origin"))
            if child in SENSORS:
                to = d405_map(child, xyz, R) if "D405" in child else None
                site = visual(body, child, (R, xyz), geom_name=child, to=to)
                ET.SubElement(
                    body,
                    "site",
                    name=SENSORS[child],
                    pos=fmt(site[1]),
                    quat=fmt(quat(site[0])),
                    size="0.005",
                )
                mass = ET.SubElement(
                    body, "body", name=child, pos=fmt(xyz), quat=fmt(quat(R))
                )
                add_inertial(mass, links[child])
                continue
            if hand is not None and joint.get("type") == "prismatic":
                # A finger: at wheel_robot_4.0's origin in the hand, the URDF's
                # slide (name, range, sign) along the same line.
                n = 1 if child.endswith("01_Link") else 2
                Rf, tf = FRAMES[child]
                limit = joint.find("limit")
                stroke = f"{limit.get('lower')} {limit.get('upper')}"
                axis = Rf @ np.array(joint.find("axis").get("xyz").split(), float)
                finger = ET.SubElement(
                    hand, "body", name=child, pos=fmt(FINGER_ORIGIN[side][n])
                )
                ET.SubElement(
                    finger,
                    "joint",
                    {"name": joint.get("name"), "type": "slide", "axis": fmt(axis)}
                    | {"range": stroke, "class": "servo"},
                )
                actuators.append((joint.get("name"), stroke))
                add_inertial(finger, links[child], tf, Rf)
                visual(finger, child, (Rf, tf))
                continue
            sub = ET.SubElement(
                body, "body", name=child, pos=fmt(xyz), quat=fmt(quat(R))
            )
            kind = joint.get("type")
            if kind != "fixed":
                attrs = {
                    "name": joint.get("name"),
                    "axis": joint.find("axis").get("xyz"),
                }
                limit = joint.find("limit")
                if kind == "continuous":
                    attrs.update(type="hinge", **{"class": "wheel"})
                else:
                    attrs.update(
                        type="slide" if kind == "prismatic" else "hinge",
                        range=f"{limit.get('lower')} {limit.get('upper')}",
                        **{"class": "servo"},
                    )
                ET.SubElement(sub, "joint", attrs)
                actuators.append((joint.get("name"), attrs.get("range")))
            add_inertial(sub, links[child])
            if child in TIRES:
                visual(sub, child)
                ET.SubElement(
                    sub,
                    "geom",
                    {"class": "wheel"},
                    name=child,
                    type="cylinder",
                    size="0.0826648 0.06475",
                    pos=fmt(
                        [
                            0,
                            0.01875 if child != "right_front_wheel_Link" else -0.01875,
                            0,
                        ]
                    ),
                    quat="0.707107 0.707107 0 0",
                )
                continue
            build(sub, child)

    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(
        world,
        "body",
        name="base_link",
        pos=fmt(BASE_POS),
        quat=fmt(quat(BASE_ROT)),
        childclass="franzi",
    )
    add_inertial(base, links["base_link"])
    build(base, "base_link")
    act = ET.SubElement(root, "actuator")
    for name, ctrlrange in actuators:
        if name in SERVOS:
            kp, force = SERVOS[name]
            ET.SubElement(
                act,
                "position",
                name=name,
                joint=name,
                kp=str(kp),
                ctrlrange=ctrlrange,
                forcerange=f"-{force} {force}",
            )
        else:
            ET.SubElement(
                act,
                "velocity",
                name=name,
                joint=name,
                kv=str(WHEEL_KV),
                ctrlrange="-1000 1000",
                forcerange="-100 100",
            )
    report = {
        "urdf": URDF.name,
        "inertia_from_sphere": sorted(set(POINT_MASSES)),
        "base": {"pos": list(BASE_POS), "quat": quat(BASE_ROT)},
        "link_collision": {
            "method": "convex hull of each link STL, cut to at most 255 vertices "
            "(sim/link_hulls.py); the wheels' cylinders",
            "links": len(hulls),
            "max_cut_inside_full_hull_mm": round(
                max(e for _, e in hulls.values()) * 1000, 3
            ),
        },
    }
    return root, report
