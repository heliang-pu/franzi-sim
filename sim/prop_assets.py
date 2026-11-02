"""Benchmark props built in Blender (assets/props/<id>/, docs/benchmark_props.md):
add a prop's parts to a scene tree as mesh assets, textured materials, bodies,
joints and collision geoms.

Every part is exported in the prop frame, so each part body sits at its
parent's origin and prop.json's joint anchors and axes apply as they are. The
root part's body is <name>_part and the others <name>_<part>_part: the suffix
keeps props out of Scene.robot_bodies. Visual geoms (group 2) carry the baked
texture and neither mass nor contacts. Collision pieces (group 3, hidden) carry
the part's friction and its mass, spread over them by convex volume; MuJoCo
collides each piece as its convex hull. Joints take prop.json's axis, anchor,
range (limits as stiff as the contacts), damping, frictionloss, spring and
frictionloss solver settings (solreffriction/solimpfriction, e.g. the lidded
tote's static hold). A part's contact_priority makes its friction win over what
touches it (the flow lane's slide); detents,
latches and similar behaviours belong to the scene. A jointed part starts at
its recommended_initial_qpos (or the value add() is given): its body is posed
there and the joint's ref makes that pose qpos0, so ranges and springs keep the
prop's coordinates and nothing writes qpos at run time.
"""

import json
import xml.etree.ElementTree as ET
from functools import cache

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from sim.build_task import ROOT, fmt

PROPS = ROOT / "assets/props"
# The existing scenes' contact parameters for parts and surfaces.
SOLREF, SOLIMP = "0.004 1", "0.95 0.99 0.001"
ARMATURE = 0.0005  # keeps light jointed parts (buttons, knobs) stable at 500 Hz
# Joint limits are as stiff as the contacts. With MuJoCo's default (0.02 s,
# impedance 0.9-0.95), a 20 g button cap pushed with 10 N ran several times past
# its 4-6 mm travel, and a pushed box lid past its open stop (2026-10-25).
LIMIT_SOLREF, LIMIT_SOLIMP = SOLREF, SOLIMP


@cache
def spec(prop_id):
    return json.loads((PROPS / prop_id / "prop.json").read_text())


@cache
def _mesh(path):
    return trimesh.load(path, force="mesh")


def variant(prop_id, name):
    for v in spec(prop_id).get("variants", []):
        if v.get("name") == name:
            return v
    raise KeyError(f"{prop_id} has no variant {name!r}")


def _ensure(asset, tag, name, **attrs):
    if asset.find(f"{tag}[@name='{name}']") is None:
        ET.SubElement(asset, tag, name=name, **attrs)
    return name


def _material(asset, prop_id, part, texture):
    folder = PROPS / prop_id
    look = part.get("mujoco_material") or {}
    attrs = {
        k: str(look[k]) for k in ("specular", "shininess", "reflectance") if k in look
    }
    if part.get("emission"):
        attrs["emission"] = str(part["emission"])
    if texture:
        tex = _ensure(
            asset,
            "texture",
            f"tex_{prop_id}_{texture[:-4]}",
            type="2d",
            file=str(folder / texture),
        )
        return _ensure(
            asset,
            "material",
            f"mat_{prop_id}_{part['name']}_{texture[:-4]}",
            texture=tex,
            **attrs,
        )
    rgba = part.get("rgba")
    if rgba is None and part.get("base_color"):
        h = part["base_color"].lstrip("#")
        rgba = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1]
    return _ensure(
        asset,
        "material",
        f"mat_{prop_id}_{part['name']}",
        rgba=fmt(rgba or [0.7, 0.7, 0.7, 1]),
        **attrs,
    )


def add(
    root,
    prop_id,
    name,
    pos=(0, 0, 0),
    quat=(1, 0, 0, 0),
    *,
    parent=None,
    free=False,
    variant_name=None,
    mass=None,
    initial=None,
):
    """Add instance `name` of a prop under `parent` (a body element; the world
    by default) at pos/quat. free=True gives the root part a freejoint (props
    with one root part only). mass overrides the root part's mass. initial
    {part name: joint value} sets where jointed parts start, over their
    recommended_initial_qpos. Returns {part name: body name}."""
    s = spec(prop_id)
    initial = initial or {}
    v = variant(prop_id, variant_name) if variant_name else {}
    parts = [
        p
        for p in (v.get("parts") or s["parts"])
        if p["name"] not in set(v.get("disabled_parts", []))
    ]
    textures = v.get("textures", {})
    masses = dict(v.get("part_masses_kg", {}))
    asset = root.find("asset")
    holder = parent if parent is not None else root.find("worldbody")
    roots = [p for p in parts if not p.get("parent")]
    if free and len(roots) != 1:
        raise ValueError(f"{prop_id}: a free prop needs exactly one root part")
    if mass is not None:
        masses[roots[0]["name"]] = mass
    names = {}
    if len(roots) == 1:
        names[roots[0]["name"]] = f"{name}_part"
        base = None
    else:
        base = ET.SubElement(
            holder, "body", name=f"{name}_part", pos=fmt(pos), quat=fmt(quat)
        )
    for p in parts:
        names.setdefault(p["name"], f"{name}_{p['name']}_part")
    bodies = {}
    pending = list(parts)
    while pending:  # parents before children
        for p in list(pending):
            up = p.get("parent")
            if up and up not in bodies:
                continue
            pending.remove(p)
            if up:
                body = ET.SubElement(bodies[up], "body", name=names[p["name"]])
            elif base is not None:
                body = ET.SubElement(base, "body", name=names[p["name"]])
            else:
                body = ET.SubElement(
                    holder, "body", name=names[p["name"]], pos=fmt(pos), quat=fmt(quat)
                )
                if free:
                    # The name sim.tabletop.ObjectScene looks up.
                    ET.SubElement(body, "freejoint", name=f"{name}_part_joint")
            bodies[p["name"]] = body
            _fill(
                body,
                asset,
                prop_id,
                name,
                p,
                textures.get(p["name"], p.get("texture")),
                masses.get(p["name"], p.get("mass_kg", 0.0)),
                moving=free or bool(p.get("joint")),
                start=initial.get(p["name"], p.get("recommended_initial_qpos")),
            )
    excludes = s.get("collision_exclusions", [])
    if excludes:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for a, b in excludes:
            if a in names and b in names:
                ET.SubElement(contact, "exclude", body1=names[a], body2=names[b])
    return names


def set_geoms(root, prefix, **attrs):
    """Set attributes on every geom whose name starts with prefix, e.g. a
    contact priority so that a prop's own friction governs its contacts."""
    hits = [g for g in root.iter("geom") if (g.get("name") or "").startswith(prefix)]
    if not hits:
        raise KeyError(f"no geom named {prefix}*")
    for geom in hits:
        for k, v in attrs.items():
            geom.set(k, str(v))
    return len(hits)


def joint_anchor(prop_id, part):
    """A part's joint anchor in the prop frame (m)."""
    for p in spec(prop_id)["parts"]:
        if p["name"] == part:
            return np.array(p["joint"]["anchor_m"], float)
    raise KeyError(f"{prop_id} has no part {part!r}")


def _start_at(body, joint, value):
    """Pose a jointed part's body at joint value `value`: turned about the
    joint's anchor (hinge) or moved along its axis (slide), in the frame of
    the body's own pos/quat. The anchor and axis keep their body coordinates."""
    axis = np.asarray(joint["axis"], float)
    axis /= np.linalg.norm(axis)
    if joint["type"] == "hinge":
        turn = Rotation.from_rotvec(value * axis)
        anchor = np.asarray(joint.get("anchor_m", [0, 0, 0]), float)
        shift = anchor - turn.apply(anchor)
    elif joint["type"] == "slide":
        turn, shift = Rotation.identity(), value * axis
    else:
        raise ValueError(f"no start value for a {joint['type']} joint")
    pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
    w, x, y, z = (float(v) for v in body.get("quat", "1 0 0 0").split())
    frame = Rotation.from_quat([x, y, z, w])
    x, y, z, w = (frame * turn).as_quat()
    body.set("pos", fmt(pos + frame.apply(shift)))
    body.set("quat", fmt([w, x, y, z]))


def _fill(body, asset, prop_id, name, part, texture, mass, moving, start=None):
    folder = PROPS / prop_id
    pname = part["name"]
    joint = part.get("joint")
    if joint:
        attrs = {
            "name": f"{name}_{pname}",
            "type": joint["type"],
            "axis": fmt(joint["axis"]),
            "pos": fmt(joint.get("anchor_m", [0, 0, 0])),
            "damping": str(joint.get("damping", 0)),
            "frictionloss": str(joint.get("frictionloss", 0)),
            "armature": str(ARMATURE),
        }
        for key in ("solreffriction", "solimpfriction"):
            if key in joint:
                attrs[key] = fmt(joint[key])
        if start:
            _start_at(body, joint, start)
            attrs["ref"] = str(start)
        if (
            joint.get("range") is not None
            and joint.get("limited", True)
            and not joint.get("continuous")
        ):
            attrs.update(
                limited="true",
                range=fmt(joint["range"]),
                solreflimit=LIMIT_SOLREF,
                solimplimit=LIMIT_SOLIMP,
            )
        else:
            attrs["limited"] = "false"
        spring = joint.get("spring")
        if spring:
            attrs.update(
                stiffness=str(spring["stiffness"]),
                springref=str(spring.get("reference", 0)),
            )
        ET.SubElement(body, "joint", **attrs)
    if part.get("visual"):
        mesh = _ensure(
            asset, "mesh", f"vis_{prop_id}_{pname}", file=str(folder / part["visual"])
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_{pname}_visual",
            type="mesh",
            mesh=mesh,
            material=_material(asset, prop_id, part, texture),
            contype="0",
            conaffinity="0",
            group="2",
            mass="0",
        )
    pieces = part.get("collision") or []
    volumes = [_mesh(str(folder / f)).convex_hull.volume for f in pieces]
    density = mass / sum(volumes) if mass and sum(volumes) > 0 else 1000.0
    friction = f"{part.get('friction', 0.5)} 0.005 0.0001"
    contact = {
        "friction": friction,
        "solref": SOLREF,
        "solimp": SOLIMP,
        "condim": "4" if moving else "3",
    }
    if part.get("contact_priority"):
        # This part's friction wins over what touches it (the flow lane's low
        # friction); MuJoCo otherwise takes the larger of the two.
        contact["priority"] = str(part["contact_priority"])
    for k, f in enumerate(pieces):
        mesh = _ensure(
            asset, "mesh", f"col_{prop_id}_{pname}_{k:02d}", file=str(folder / f)
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_{pname}_col{k:02d}",
            type="mesh",
            mesh=mesh,
            group="3",
            rgba="0.85 0.35 0.3 0.4",
            density=str(density),
            **contact,
        )
    for k, prim in enumerate(part.get("collision_primitives") or []):
        if prim["type"] == "plane":
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_{pname}_plane{k}",
                type="plane",
                pos=fmt(prim.get("pos_m", [0, 0, 0])),
                size=fmt(prim.get("size_m", [3, 3, 0.01])),
                group="3",
                rgba="0 0 0 0",
                **contact,
            )
    if moving and not pieces:
        # A moving part without colliders still needs mass: a box over its mesh.
        lo, hi = _mesh(str(folder / part["visual"])).bounds
        size = np.maximum(hi - lo, 1e-3)
        m = max(mass, 1e-3)
        inertia = (
            m
            / 12
            * np.array(
                [
                    size[1] ** 2 + size[2] ** 2,
                    size[0] ** 2 + size[2] ** 2,
                    size[0] ** 2 + size[1] ** 2,
                ]
            )
        )
        ET.SubElement(
            body,
            "inertial",
            pos=fmt((lo + hi) / 2),
            mass=str(m),
            diaginertia=fmt(inertia),
        )
