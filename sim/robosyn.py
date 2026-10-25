"""RoboSynChallenge props for the scenes: each prop scaled to the size the
scenes use, upright with its base at z = 0 and its axis at x = y = 0, with flat
or sampled colours and convex colliders (see sim/props).

The raw files are the RoboSynChallenge repository's assets/ (Apache-2.0) and
EmbodiChain's data packs (the plate, the drawer), copied unmodified into
assets/robosyn/ (sim.props.RAW, ignored by git). prepare(out_dir, names) makes
the props a scene asks for and caches them in its out_dir.

Scales are against the raw files and depart from RoboSyn's own where this
robot needs it, as each maker says: the gripper housing spans about 85 mm
either side of the grasp point and 65 mm behind it, the jaws open 95 mm, and
the fingertips reach 17 mm past the grasp point.
"""

import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from sim import props as pp
from sim.build_task import TABLE_Z, fmt

SOURCES = {
    "pot": "robosyn_assets/Pan/pot.ply",
    "lid": "robosyn_assets/Pan/lid_remeshed.ply",
    "tomato": "robosyn_assets/tomato/tomato_asset_simready/asset_simready.obj",
    "apple": "robosyn_assets/Apple/apple.ply",
    "plate": "embodichain_data/TableWare/tableware/plate/3_center.ply",
    "button_base": "robosyn_assets/button/Visual/button_base.obj",
    "button_cap": "robosyn_assets/button/Visual/button_cover.obj",
    "beaker": "robosyn_assets/Beaker/beaker.ply",
    "cup_red": "robosyn_assets/PaperCup/red_cup.ply",
    "cup_green": "robosyn_assets/PaperCup/green_cup.ply",
    "cup_blue": "robosyn_assets/PaperCup/blue_cup.ply",
    "drawer_outer": "embodichain_data/SlidingBoxDrawer/OuterBox.stl",
    "drawer_inner": "embodichain_data/SlidingBoxDrawer/InnerBox.stl",
}
VERSION = 3
STEEL = [0.70, 0.71, 0.73, 1]
BLACK = [0.07, 0.07, 0.08, 1]


def make_pot(export):
    """RoboSyn's saucepan, handle turned to +x. RoboSyn's open_pan scales it
    (0.001, 0.001, 0.000675); here it is 10% wider, so the gripper housing fits
    inside it beside an ingredient."""
    mesh = pp.load(SOURCES["pot"], (0.0011, 0.0011, 0.000675))
    mesh.apply_transform(pp.rot_z(np.pi))
    pp.settle(mesh, centre=[0, 0, 0])
    v = mesh.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    h = mesh.bounds[1][2]
    # Every vertex of the body lies on its wall: the bottom is one polygon
    # spanning the lowest ring, the wall long faces up to the rim ring.
    top = (v[:, 2] > 0.85 * h) & (r < 0.13)
    inner, outer = np.percentile(r[top], 5), np.percentile(r[top], 95)
    handle = v[:, 0] > outer + 0.002
    low = (v[:, 2] < 0.2 * h) & ~handle
    floor = float(v[low & (r < inner + 0.003), 2].max())
    bottom = low & (v[:, 2] <= floor + 0.0005)
    hv = v[handle]
    grip = mesh.vertices[mesh.faces][:, :, 0].mean(1) > outer + 0.004
    handle_mesh, body = pp.split_faces(mesh, grip)
    return {
        "visual": [
            {"file": export(body, "pot_body"), "rgba": STEEL},
            {"file": export(handle_mesh, "pot_handle"), "rgba": BLACK},
        ],
        "collision": [pp.hull(v[bottom])] + pp.ring(v[~handle]) + [pp.hull(hv)],
        "dims": {
            "inner_radius": float(inner),
            "outer_radius": float(outer),
            "floor_z": floor,
            "rim_z": float(h),
            "handle_x": [float(hv[:, 0].min()), float(hv[:, 0].max())],
        },
    }


def make_lid(export):
    """The pot's lid: a disk with a ring knob on top, collided as the two.
    RoboSyn scales it (0.315, 0.315, 0.42); here it widens with the pot and its
    knob is taller, so the fingertips clear the lid while the pads hold the
    knob."""
    mesh = pp.settle(pp.load(SOURCES["lid"], (0.3465, 0.3465, 0.5)))
    v = mesh.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    # The knob starts where the lid narrows below the knob's radius.
    zs = np.linspace(0, mesh.bounds[1][2], 200)
    reach = np.array([r[np.abs(v[:, 2] - z) < 0.001].max(initial=0) for z in zs])
    knob_lo = float(zs[np.argmax((reach < 0.04) & (zs > 0.01))])
    disk = v[:, 2] <= knob_lo
    knob = v[:, 2] >= knob_lo - 0.001
    # Width across the knob (along y, where the fingers close) around its
    # middle, where the pads hold it; the widest ring is 1.5 mm wider.
    mid = (knob_lo + v[:, 2].max()) / 2
    band = knob & (np.abs(v[:, 2] - mid) < 0.006)
    top, base = pp.split_faces(
        mesh, mesh.vertices[mesh.faces][:, :, 2].mean(1) > knob_lo
    )
    return {
        "visual": [
            {"file": export(base, "lid_body"), "rgba": [0.80, 0.81, 0.83, 1]},
            {"file": export(top, "lid_knob"), "rgba": BLACK},
        ],
        "collision": [pp.hull(v[disk]), pp.hull(v[knob])],
        "dims": {
            "radius": float(r[disk].max()),
            "disk_top_z": float(v[disk, 2].max()),
            "knob_grip_width": float(2 * np.abs(v[band, 1]).max()),
            "knob_z": [knob_lo, float(v[:, 2].max())],
        },
    }


def make_tomato(export):
    """RoboSyn's tomato at 0.85 (70 mm tall at 1.0 would touch the pot lid),
    coloured from its texture: red body, green stem."""
    mesh = pp.load(SOURCES["tomato"], (0.85, 0.85, 0.85))
    mesh.visual = mesh.visual.to_color()
    mesh = pp.settle(pp.cluster(mesh, 0.0012))
    face = mesh.visual.vertex_colors[:, :3].astype(float)[mesh.faces].mean(1)
    green = face[:, 1] > face[:, 0]
    stem, body = pp.split_faces(mesh, green)
    mean = {
        name: (np.r_[face[mask].mean(0) / 255, 1]).round(3).tolist()
        for name, mask in (("body", ~green), ("stem", green))
    }
    # Its round bottom rolled it 4 cm off its slot; the collider gets a flat
    # base where the skin is within 6 mm of the table, as a real one sits.
    base = mesh.vertices.copy()
    base[base[:, 2] < 0.006, 2] = 0.0
    return {
        "visual": [
            {"file": export(body, "tomato_body"), "rgba": mean["body"]},
            {"file": export(stem, "tomato_stem"), "rgba": mean["stem"]},
        ],
        "collision": [pp.hull(base)],
        "dims": {"size": mesh.extents.tolist()},
    }


def make_apple(export):
    """RoboSyn's apple (raw in mm, stem along +y), stem up, 57-62 mm."""
    mesh = pp.load(SOURCES["apple"], (0.00075, 0.00075, 0.00075))
    mesh.apply_transform(pp.rot_x(np.pi / 2))
    mesh = pp.settle(mesh)
    centre = mesh.vertices[mesh.faces].mean(1)
    stem = (np.hypot(centre[:, 0], centre[:, 1]) < 0.006) & (
        centre[:, 2] > 0.8 * mesh.extents[2]
    )
    tip, body = pp.split_faces(mesh, stem)
    base = mesh.vertices.copy()
    base[base[:, 2] < 0.004, 2] = 0.0
    return {
        "visual": [
            {"file": export(body, "apple_body"), "rgba": [0.60, 0.77, 0.22, 1]},
            {"file": export(tip, "apple_stem"), "rgba": [0.36, 0.24, 0.12, 1]},
        ],
        "collision": [pp.hull(base)],
        "dims": {"size": mesh.extents.tolist()},
    }


def make_plate(export):
    """EmbodiChain's plate (raw in mm and upside down), right side up and 10%
    smaller: a flat centre and a sloped rim."""
    mesh = pp.load(SOURCES["plate"], (0.0009, 0.0009, 0.0008))
    mesh.apply_transform(pp.rot_x(np.pi))
    mesh = pp.settle(pp.cluster(mesh, 0.0015))
    v = mesh.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    flat = r < 0.05
    return {
        "visual": [{"file": export(mesh, "plate"), "rgba": [0.94, 0.94, 0.91, 1]}],
        "collision": [pp.hull(v[flat])] + pp.ring(v[r > 0.048], thicken=0.002),
        "dims": {
            "radius": float(r.max()),
            "floor_z": float(v[flat, 2].max()),
            "rim_z": float(v[:, 2].max()),
        },
    }


def make_button(export):
    """RoboSyn's push button: a fixed base and a cap that presses 5 mm."""
    base = pp.load(SOURCES["button_base"], (1, 1, 1))
    cap = pp.load(SOURCES["button_cap"], (1, 1, 1))
    return {
        "visual": [
            {"file": export(base, "button_base"), "rgba": [0.27, 0.61, 0.69, 1]},
            {"file": export(cap, "button_cap"), "rgba": [0.78, 0.65, 0.40, 1]},
        ],
        "collision": [pp.hull(base.vertices), pp.hull(cap.vertices)],
        "dims": {
            "base_top_z": float(base.bounds[1][2]),
            "base_radius": float(np.hypot(*base.vertices[:, :2].T).max()),
            "cap_z": [float(cap.bounds[0][2]), float(cap.bounds[1][2])],
            "cap_radius": float(np.hypot(*cap.vertices[:, :2].T).max()),
            "travel": 0.005,
        },
    }


def make_beaker(color, rgba):
    """RoboSyn's beaker, 60-64 mm across and 72 mm tall, tinted glass.
    Collided as its hull: nothing goes inside it."""

    def make(export):
        mesh = pp.settle(pp.load(SOURCES["beaker"], (1, 1, 1)))
        return {
            "visual": [{"file": export(mesh, f"beaker_{color}"), "rgba": rgba}],
            "collision": [pp.hull(mesh.vertices)],
            "dims": {"size": mesh.extents.tolist()},
        }

    return make


def make_cup(color, rgba):
    """A RoboSyn paper cup (75 mm across at the rim, 81-104 mm tall). Collided
    as its hull: the scenes only stand them up."""

    def make(export):
        mesh = pp.settle(pp.load(SOURCES[f"cup_{color}"], (1, 1, 1)))
        v = mesh.vertices
        r = np.hypot(v[:, 0], v[:, 1])
        return {
            "visual": [{"file": export(mesh, f"cup_{color}"), "rgba": rgba}],
            "collision": [pp.hull(v)],
            "dims": {
                "size": mesh.extents.tolist(),
                "base_radius": float(r[v[:, 2] < 0.004].max()),
                "rim_radius": float(r.max()),
            },
        }

    return make


def make_drawer(export):
    """EmbodiChain's SlidingBoxDrawer (RoboSyn's drawer_open_place), 30%
    wider, so the gripper housing fits inside the drawer: a cabinet open
    toward +x, 90 mm tall (0.45 of the raw height; RoboSyn uses 0.35), and a
    drawer sliding out along +x whose walls stand 57 mm (0.30), so a hand
    holding a 72 mm beaker near its rim stays above them while the beaker
    still fits under the cabinet's top. Its 6 mm handle bar is too thin for
    these pads, so a 20 mm knob replaces it. Collided as boxes: the cabinet's
    five walls, the drawer's floor and four walls, the knob."""
    s_cab, s_box = np.array([1.0, 1.3, 0.45]), np.array([1.0, 1.3, 0.30])
    cabinet = pp.load(SOURCES["drawer_outer"], s_cab)
    drawer = pp.load(SOURCES["drawer_inner"], s_box)
    bar = drawer.vertices[:, 0] > 0.1
    keep = ~bar[drawer.faces].any(1)
    drawer = drawer.submesh([np.flatnonzero(keep)], append=True)

    # Raw dimensions (m): cabinet x -0.1..0.1, y +-0.1, z 0..0.2, 10 mm walls;
    # drawer x -0.09..0.09, y +-0.09, z 0.01..0.19, 10 mm walls and floor.
    def scaled(lo, hi, s):
        return pp.box(np.multiply(lo, s), np.multiply(hi, s))

    walls = [
        scaled([-0.1, -0.1, 0.0], [-0.09, 0.1, 0.2], s_cab),  # back
        scaled([-0.1, 0.09, 0.0], [0.1, 0.1, 0.2], s_cab),
        scaled([-0.1, -0.1, 0.0], [0.1, -0.09, 0.2], s_cab),
        scaled([-0.1, -0.1, 0.0], [0.1, 0.1, 0.01], s_cab),  # floor
        scaled([-0.1, -0.1, 0.19], [0.1, 0.1, 0.2], s_cab),  # top
    ]
    box = [
        scaled([-0.09, -0.09, 0.01], [0.09, 0.09, 0.02], s_box),  # floor
        scaled([-0.09, -0.09, 0.01], [-0.08, 0.09, 0.19], s_box),
        scaled([0.08, -0.09, 0.01], [0.09, 0.09, 0.19], s_box),  # front
        scaled([-0.09, 0.08, 0.01], [0.09, 0.09, 0.19], s_box),
        scaled([-0.09, -0.09, 0.01], [0.09, -0.08, 0.19], s_box),
    ]
    knob_z = 0.1 * s_box[2]
    angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle = np.c_[np.zeros(16), 0.010 * np.cos(angle), knob_z + 0.010 * np.sin(angle)]
    knob = np.r_[circle + [0.09, 0, 0], circle + [0.135, 0, 0]]
    knob_mesh = trimesh.convex.convex_hull(knob)
    return {
        "visual": [
            {"file": export(cabinet, "drawer_cabinet"), "rgba": [0.86, 0.83, 0.76, 1]},
            {"file": export(drawer, "drawer_box"), "rgba": [0.62, 0.45, 0.30, 1]},
            {"file": export(knob_mesh, "drawer_knob"), "rgba": BLACK},
        ],
        "collision": walls + box + [pp.hull(knob)],
        "dims": {
            "cabinet": (0.2 * s_cab).tolist(),
            "cabinet_inner_top_z": 0.19 * s_cab[2],
            "cavity_half_y": 0.08 * s_box[1],
            "floor_z": 0.02 * s_box[2],
            "wall_top_z": 0.19 * s_box[2],
            "drawer_x": [-0.09, 0.09],
            "knob_x": [0.09, 0.135],
            "knob_z": knob_z,
            "knob_width": 0.020,
            "pieces": {"cabinet": len(walls), "drawer": len(box), "knob": 1},
        },
    }


MAKERS = {
    "pot": make_pot,
    "lid": make_lid,
    "tomato": make_tomato,
    "apple": make_apple,
    "plate": make_plate,
    "button": make_button,
    "beaker_blue": make_beaker("blue", [0.30, 0.55, 0.95, 0.75]),
    "beaker_amber": make_beaker("amber", [0.95, 0.62, 0.18, 0.75]),
    "cup_red": make_cup("red", [0.85, 0.18, 0.16, 1]),
    "cup_green": make_cup("green", [0.22, 0.66, 0.32, 1]),
    "cup_blue": make_cup("blue", [0.16, 0.42, 0.85, 1]),
    "drawer": make_drawer,
}
RAW_FILES = {
    "beaker_blue": ["beaker"],
    "beaker_amber": ["beaker"],
    "button": ["button_base", "button_cap"],
    "drawer": ["drawer_outer", "drawer_inner"],
}


def prepare(out_dir, names):
    """The named props, made into out_dir (cached; see sim.props.prepare)."""
    sources = [SOURCES[f] for n in names for f in RAW_FILES.get(n, [n])]
    return pp.prepare(out_dir, VERSION, sources, {n: MAKERS[n] for n in names})


def add_button(root, name, prop, mesh_dir, xy, contact):
    """The push button at xy on the table: a fixed base body <name>_base_part
    and the cap <name>_part on a sprung 5 mm slide, <name>_part_joint. Both are
    *_part bodies, so they never count as robot. MuJoCo's parent filter skips a
    fixed parent (it is welded to the world), so an exclude lets the cap sink
    into the base: without it the cap stopped after 1.2 mm."""
    asset, world = root.find("asset"), root.find("worldbody")
    visual, collision = pp.mesh_assets(asset, name, prop, mesh_dir)
    exclude = root.find("contact")
    if exclude is None:
        exclude = ET.SubElement(root, "contact")
    ET.SubElement(exclude, "exclude", body1=f"{name}_base_part", body2=f"{name}_part")
    base = ET.SubElement(
        world, "body", name=f"{name}_base_part", pos=fmt([*xy, TABLE_Z])
    )
    pp.add_geoms(base, f"{name}_base", visual[:1], collision[:1], contact)
    cap = ET.SubElement(base, "body", name=f"{name}_part")
    dims = prop["dims"]
    ET.SubElement(
        cap,
        "joint",
        name=f"{name}_part_joint",
        type="slide",
        axis="0 0 1",
        range=fmt([-dims["travel"], 0]),
        limited="true",
        stiffness="300",
        springref="0",
        damping="3",
        armature="0.0001",
    )
    ET.SubElement(
        cap,
        "inertial",
        pos=fmt([0, 0, sum(dims["cap_z"]) / 2]),
        mass="0.01",
        diaginertia="1e-6 1e-6 1e-6",
    )
    pp.add_geoms(cap, f"{name}_cap", visual[1:], collision[1:], contact)


def add_drawer(root, name, prop, mesh_dir, xy, yaw, contact, travel=0.16):
    """The drawer cabinet at xy on the table, turned yaw (pi faces the robot):
    a fixed body <name>_cabinet_part and the drawer <name>_part on a slide
    <name>_part_joint (0 closed, out to travel), knob included. The two never
    collide (the slide holds the drawer; at zero clearance they rubbed)."""
    asset, world = root.find("asset"), root.find("worldbody")
    visual, collision = pp.mesh_assets(asset, name, prop, mesh_dir)
    pieces = prop["dims"]["pieces"]
    exclude = root.find("contact")
    if exclude is None:
        exclude = ET.SubElement(root, "contact")
    ET.SubElement(
        exclude, "exclude", body1=f"{name}_cabinet_part", body2=f"{name}_part"
    )
    cabinet = ET.SubElement(
        world,
        "body",
        name=f"{name}_cabinet_part",
        pos=fmt([*xy, TABLE_Z]),
        quat=fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
    )
    pp.add_geoms(
        cabinet, f"{name}_cabinet", visual[:1], collision[: pieces["cabinet"]], contact
    )
    drawer = ET.SubElement(cabinet, "body", name=f"{name}_part")
    ET.SubElement(
        drawer,
        "joint",
        name=f"{name}_part_joint",
        type="slide",
        axis="1 0 0",
        range=fmt([0, travel]),
        limited="true",
        damping="2",
        frictionloss="0.3",
        armature="0.05",
    )
    size = prop["dims"]["cabinet"]
    pp.inertial(drawer, 0.3, [0.18, size[1] * 0.9, size[2] * 0.9])
    pp.add_geoms(
        drawer, f"{name}_box", visual[1:], collision[pieces["cabinet"] :], contact
    )
