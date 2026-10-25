"""Scene props from third-party meshes: scaled visual meshes, convex colliders
and their MJCF.

A scene's props module (scenes/<name>/props.py) lists its raw files and how to
turn each into a prop, and prepare() caches the result in the scene's
assets/derived/, keyed by the raw bytes and the scene's version. A prop is
{"visual": [{"file", "rgba"} or {"file", "texture"}], "collision": [vertex
lists], "dims": {...}}: visual files are STL (flat colour) or OBJ with a PNG
(textured), and each collision piece is a vertex list whose convex hull MuJoCo
builds, since MuJoCo collides any mesh as its convex hull.

Raw RoboSynChallenge and EmbodiChain props live in assets/robosyn/ (RAW), which
git ignores like sdk_review/: copied unmodified from their releases, they are
not this project's to publish.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from sim.build_task import ROOT, fmt

RAW = ROOT / "assets/robosyn"
SECTORS = 24


def load(rel, scale):
    """A raw mesh (path relative to RAW), scaled per axis."""
    mesh = trimesh.load(RAW / rel, force="mesh", process=False)
    mesh.apply_scale(scale)
    return mesh


def rot_x(a):
    return trimesh.transformations.rotation_matrix(a, [1, 0, 0])


def rot_z(a):
    return trimesh.transformations.rotation_matrix(a, [0, 0, 1])


def settle(mesh, centre=None):
    """Translate mesh so its base is at z = 0 and centre (default the bounding
    box centre) at x = y = 0."""
    c = mesh.bounds.mean(0) if centre is None else np.asarray(centre, float)
    mesh.apply_translation([-c[0], -c[1], -mesh.bounds[0][2]])
    return mesh


def cluster(mesh, cell):
    """Vertex-clustering decimation: vertices in one cubic cell (m) merge into
    their mean, vertex colours averaged; degenerate faces drop."""
    keys = np.floor(mesh.vertices / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    count = np.bincount(inverse)
    vertices = np.zeros((len(count), 3))
    np.add.at(vertices, inverse, mesh.vertices)
    vertices /= count[:, None]
    faces = inverse[mesh.faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    out = trimesh.Trimesh(vertices, faces[keep], process=False)
    if mesh.visual.kind == "vertex":
        colors = np.zeros((len(count), 4))
        np.add.at(colors, inverse, mesh.visual.vertex_colors.astype(float))
        out.visual.vertex_colors = (colors / count[:, None]).astype(np.uint8)
    return out


def hull(points):
    """Vertices of the convex hull of points (m), rounded to 0.1 mm."""
    return np.round(trimesh.convex.convex_hull(points).vertices, 4).tolist()


def ring(points, count=SECTORS, thicken=0.003):
    """Convex pieces of a round wall: its points cut into count angular
    sectors (overlapping by a degree each side), each pushed outward by thicken
    as well, so even a single-sided shell gives solid pieces."""
    angle = np.arctan2(points[:, 1], points[:, 0])
    r = np.hypot(points[:, 0], points[:, 1])[:, None]
    outward = np.c_[points[:, :2] / r * thicken, np.zeros(len(points))]
    width, pad = 2 * np.pi / count, np.radians(1)
    pieces = []
    for k in range(count):
        mid = -np.pi + (k + 0.5) * width
        gap = (angle - mid + np.pi) % (2 * np.pi) - np.pi
        mask = np.abs(gap) <= width / 2 + pad
        if mask.sum() >= 4:
            pieces.append(hull(np.r_[points[mask], points[mask] + outward[mask]]))
    return pieces


def box(lo, hi):
    """The 8 corners of an axis-aligned box, a collision piece."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return [
        [x, y, z]
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]


def split_faces(mesh, mask):
    """The submeshes of mesh's faces where mask is true and false."""
    return (
        mesh.submesh([np.flatnonzero(mask)], append=True),
        mesh.submesh([np.flatnonzero(~mask)], append=True),
    )


def prepare(out_dir, version, sources, makers):
    """Derived props, cached in out_dir/props.json: makers maps each prop name
    to a function of export (export(mesh, name) writes out_dir/<name>.stl and
    returns its file name) that returns the prop. The cache key covers the raw
    files in sources (paths relative to RAW) and version."""
    out_dir = Path(out_dir)
    h = hashlib.sha256(str(version).encode())
    for rel in sorted(sources):
        h.update((RAW / rel).read_bytes())
    key = h.hexdigest()[:16]
    path = out_dir / "props.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return cached["props"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def export(mesh, name, texture=None):
        if texture is None:
            target = out_dir / f"{name}.stl"
            mesh.export(target)
        else:
            target = out_dir / f"{name}.obj"
            mesh.export(target, include_texture=False)
            texture.save(out_dir / f"{name}.png")
        return target.name

    props = {name: make(export) for name, make in makers.items()}
    path.write_text(json.dumps({"key": key, "props": props}))
    return props


def mesh_assets(asset, name, prop, mesh_dir):
    """MJCF assets for a prop: its visual meshes (shell inertia, so open
    surfaces compile; textured ones with their texture and material) and its
    convex collision pieces (vertex lists). Returns ([(mesh, rgba or material)],
    [collision mesh])."""
    visual, collision = [], []
    for k, item in enumerate(prop["visual"]):
        mesh = f"{name}_vis{k}"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh,
            file=str(Path(mesh_dir) / item["file"]),
            inertia="shell",
        )
        if "texture" in item:
            ET.SubElement(
                asset,
                "texture",
                name=f"{mesh}_tex",
                type="2d",
                file=str(Path(mesh_dir) / item["texture"]),
            )
            ET.SubElement(asset, "material", name=f"{mesh}_mat", texture=f"{mesh}_tex")
            visual.append((mesh, {"material": f"{mesh}_mat"}))
        else:
            visual.append((mesh, {"rgba": fmt(item["rgba"])}))
    for k, vertices in enumerate(prop["collision"]):
        mesh = f"{name}_col{k:02d}"
        ET.SubElement(asset, "mesh", name=mesh, vertex=fmt(np.ravel(vertices)))
        collision.append(mesh)
    return visual, collision


def add_geoms(body, name, visual, collision, contact):
    """The prop's visual geoms (group 1, no contact) and collision geoms
    (group 3, invisible) on body."""
    for k, (mesh, look) in enumerate(visual):
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual{k}",
            type="mesh",
            mesh=mesh,
            contype="0",
            conaffinity="0",
            group="1",
            **look,
        )
    for k, mesh in enumerate(collision):
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_collision{k:02d}",
            type="mesh",
            mesh=mesh,
            group="3",
            rgba="0.5 0.5 0.5 0",
            **contact,
        )


def inertial(body, mass, size, centre=None):
    """An explicit inertial: a solid box of size (m) around centre (default
    half its height above the body origin), so neither open visual shells nor
    overlapping colliders set the mass."""
    sx, sy, sz = size
    ET.SubElement(
        body,
        "inertial",
        pos=fmt([0, 0, sz / 2] if centre is None else centre),
        mass=fmt(mass),
        diaginertia=fmt(
            [
                mass * (sy**2 + sz**2) / 12,
                mass * (sx**2 + sz**2) / 12,
                mass * (sx**2 + sy**2) / 12,
            ]
        ),
    )


def free_body(world, name, pos, yaw, mass, size):
    """A free object body named <name>_part (Scene.robot_bodies excludes that
    suffix), its freejoint <name>_part_joint, with an explicit inertial."""
    body = ET.SubElement(
        world,
        "body",
        name=f"{name}_part",
        pos=fmt(pos),
        quat=fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
    )
    ET.SubElement(body, "freejoint", name=f"{name}_part_joint")
    inertial(body, mass, size)
    return body
