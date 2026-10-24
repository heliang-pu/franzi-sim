"""Visual-only fittings the CAD export lacks, placed flush on the source shells.

Covers the lower-back ribbed panel, upper-back controls, thigh back cover and base
light strips seen on the real robot. Nothing here has mass or contacts.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

PARAMS = {
    "version": 7,
    "raster": 0.001,
    "cell": 0.002,
    "panel_inset": 0.003,
    "panel_thickness": 0.002,
    "panel_margin": 0.003,
    "rim": [0.004, 0.008],
    "rib_y": [-0.07, -0.035, 0.0, 0.035, 0.07],
    "rib_width": 0.008,
    "rib_height": 0.0025,
    "rib_end_margin": 0.012,
    "corner_z": [0.141, 0.173],
    "corner_width": 0.006,
}
SOURCE_MESHES = ("torso_Link", "base_link")
VISUAL = {"contype": "0", "conaffinity": "0", "mass": "0", "group": "1"}
# Hole centres (y, z) in the torso frame with the shell samples used to fit the
# surface across each hole: a ring (inner, outer radius) or, for the thin slit,
# bands above and below it (half width, gap from centre, band height).
HOLES = {
    "estop": ((0.0355, 0.3465), {"ring": (0.020, 0.032)}),
    "ports": ((-0.0205, 0.346), {"ring": (0.031, 0.042)}),
    "led": ((-0.0855, 0.349), {"ring": (0.013, 0.025)}),
    "plug": ((0.0855, 0.349), {"ring": (0.010, 0.022)}),
    "slit": ((0.0, 0.3685), {"band": (0.024, 0.003, 0.005)}),
}
# Regions (centre y, z, half y, half z) without shell, excluded from plane fits.
HOLE_BOXES = [
    (0.0355, 0.3465, 0.0195, 0.0175),
    (-0.0205, 0.346, 0.0290, 0.0125),
    (-0.0855, 0.349, 0.0120, 0.0120),
    (0.0855, 0.349, 0.0095, 0.0095),
    (0.0, 0.3685, 0.0255, 0.0030),
]
# Upper-back controls: name and material without their fit_/fa_ prefixes, hole,
# geom type, size and centre offset along the outward normal.
CONTROLS = [
    ("estop_collar", "black", "estop", "cylinder", (0.0185, 0.002), -0.0012),
    ("estop_stem", "estop_red", "estop", "cylinder", (0.015, 0.0045), 0.005),
    ("estop_cap", "estop_red", "estop", "ellipsoid", (0.015, 0.015, 0.0028), 0.0095),
    ("ports_block", "black", "ports", "box", (0.0273, 0.0108, 0.004), -0.0055),
    ("port_neg_y", "graphite", "ports", "box", (0.0095, 0.0072, 0.0015), -0.0018),
    ("port_pos_y", "graphite", "ports", "box", (0.0095, 0.0072, 0.0015), -0.0018),
    ("port_neg_y_socket", "black", "ports", "box", (0.0062, 0.0042, 0.0004), -0.0001),
    ("port_pos_y_socket", "black", "ports", "box", (0.0062, 0.0042, 0.0004), -0.0001),
    ("port_neg_y_led", "led_green", "ports", "cylinder", (0.0012, 0.0004), 0.0001),
    ("led_ring", "led_blue", "led", "cylinder", (0.0098, 0.0015), -0.0012),
    ("led_center", "graphite", "led", "cylinder", (0.0062, 0.0015), -0.0009),
    ("plug", "black", "plug", "cylinder", (0.0072, 0.002), -0.0023),
    ("slit", "black", "slit", "box", (0.0237, 0.0013, 0.002), -0.0025),
]
# In-plane offsets from the hole centre; on the back shell frame x points to -y.
OFFSETS = {
    "port_neg_y": (0.013, 0),
    "port_pos_y": (-0.013, 0),
    "port_neg_y_socket": (0.013, 0),
    "port_pos_y_socket": (-0.013, 0),
    "port_neg_y_led": (0.0208, 0.0048),
}


def _parts(source, name):
    mesh = trimesh.load(source / "meshes" / f"{name}.STL", process=True, force="mesh")
    return sorted(
        mesh.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True
    )


def _terms(u, v, square_v):
    """Quadratic design matrix; the v*v term needs samples on both sides of v."""
    cols = [np.ones_like(u), u, v, u * u, u * v] + ([v * v] if square_v else [])
    return np.column_stack(cols)


class _Grid:
    """Raster over the (y, z) plane; row 0 is the lowest z."""

    def __init__(self, box, step):
        self.y0, self.z0, y1, z1 = box
        self.step = step
        self.shape = (round((z1 - self.z0) / step), round((y1 - self.y0) / step))
        self.y = self.y0 + (np.arange(self.shape[1]) + 0.5) * step
        self.z = self.z0 + (np.arange(self.shape[0]) + 0.5) * step

    def coverage(self, mesh):
        image = Image.new("1", (self.shape[1], self.shape[0]), 0)
        draw = ImageDraw.Draw(image)
        cols = (mesh.triangles[:, :, 1] - self.y0) / self.step
        rows = self.shape[0] - (mesh.triangles[:, :, 2] - self.z0) / self.step
        for c, r in zip(cols, rows):
            draw.polygon(list(zip(c, r)), fill=1)
        return np.array(image)[::-1]

    def outer_x(self, mesh, sign):
        """Outermost surface x per cell seen from -x (sign -1) or +x (sign 1)."""
        pts, _ = trimesh.sample.sample_surface(mesh, 1_500_000, seed=0)
        iy = ((pts[:, 1] - self.y0) / self.step).astype(int)
        iz = ((pts[:, 2] - self.z0) / self.step).astype(int)
        ok = (iy >= 0) & (iy < self.shape[1]) & (iz >= 0) & (iz < self.shape[0])
        grid = np.full(self.shape, np.inf)
        np.minimum.at(grid, (iz[ok], iy[ok]), -sign * pts[ok, 0])
        return -sign * grid

    def surface(self, outer, sign, center, region, exclude=()):
        """Robust quadratic fit of the shell across a hole; point, outward normal."""
        Y, Z = np.meshgrid(self.y, self.z)
        u, v = Y - center[0], Z - center[1]
        if "ring" in region:
            r0, r1 = region["ring"]
            sel = (np.hypot(u, v) > r0) & (np.hypot(u, v) < r1)
        else:
            hy, gap, height = region["band"]
            sel = (np.abs(u) < hy) & (np.abs(v) > gap) & (np.abs(v) < gap + height)
        sel &= np.isfinite(outer)
        for yc, zc, hy, hz in exclude:
            sel &= (np.abs(Y - yc) > hy + 0.002) | (np.abs(Z - zc) > hz + 0.002)
        A, x = _terms(u[sel], v[sel], "ring" in region), outer[sel]
        keep = np.ones(len(x), bool)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[keep], x[keep], rcond=None)
            # Element-wise products: Accelerate's matmul raises spurious FP warnings.
            keep = np.abs((A * coef).sum(1) - x) < 0.0015
        point = np.array([coef[0], *center])
        normal = sign * np.array([1.0, -coef[1], -coef[2]])
        return point, normal / np.linalg.norm(normal)


def _frame(normal):
    """Rotation whose z is the outward normal and whose y is closest to world up."""
    up = np.array([0.0, 0, 1])
    ey = up - up @ normal * normal
    ey /= np.linalg.norm(ey)
    return np.c_[np.cross(ey, normal), ey, normal]


def _slab(mask, y0, z0, cell, x_front, thickness):
    """Closed slab over a cell mask whose outer face follows x_front(y, z)."""
    jj, ii = np.nonzero(mask)
    used = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), bool)
    for dj in (0, 1):
        for di in (0, 1):
            used[jj + dj, ii + di] = True
    nj, ni = np.nonzero(used)
    index = np.full(used.shape, -1)
    index[nj, ni] = np.arange(len(nj))
    y, z = y0 + ni * cell, z0 + nj * cell
    xf = x_front(y, z)
    vertices = np.r_[np.c_[xf, y, z], np.c_[xf + thickness, y, z]]
    nv = len(nj)
    a, b = index[jj, ii], index[jj, ii + 1]
    c, d = index[jj + 1, ii + 1], index[jj + 1, ii]
    faces = [np.c_[a, c, b], np.c_[a, d, c], np.c_[a, b, c] + nv, np.c_[a, c, d] + nv]
    padded = np.pad(mask, 1)
    for dj, di, n1, n2 in [
        (0, -1, (0, 0), (1, 0)),
        (0, 1, (0, 1), (1, 1)),
        (-1, 0, (0, 0), (0, 1)),
        (1, 0, (1, 0), (1, 1)),
    ]:
        edge = ~padded[jj + 1 + dj, ii + 1 + di]
        p = index[jj[edge] + n1[0], ii[edge] + n1[1]]
        q = index[jj[edge] + n2[0], ii[edge] + n2[1]]
        faces += [np.c_[p, q, q + nv], np.c_[p, q + nv, p + nv]]
    mesh = trimesh.Trimesh(vertices, np.concatenate(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _harmonic(domain, known):
    """Laplace fill of `domain` cells from finite `known` values on its border."""
    idx = np.full(domain.shape, -1)
    idx[domain] = np.arange(domain.sum())
    rows, cols, vals = [], [], []
    rhs = np.zeros(domain.sum())
    for j, i in zip(*np.nonzero(domain)):
        k = idx[j, i]
        degree = 0
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jj, ii = j + dj, i + di
            if domain[jj, ii]:
                rows.append(k)
                cols.append(idx[jj, ii])
                vals.append(-1.0)
                degree += 1
            elif np.isfinite(known[jj, ii]):
                rhs[k] += known[jj, ii]
                degree += 1
        rows.append(k)
        cols.append(k)
        vals.append(float(degree))
    n = domain.sum()
    solution = spsolve(sparse.csr_matrix((vals, (rows, cols)), shape=(n, n)), rhs)
    out = np.where(np.isfinite(known), known, np.nan)
    out[domain] = solution
    return out


def _back_panel(torso, p):
    grid = _Grid((-0.2, 0.0, 0.2, 0.5), p["raster"])
    shells = trimesh.util.concatenate([torso[0], torso[3]])
    opening = grid.coverage(torso[2]) & ~grid.coverage(shells)
    opening &= (grid.z < 0.3)[:, None]
    labels, count = ndimage.label(opening)
    largest = np.argmax(ndimage.sum(opening, labels, range(1, count + 1))) + 1
    opening = ndimage.binary_fill_holes(labels == largest)
    outer = grid.outer_x(shells, -1)
    dist = ndimage.distance_transform_edt(~opening) * grid.step
    # The panel sits behind the shell lip, flush with the recessed rim around it:
    # a harmonic fill cannot bulge past the rim, so it never leaves the silhouette.
    rim = (dist > p["rim"][0]) & (dist <= p["rim"][1]) & np.isfinite(outer)
    domain = dist <= p["rim"][0]
    filled = _harmonic(domain, np.where(rim, outer, np.inf))
    near = ndimage.distance_transform_edt(
        np.isnan(filled), return_distances=False, return_indices=True
    )
    filled = filled[tuple(near)]
    interp = RegularGridInterpolator(
        (grid.z, grid.y), filled, bounds_error=False, fill_value=None
    )

    def surface(y, z):
        return interp(np.c_[z, y])

    cell = p["cell"]
    k = round(cell / grid.step)
    grown = dist <= p["panel_margin"]
    h, w = grid.shape[0] // k, grid.shape[1] // k
    coarse = grown[: h * k, : w * k].reshape(h, k, w, k).mean((1, 3)) > 0.5
    coarse = ndimage.binary_opening(coarse)
    inset = p["panel_inset"]
    panel = _slab(
        coarse,
        grid.y0,
        grid.z0,
        cell,
        lambda y, z: surface(y, z) + inset,
        p["panel_thickness"],
    )
    core = ndimage.binary_erosion(coarse, iterations=round(p["rib_end_margin"] / cell))
    yc = grid.y0 + (np.arange(w) + 0.5) * cell
    ribs = []
    for y_rib in p["rib_y"]:
        mask = core & (np.abs(yc - y_rib) <= p["rib_width"] / 2)[None, :]
        ribs.append(
            _slab(
                mask,
                grid.y0,
                grid.z0,
                cell,
                lambda y, z: surface(y, z) + inset - p["rib_height"],
                p["rib_height"] + 0.0005,
            )
        )
    oz, oy = np.nonzero(opening)
    report = {
        "opening_y": [
            round(float(grid.y[oy.min()]), 4),
            round(float(grid.y[oy.max()]), 4),
        ],
        "opening_z": [
            round(float(grid.z[oz.min()]), 4),
            round(float(grid.z[oz.max()]), 4),
        ],
        "panel_x_range": [
            round(float(filled[opening].min() + inset), 4),
            round(float(filled[opening].max() + inset), 4),
        ],
        "shell_min_x": round(float(shells.vertices[:, 0].min()), 4),
    }
    return panel, ribs, report


def _corner_strip(base, side, p):
    """Thin ribbon down the rounded front corner (outermost along the diagonal)."""
    z0, z1 = p["corner_z"]
    pts, face = trimesh.sample.sample_surface(base, 600_000, seed=4)
    keep = (pts[:, 0] > 0.12) & (side * pts[:, 1] > 0.15)
    keep &= (pts[:, 2] > z0 - 0.003) & (pts[:, 2] < z1 + 0.003)
    pts, normal = pts[keep], base.face_normals[face[keep]]
    diagonal = np.array([1.0, side, 0.0]) / np.sqrt(2)
    zs = np.linspace(z0, z1, 24)
    support, normals = [], []
    for z in zs:
        band = np.nonzero(np.abs(pts[:, 2] - z) < 0.001)[0]
        best = pts[band[np.argmax((pts[band] * diagonal).sum(1))]]
        near = np.linalg.norm(pts - best, axis=1) < 0.004
        support.append(best)
        normals.append(normal[near].mean(0))
    # Quadratic in z smooths the jitter of discrete support points.
    support, normals = np.array(support), np.array(normals)
    centers = np.c_[
        [np.polyval(np.polyfit(zs, support[:, i], 2), zs) for i in (0, 1)]
    ].T
    centers = np.c_[centers, zs]
    normals = np.c_[
        [np.polyval(np.polyfit(zs, normals[:, i], 2), zs) for i in range(3)]
    ].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangent = np.gradient(centers, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    width = np.cross(normals, tangent)
    width /= np.linalg.norm(width, axis=1, keepdims=True)
    hw = p["corner_width"] / 2
    edges = []
    for lift in (0.0009, -0.0006):
        line = centers + normals * lift
        edges += [line - width * hw, line + width * hw]
    m = len(centers)
    faces = []
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        for i in range(m - 1):
            p0, p1 = a * m + i, b * m + i
            faces += [[p0, p1, p1 + 1], [p0, p1 + 1, p0 + 1]]
    e = m - 1
    faces += [[0, m, 3 * m], [0, 3 * m, 2 * m]]
    faces += [[e, 2 * m + e, 3 * m + e], [e, 3 * m + e, m + e]]
    mesh = trimesh.Trimesh(np.concatenate(edges), np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _slot_frame(shell):
    """Opening plane of the front light slot, spanning its lower and upper lips."""
    grid = _Grid((-0.3, 0.1, 0.3, 0.26), 0.0005)
    front = grid.outer_x(shell, 1)
    cols = np.abs(grid.y) < 0.08
    face = front[:, cols] > 0.15
    below = [j for j in range(grid.shape[0]) if grid.z[j] < 0.1705]
    above = [j for j in range(grid.shape[0]) if grid.z[j] > 0.1705]
    j0 = max(j for j in below if face[j].mean() > 0.9)
    j1 = min(j for j in above if face[j].mean() > 0.9)
    lips = []
    for j in (j0, j1):
        lips.append([np.median(front[j, cols][face[j]]), grid.z[j]])
    (x0, z0), (x1, z1) = lips
    rows = (grid.z > z0) & (grid.z < z1)
    gap = (front[rows] < 0.15).mean(0) > 0.5
    labels, _ = ndimage.label(gap)
    ys = grid.y[labels == labels[np.argmin(np.abs(grid.y))]]
    span = np.hypot(x1 - x0, z1 - z0)
    normal = np.array([z1 - z0, 0.0, -(x1 - x0)]) / span
    frame = _as_frame(
        np.array([(x0 + x1) / 2, (ys.min() + ys.max()) / 2, (z0 + z1) / 2]), normal
    )
    frame["size"] = [
        round(float(ys.max() - ys.min()) / 2, 5),
        round(float(span) / 2, 5),
    ]
    return frame


def _digest(source, params):
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
    h.update(json.dumps([HOLES, HOLE_BOXES]).encode())
    for name in SOURCE_MESHES:
        h.update((source / "meshes" / f"{name}.STL").read_bytes())
    return h.hexdigest()


def _as_frame(point, normal):
    return {"point": point.round(5).tolist(), "normal": normal.round(5).tolist()}


def build_fittings(source, out_dir):
    """Generate curved fitting meshes and flush placements, cached by source hash."""
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = _digest(source, PARAMS)
    manifest = out_dir / "fittings.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info.get("digest") == digest and all(
            Path(f).exists() for f in info["meshes"].values()
        ):
            return info
    torso, base = _parts(source, "torso_Link"), _parts(source, "base_link")
    panel, ribs, panel_report = _back_panel(torso, PARAMS)
    meshes = {"fit_back_panel": panel}
    meshes.update({f"fit_back_rib_{i}": rib for i, rib in enumerate(ribs)})
    for side, label in ((1, "left"), (-1, "right")):
        meshes[f"fit_base_corner_led_{label}"] = _corner_strip(base[1], side, PARAMS)
    files = {}
    for name, mesh in meshes.items():
        files[name] = str(out_dir / f"{name}.stl")
        mesh.export(files[name])
    grid = _Grid((-0.2, 0.25, 0.2, 0.45), 0.001)
    outer = grid.outer_x(torso[3], -1)
    frames = {
        name: _as_frame(*grid.surface(outer, -1, center, region, HOLE_BOXES))
        for name, (center, region) in HOLES.items()
    }
    frames["base_slot"] = _slot_frame(base[1])
    info = {
        "digest": digest,
        "meshes": files,
        "frames": frames,
        "back_panel": panel_report,
    }
    manifest.write_text(json.dumps(info, indent=2))
    return info


def _quat(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return f"{w:.7g} {x:.7g} {y:.7g} {z:.7g}"


def _vec(v):
    return " ".join(f"{a:.7g}" for a in v)


def attach_fittings(root, info):
    """Add the fitting geoms to their bodies; returns what was attached."""
    asset = root.find("asset")
    for name, path in info["meshes"].items():
        ET.SubElement(asset, "mesh", name=name, file=path, inertia="shell")
    bodies = {b.get("name"): b for b in root.iter("body")}
    torso, thigh, base = bodies["torso_Link"], bodies["thigh_Link"], bodies["base_link"]
    added = []

    def geom(body, name, **attrs):
        ET.SubElement(body, "geom", name=name, **attrs, **VISUAL)
        added.append(name)

    geom(
        torso, "fit_back_panel", type="mesh", mesh="fit_back_panel", material="fa_navy"
    )
    for name in info["meshes"]:
        if name.startswith("fit_back_rib_"):
            geom(torso, name, type="mesh", mesh=name, material="fa_graphite")
    for name, material, hole, kind, size, lift in CONTROLS:
        frame = info["frames"][hole]
        point, normal = np.array(frame["point"]), np.array(frame["normal"])
        rot = _frame(normal)
        dx, dy = OFFSETS.get(name, (0, 0))
        pos = point + rot[:, 0] * dx + rot[:, 1] * dy + normal * lift
        geom(
            torso,
            f"fit_{name}",
            type=kind,
            size=_vec(size),
            pos=_vec(pos),
            quat=_quat(rot),
            material=f"fa_{material}",
        )
    # Between the rails, behind both windows; it ends inside the crossbar below and
    # behind the waist actuator above, clear of the calf over the full knee range.
    geom(
        thigh,
        "fit_thigh_back_cover",
        type="box",
        size="0.0085 0.0758 0.1053",
        pos="-0.073 0 0.1667",
        material="fa_navy",
    )
    # Light strip filling the front slot, 1 mm behind the plane of its two lips.
    frame = info["frames"]["base_slot"]
    point, normal = np.array(frame["point"]), np.array(frame["normal"])
    half_y, half_span = frame["size"]
    geom(
        base,
        "fit_base_front_led",
        type="box",
        size=_vec((half_y + 0.0005, half_span + 0.0005, 0.001)),
        pos=_vec(point - normal * 0.002),
        quat=_quat(_frame(normal)),
        material="fa_led_green",
    )
    for label in ("left", "right"):
        name = f"fit_base_corner_led_{label}"
        geom(base, name, type="mesh", mesh=name, material="fa_led_green")
    return {"geoms": added, "back_panel": info["back_panel"]}
