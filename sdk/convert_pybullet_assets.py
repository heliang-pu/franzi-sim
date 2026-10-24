"""Derive a URDF scene from the compiled MuJoCo scene, with MuJoCo's colliders.

    /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -m sdk.convert_pybullet_assets

The Isaac Sim draft (simbridge/backends/isaacsim_backend.py) loads this scene.
It was first made for the PyBullet backend, which was retired on 2026-10-24
(user decision: simbridge runs on MuJoCo only), hence the file and folder names.

The input is run_expert.Scene, i.e. build_task.build() plus the cameras and the
actuator overrides that every MuJoCo session uses, together with the source URDF
(never edited). Everything is written to assets/derived/pybullet/:

- ``franzi_pybullet.urdf``: the source URDF's links, joints, limits and
  inertials. Visuals are the source STLs by absolute path, coloured with the
  source MuJoCo materials. Every collision element is replaced by the MuJoCo
  collider of the same body, in the body (= link) frame: boxes and cylinders as
  primitives, every mesh collider (49 wrist hulls per side, one hull per finger)
  as its own convex-hull OBJ in ``collision/``, vertices in link coordinates.
  The fixed sensor links keep their joints and get no collision element, as in
  MuJoCo.
- ``scene.json``: base pose, world geoms, free parts, the MuJoCo contact-pair
  matrix and its realisation as group/mask classes, friction values, actuator
  limits, cameras and render settings.
- ``conversion_report.json``: input/output hashes, collider counts, URDF-vs-MuJoCo
  frame/limit/mass checks and the contact-pair matrix.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from sim.build_task import ROOT, SOURCE
from sim.run_expert import Scene

OUT = ROOT / "assets/derived/pybullet"
URDF_SOURCE = SOURCE / "wheel_robot_4.0.urdf"
URDF_OUT = OUT / "franzi_pybullet.urdf"
SCENE_OUT = OUT / "scene.json"
REPORT_OUT = OUT / "conversion_report.json"
COLLISION_DIR = OUT / "collision"
OBJECTS_DIR = OUT / "objects"
# A conversion is stale when any of these changed after it.
STRICT_INPUTS = [
    "assets/franzi_source/wheel_robot_4.0.urdf",
    "assets/franzi_source/franzi.xml",
    "sim/build_task.py",
    "sim/camera_rig.py",
]
# Recorded for provenance only.
TRACKED_INPUTS = [
    "sim/run_expert.py",
    "sim/camera_rig.json",
    "sdk/convert_pybullet_assets.py",
]
MJ_GEOM = {
    0: "plane",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def g17(values):
    return " ".join(f"{float(v):.17g}" for v in np.asarray(values, float).reshape(-1))


def quat_matrix(q):
    """MuJoCo (w, x, y, z) quaternion -> rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rpy(R):
    """URDF roll-pitch-yaw (R = Rz Ry Rx) of a rotation matrix."""
    return Rotation.from_matrix(R).as_euler("xyz")


def rpy_matrix(values):
    return Rotation.from_euler("xyz", values).as_matrix()


def axis_matrix(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) * angle).as_matrix()


def xyzw(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    return [x, y, z, w]


def rot_angle(R1, R2):
    """Angle of R1^T R2, accurate near zero (atan2 instead of arccos)."""
    dR = R1.T @ R2
    w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return float(np.arctan2(np.linalg.norm(w) / 2, (np.trace(dR) - 1) / 2))


# ----- source URDF ------------------------------------------------------------
def read_urdf():
    root = ET.parse(URDF_SOURCE).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        limit = j.find("limit")
        joints[j.get("name")] = {
            "element": j,
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
            "rpy": np.fromstring(origin.get("rpy", "0 0 0"), sep=" "),
            "axis": np.fromstring(j.find("axis").get("xyz"), sep=" ")
            if j.find("axis") is not None
            else np.zeros(3),
            "limit": None
            if limit is None or j.get("type") == "continuous"
            else (float(limit.get("lower")), float(limit.get("upper"))),
        }
    children = {j["child"] for j in joints.values()}
    roots = [name for name in links if name not in children]
    assert len(roots) == 1, roots
    return root, links, joints, roots[0]


def urdf_fk(joints, root_link, base_R, base_p, q):
    """World frames {link: (R, p)} of every URDF link for joint values q (dict)."""
    frames = {root_link: (base_R, base_p)}
    pending = list(joints.values())
    while pending:
        rest = []
        for j in pending:
            if j["parent"] not in frames:
                rest.append(j)
                continue
            R, p = frames[j["parent"]]
            R0 = rpy_matrix(j["rpy"])
            p_child = p + R @ j["xyz"]
            R_child = R @ R0
            value = q.get(j["element"].get("name"), 0.0)
            if j["type"] in ("revolute", "continuous"):
                R_child = R_child @ axis_matrix(
                    j["axis"] / np.linalg.norm(j["axis"]), value
                )
            elif j["type"] == "prismatic":
                p_child = p_child + R_child @ (
                    j["axis"] / np.linalg.norm(j["axis"]) * value
                )
            frames[j["child"]] = (R_child, p_child)
        assert len(rest) < len(pending), "URDF tree is not connected"
        pending = rest
    return frames


def source_materials():
    """Mesh name -> (material name, rgba) from the source MuJoCo model."""
    root = ET.parse(SOURCE / "franzi.xml").getroot()
    materials = {
        mat.get("name"): [float(v) for v in mat.get("rgba", "1 1 1 1").split()]
        for mat in root.find("asset").findall("material")
    }
    meshes = {}
    for geom in root.iter("geom"):
        if geom.get("mesh") and geom.get("material") in materials:
            meshes[geom.get("mesh")] = (
                geom.get("material"),
                materials[geom.get("material")],
            )
    return meshes


# ----- compiled MuJoCo model --------------------------------------------------
def hull(m, geom):
    """Convex hull of a mesh geom in its body frame: MuJoCo's own hull graph."""
    mesh = int(m.geom_dataid[geom])
    adr, num = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
    verts = m.mesh_vert[adr : adr + num].astype(float)
    gadr = int(m.mesh_graphadr[mesh])
    reference = ConvexHull(verts)
    if gadr >= 0:
        g = m.mesh_graph
        nv, nf = int(g[gadr]), int(g[gadr + 1])
        ids = np.asarray(g[gadr + 2 + nv : gadr + 2 + 2 * nv], int)
        faces = np.asarray(
            g[gadr + 2 + 3 * nv + 3 * nf : gadr + 2 + 3 * nv + 6 * nf], int
        )
        faces = faces.reshape(nf, 3)
        local = {int(gid): i for i, gid in enumerate(ids)}
        faces = np.vectorize(local.__getitem__)(faces)
        points = verts[ids]
        source = "mujoco mesh_graph"
    else:
        points = verts[reference.vertices]
        local = {int(gid): i for i, gid in enumerate(reference.vertices)}
        faces = np.vectorize(local.__getitem__)(reference.simplices)
        source = "scipy qhull"
    # Body frame, outward winding.
    R = quat_matrix(m.geom_quat[geom])
    points = m.geom_pos[geom] + np.einsum("ij,kj->ki", R, points)
    if not np.isfinite(points).all():
        raise ValueError(f"{m.geom(geom).name}: non-finite hull vertices")
    centre = points.mean(0)
    for f in faces:
        a, b, c = points[f]
        if np.dot(np.cross(b - a, c - a), a - centre) < 0:
            f[[1, 2]] = f[[2, 1]]
    volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                points[faces[:, 0]],
                np.cross(points[faces[:, 1]], points[faces[:, 2]]),
            )
        )
        / 6
    )
    return (
        points,
        faces,
        {
            "hull_source": source,
            "vertices": len(points),
            "faces": len(faces),
            "volume_m3": volume,
            "qhull_volume_m3": float(reference.volume),
            "mesh_vertices": num,
        },
    )


def write_obj(path, points, faces, name):
    lines = [
        f"# convex hull of MuJoCo geom {name}, link-frame coordinates (m)",
        f"o {name}",
    ]
    lines += [f"v {p[0]:.10g} {p[1]:.10g} {p[2]:.10g}" for p in points]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def collision_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]


def mujoco_pair_enabled(m, g1, g2):
    """MuJoCo's own filters: same body/weld, parent filter, contype/conaffinity."""
    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
    if w1 == w2:
        return False
    p1 = int(m.body_weldid[m.body_parentid[w1]])
    p2 = int(m.body_weldid[m.body_parentid[w2]])
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    if filterparent and w1 != 0 and w2 != 0 and (w1 == p2 or w2 == p1):
        return False
    t1, a1 = int(m.geom_contype[g1]), int(m.geom_conaffinity[g1])
    t2, a2 = int(m.geom_contype[g2]), int(m.geom_conaffinity[g2])
    return bool((t1 & a2) or (t2 & a1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cases", type=int, default=64, help="random joint vectors for frame checks"
    )
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        scene = Scene(Path(tmp), record=False)
        scene_xml_sha = sha256(Path(tmp) / "scene.xml")
    m, d = scene.m, scene.d
    assert m.nexclude == 0 and m.npair == 0, (
        "explicit <exclude>/<pair> are not converted"
    )
    urdf_root, links, joints, root_link = read_urdf()
    body_names = [m.body(i).name for i in range(m.nbody)]
    site_names = [m.site(i).name for i in range(m.nsite)]
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mujoco_version": mujoco.__version__,
        "inputs": {p: sha256(ROOT / p) for p in STRICT_INPUTS + TRACKED_INPUTS},
        "strict_inputs": STRICT_INPUTS,
        "mujoco_scene_xml_sha256": scene_xml_sha,
    }

    # ----- links <-> MuJoCo bodies/sites -----------------------------------------
    link_map = {}
    for name in links:
        if name in body_names:
            link_map[name] = {"mujoco": "body", "name": name}
            continue
        site = name.removesuffix("_Link")
        if site not in site_names:
            raise ValueError(
                f"URDF link {name} is neither a MuJoCo body nor a sensor site"
            )
        joint = next(j for j in joints.values() if j["child"] == name)
        if joint["type"] != "fixed":
            raise ValueError(f"{name}: only fixed sensor links may be merged by MuJoCo")
        sid = site_names.index(site)
        parent = body_names[m.site_bodyid[sid]]
        if parent != joint["parent"]:
            raise ValueError(
                f"{name}: MuJoCo site {site} sits on {parent}, URDF parent is {joint['parent']}"
            )
        link_map[name] = {"mujoco": "site", "name": site, "parent": parent}
    robot_bodies = {v["name"] for v in link_map.values() if v["mujoco"] == "body"}
    free_bodies = [
        body_names[m.jnt_bodyid[j]]
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    others = set(body_names) - robot_bodies - set(free_bodies) - {"world"}
    if others:
        raise ValueError(
            f"MuJoCo bodies with no URDF link and no free joint: {sorted(others)}"
        )
    report["link_map"] = link_map

    # ----- joint axes and limits ------------------------------------------------
    limits = {}
    for name, j in joints.items():
        if j["type"] == "fixed":
            continue
        jid = m.joint(name).id
        mj_axis = m.jnt_axis[jid]
        axis = j["axis"] / np.linalg.norm(j["axis"])
        limited = bool(m.jnt_limited[jid])
        entry = {
            "type": j["type"],
            "urdf": j["limit"],
            "mujoco": m.jnt_range[jid].tolist() if limited else None,
            "axis_error": float(np.linalg.norm(mj_axis - axis)),
        }
        # build_task.align_joint_limits may clip a MuJoCo range to the controller
        # limits; it must never widen one. The derived URDF takes MuJoCo's range.
        if (j["limit"] is None) == limited or (
            limited
            and (
                m.jnt_range[jid][0] < j["limit"][0] - 1e-12
                or m.jnt_range[jid][1] > j["limit"][1] + 1e-12
            )
        ):
            raise ValueError(
                f"{name}: URDF limit {j['limit']} vs MuJoCo {entry['mujoco']}"
            )
        entry["aligned"] = bool(
            limited and not np.allclose(m.jnt_range[jid], j["limit"], atol=1e-12)
        )
        if entry["axis_error"] > 1e-12 or np.linalg.norm(m.jnt_pos[jid]) > 1e-12:
            raise ValueError(
                f"{name}: joint axis/position differs between URDF and MuJoCo"
            )
        limits[name] = entry
    report["joint_check"] = {
        "joints": len(limits),
        "clipped_to_controller_limits": [n for n, e in limits.items() if e["aligned"]],
        "detail": limits,
    }

    # ----- masses (fixed sensor links are merged without mass in MuJoCo) --------
    mass = {}
    for name, link in links.items():
        inertial = link.find("inertial")
        urdf_mass = (
            float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        )
        mj_mass = float(m.body_mass[m.body(name).id]) if name in robot_bodies else None
        mass[name] = {"urdf_kg": urdf_mass, "mujoco_body_kg": mj_mass}
    missing = {k: v["urdf_kg"] for k, v in mass.items() if v["mujoco_body_kg"] is None}
    report["mass_check"] = {
        "note": "the derived URDF keeps the source inertials; the source MuJoCo model "
        "merged the fixed sensor links into their parents without their mass",
        "links_without_mujoco_body_kg": missing,
        "bodies_with_different_mass": {
            k: v
            for k, v in mass.items()
            if v["mujoco_body_kg"] is not None
            and abs(v["urdf_kg"] - v["mujoco_body_kg"]) > 1e-9
        },
    }

    # ----- frames: MuJoCo body/site frames vs URDF FK for random joint vectors ---
    base_id = m.body(root_link).id
    assert m.body_parentid[base_id] == 0 and m.body_weldid[base_id] == 0, (
        "base must be fixed"
    )
    base_p = m.body_pos[base_id].copy()
    base_R = quat_matrix(m.body_quat[base_id])
    movable = [n for n, j in joints.items() if j["type"] != "fixed"]
    cases = []
    worst = {"position_m": 0.0, "rotation_rad": 0.0}
    worst_kind = {"body": [0.0, 0.0, None], "site": [0.0, 0.0, None]}
    for k in range(args.cases):
        q = {}
        for n in movable:
            lo, hi = joints[n]["limit"] or (-np.pi, np.pi)
            q[n] = 0.0 if k == 0 else float(rng.uniform(lo, hi))
        for n, value in q.items():
            d.qpos[m.joint(n).qposadr[0]] = value
        mujoco.mj_kinematics(m, d)
        expected = urdf_fk(joints, root_link, base_R, base_p, q)
        frames = {}
        for link, target in link_map.items():
            if target["mujoco"] == "body":
                bid = m.body(target["name"]).id
                p, R = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
            else:
                sid = m.site(target["name"]).id
                p, R = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
            Ru, pu = expected[link]
            dp, dr = float(np.linalg.norm(p - pu)), rot_angle(R, Ru)
            worst["position_m"] = max(worst["position_m"], dp)
            worst["rotation_rad"] = max(worst["rotation_rad"], dr)
            w = worst_kind[target["mujoco"]]
            w[0] = max(w[0], dp)
            if dr > w[1]:
                w[1], w[2] = dr, link
            frames[link] = [*p.tolist(), *R.reshape(-1).tolist()]
        sites = {}
        for side in ("left", "right"):
            sid = m.site(f"{side}_grasp").id
            sites[f"{side}_grasp"] = d.site_xpos[sid].tolist()
        cases.append({"joints": q, "links": frames, "sites": sites})
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    # Bodies must match exactly. franzi.xml prints the sensor-site quaternions
    # with 6 significant digits, so sensor frames may differ by ~1e-6 rad.
    if (
        worst["position_m"] > 1e-9
        or worst_kind["body"][1] > 1e-9
        or worst_kind["site"][1] > 2e-6
    ):
        raise ValueError(f"MuJoCo frames differ from URDF link frames: {worst_kind}")
    report["frame_check"] = {
        "cases": len(cases),
        "first_case": "all joints 0",
        "frames_per_case": len(link_map),
        "max_position_error_m": worst["position_m"],
        "max_rotation_error_rad": worst["rotation_rad"],
        "bodies": {
            "max_position_error_m": worst_kind["body"][0],
            "max_rotation_error_rad": worst_kind["body"][1],
        },
        "sensor_sites": {
            "max_position_error_m": worst_kind["site"][0],
            "max_rotation_error_rad": worst_kind["site"][1],
            "worst_link": worst_kind["site"][2],
            "cause": "franzi.xml stores the sensor-site quaternions with 6 significant "
            "digits; the URDF rpy is exact. The PyBullet cameras follow the URDF.",
        },
        "method": "MuJoCo mj_kinematics body xpos/xmat (sensor links: site frames) "
        "vs NumPy FK of the source URDF from the same base pose",
    }

    # ----- colliders ----------------------------------------------------------------
    if COLLISION_DIR.exists():
        shutil.rmtree(COLLISION_DIR)
    COLLISION_DIR.mkdir(parents=True)
    geoms = collision_geoms(m)
    per_link = {}
    counts = {}
    for g in geoms:
        body = body_names[m.geom_bodyid[g]]
        if body not in robot_bodies:
            continue
        name = m.geom(g).name
        kind = MJ_GEOM[int(m.geom_type[g])]
        entry = {"geom": name, "type": kind}
        R = quat_matrix(m.geom_quat[g])
        if kind == "box":
            entry.update(
                xyz=m.geom_pos[g].tolist(),
                rpy=rpy(R).tolist(),
                size=(2 * m.geom_size[g]).tolist(),
            )
        elif kind == "cylinder":
            entry.update(
                xyz=m.geom_pos[g].tolist(),
                rpy=rpy(R).tolist(),
                radius=float(m.geom_size[g][0]),
                length=float(2 * m.geom_size[g][1]),
            )
        elif kind == "mesh":
            points, faces, info = hull(m, g)
            if abs(info["volume_m3"] - info["qhull_volume_m3"]) > 1e-9 * max(
                1.0, info["qhull_volume_m3"]
            ):
                raise ValueError(f"{name}: MuJoCo hull and qhull disagree")
            path = COLLISION_DIR / f"{name}.obj"
            write_obj(path, points, faces, name)
            entry.update(
                file=str(path), xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0], **info
            )
        else:
            raise ValueError(f"{name}: geom type {kind} has no URDF form here")
        per_link.setdefault(body, []).append(entry)
        counts[kind] = counts.get(kind, 0) + 1

    # ----- derived URDF -------------------------------------------------------------
    colors = source_materials()
    out = ET.Element("robot", name=urdf_root.get("name") + "_pybullet")
    out.append(
        ET.Comment(
            " Generated by convert_pybullet_assets.py from the compiled MuJoCo scene; do not edit. "
            "Links, joints, limits and inertials: source URDF. Collisions: MuJoCo colliders. "
        )
    )
    used = {}
    for link in links.values():
        for visual in link.findall("visual"):
            stem = Path(visual.find("geometry/mesh").get("filename")).stem
            mat, rgba = colors.get(stem, ("urdf_white", [1.0, 1.0, 1.0, 1.0]))
            used[mat] = rgba
    for mat, rgba in sorted(used.items()):
        ET.SubElement(ET.SubElement(out, "material", name=mat), "color", rgba=g17(rgba))
    visuals = {}
    for name, link in links.items():
        new = ET.SubElement(out, "link", name=name)
        inertial = link.find("inertial")
        if inertial is not None:
            new.append(inertial)
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename")
            assert filename.startswith("package://franzi_description/meshes/"), filename
            path = SOURCE / "meshes" / Path(filename).name
            assert path.is_file(), path
            v = ET.SubElement(new, "visual")
            origin = visual.find("origin")
            ET.SubElement(
                v,
                "origin",
                xyz=origin.get("xyz", "0 0 0"),
                rpy=origin.get("rpy", "0 0 0"),
            )
            ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=str(path))
            mat = colors.get(path.stem, ("urdf_white", None))[0]
            ET.SubElement(v, "material", name=mat)
            visuals[name] = {"mesh": str(path), "material": mat}
        for entry in per_link.get(name, []):
            c = ET.SubElement(new, "collision", name=entry["geom"])
            ET.SubElement(c, "origin", xyz=g17(entry["xyz"]), rpy=g17(entry["rpy"]))
            geometry = ET.SubElement(c, "geometry")
            if entry["type"] == "box":
                ET.SubElement(geometry, "box", size=g17(entry["size"]))
            elif entry["type"] == "cylinder":
                ET.SubElement(
                    geometry,
                    "cylinder",
                    radius=f"{entry['radius']:.17g}",
                    length=f"{entry['length']:.17g}",
                )
            else:
                ET.SubElement(geometry, "mesh", filename=entry["file"])
    for j in urdf_root.findall("joint"):
        entry = limits.get(j.get("name"))
        if entry and entry["mujoco"] is not None:
            lo, hi = entry["mujoco"]
            j.find("limit").set("lower", f"{lo:.17g}")
            j.find("limit").set("upper", f"{hi:.17g}")
        out.append(j)
    ET.indent(out)
    URDF_OUT.write_text(
        '<?xml version="1.0"?>\n' + ET.tostring(out, encoding="unicode") + "\n"
    )

    # ----- collision objects and the contact-pair matrix -----------------------------
    # Bullet filters per collision object (one link, or one world geom/free body).
    objects = {}
    for g in geoms:
        body = body_names[m.geom_bodyid[g]]
        if body == "world":
            key, kind = m.geom(g).name, "world_static"
        elif body in free_bodies:
            key, kind = body, "free"
        else:
            key, kind = (
                body,
                "robot_fixed" if m.body_weldid[m.geom_bodyid[g]] == 0 else "robot",
            )
        objects.setdefault(key, {"kind": kind, "geoms": [], "mujoco_body": body})[
            "geoms"
        ].append(g)
    names = list(objects)
    for key, obj in objects.items():
        gs = obj["geoms"]
        props = {
            "contype": {int(m.geom_contype[g]) for g in gs},
            "conaffinity": {int(m.geom_conaffinity[g]) for g in gs},
            "friction": {tuple(np.round(m.geom_friction[g], 12)) for g in gs},
            "condim": {int(m.geom_condim[g]) for g in gs},
            "priority": {int(m.geom_priority[g]) for g in gs},
        }
        for prop, values in props.items():
            if len(values) != 1:
                raise ValueError(
                    f"{key}: geoms disagree on {prop} {values}; one Bullet object cannot express it"
                )
        obj.update({p: next(iter(v)) for p, v in props.items()})
        obj["friction"] = list(obj["friction"])
    if len({o["priority"] for o in objects.values()}) != 1:
        raise ValueError(
            "geom priorities differ; the max-friction rule below assumes equal priority"
        )
    enabled = set()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            flags = {
                mujoco_pair_enabled(m, ga, gb)
                for ga in objects[a]["geoms"]
                for gb in objects[b]["geoms"]
            }
            if len(flags) != 1:
                raise ValueError(
                    f"{a}/{b}: geom pairs disagree; not expressible per object"
                )
            if flags.pop():
                enabled.add((a, b))
    # Empirical MuJoCo check: contacts in random poses only between enabled objects.
    geom_object = {g: key for key, o in objects.items() for g in o["geoms"]}
    seen = set()
    for case in cases:
        for n, value in case["joints"].items():
            d.qpos[m.joint(n).qposadr[0]] = value
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            a, b = (geom_object[int(g)] for g in c.geom)
            seen.add(tuple(sorted((a, b), key=names.index)))
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    stray = sorted(seen - enabled)
    if stray:
        raise ValueError(
            f"MuJoCo produced contacts for pairs the matrix disables: {stray}"
        )
    # Bullet: group/mask with the default AND test. Objects with identical rows
    # share a class bit; a class's mask is the OR of the classes it collides with.
    rows = {
        k: frozenset(b for a, b in enabled if a == k)
        | frozenset(a for a, b in enabled if b == k)
        for k in names
    }
    classes = []
    for k in names:
        for c in classes:
            if rows[c["members"][0]] == rows[k]:
                c["members"].append(k)
                break
        else:
            classes.append({"members": [k]})
    if len(classes) > 31:
        raise ValueError("more than 31 collision classes")
    member_class = {}
    for i, c in enumerate(classes):
        c["name"] = f"class_{i}"
        c["group"] = 1 << i
        for k in c["members"]:
            member_class[k] = i
    for c in classes:
        partners = {member_class[p] for p in rows[c["members"][0]]}
        c["mask"] = int(sum(1 << i for i in partners))
        c["collides_with"] = (
            sorted(classes[i]["name"] for i in partners)
            if False
            else sorted(f"class_{i}" for i in partners)
        )
    realised = set()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = classes[member_class[a]], classes[member_class[b]]
            if (ca["group"] & cb["mask"]) and (cb["group"] & ca["mask"]):
                realised.add((a, b))
    if realised != enabled:
        raise ValueError("class encoding does not reproduce the MuJoCo matrix")
    # Bullet's other filters must not remove an enabled pair: static-vs-static is
    # never tested, and URDF_USE_SELF_COLLISION skips direct parent-child links.
    static = {
        k for k, o in objects.items() if o["kind"] in ("world_static", "robot_fixed")
    }
    parent_of = {j["child"]: j["parent"] for j in joints.values()}
    for a, b in enabled:
        if a in static and b in static:
            raise ValueError(
                f"{a}/{b}: MuJoCo collides two static objects, Bullet never does"
            )
        robot_pair = objects[a]["kind"].startswith("robot") and objects[b][
            "kind"
        ].startswith("robot")
        if robot_pair and (parent_of.get(a) == b or parent_of.get(b) == a):
            raise ValueError(
                f"{a}/{b}: parent-child pair enabled; needs URDF_USE_SELF_COLLISION_INCLUDE_PARENT"
            )
    robot_robot = sorted(
        [a, b]
        for a, b in enabled
        if objects[a]["kind"].startswith("robot")
        and objects[b]["kind"].startswith("robot")
    )

    # ----- friction: MuJoCo max rule realised under Bullet's product rule ------------
    # MuJoCo (equal priority): each coefficient is the max over the pair, torsional
    # only when max(condim) >= 4. Bullet: lateral f_a*f_b, spinning s_a*f_b + f_a*s_b.
    # Free parts carry their own coefficients, every other object 1.0 / 0, which
    # is exact for every pair with one free part (the grasp, table and pad pairs).
    bullet = {}
    for k, o in objects.items():
        if o["kind"] == "free":
            bullet[k] = {
                "lateral": o["friction"][0],
                "spinning": o["friction"][1] if o["condim"] >= 4 else 0.0,
            }
        else:
            bullet[k] = {"lateral": 1.0, "spinning": 0.0}
        bullet[k]["rolling"] = 0.0
    pairs = []
    for a, b in sorted(enabled):
        oa, ob = objects[a], objects[b]
        mj_slide = max(oa["friction"][0], ob["friction"][0])
        mj_tors = (
            max(oa["friction"][1], ob["friction"][1])
            if max(oa["condim"], ob["condim"]) >= 4
            else 0.0
        )
        bl_slide = bullet[a]["lateral"] * bullet[b]["lateral"]
        bl_tors = (
            bullet[a]["spinning"] * bullet[b]["lateral"]
            + bullet[a]["lateral"] * bullet[b]["spinning"]
        )
        pairs.append(
            {
                "pair": [a, b],
                "mujoco_sliding": mj_slide,
                "bullet_lateral": bl_slide,
                "mujoco_torsional": mj_tors,
                "bullet_spinning": bl_tors,
                "with_free_part": "free" in (oa["kind"], ob["kind"]),
            }
        )
    one_free = [
        p
        for p in pairs
        if p["with_free_part"]
        and not (
            objects[p["pair"][0]]["kind"] == objects[p["pair"][1]]["kind"] == "free"
        )
    ]
    exact_free = max(
        max(
            abs(p["mujoco_sliding"] - p["bullet_lateral"]),
            abs(p["mujoco_torsional"] - p["bullet_spinning"]),
        )
        for p in one_free
    )
    if exact_free > 1e-12:
        raise ValueError("friction scheme is not exact for single-part pairs")

    # ----- actuators (after Scene's finger override) ------------------------------------
    actuators = {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        name = m.joint(jid).name
        bias = m.actuator_biasprm[a]
        kind = "position" if bias[1] != 0 else "velocity"
        fr = m.actuator_forcerange[a].tolist()
        if not m.actuator_forcelimited[a] or fr[0] != -fr[1]:
            raise ValueError(f"{name}: expected a symmetric force range")
        actuators[name] = {
            "kind": kind,
            "gain": float(m.actuator_gainprm[a, 0]),
            "force_limit": fr[1],
            "ctrlrange": m.actuator_ctrlrange[a].tolist(),
            "mujoco_joint_damping": float(m.dof_damping[m.jnt_dofadr[jid]]),
            "mujoco_armature": float(m.dof_armature[m.jnt_dofadr[jid]]),
        }

    # ----- cameras, lights, sites ---------------------------------------------------------
    rig = json.loads((ROOT / "sim/camera_rig.json").read_text())
    cameras = {}
    worst_cam = 0.0
    for i in range(m.ncam):
        name = m.camera(i).name
        parent = body_names[m.cam_bodyid[i]]
        R_parent = m.cam_mat0[i].reshape(3, 3)
        entry = {
            "parent_body": parent,
            "pos_in_parent": m.cam_pos[i].tolist(),
            "axes_in_parent": R_parent.tolist(),
            "fovy_deg": float(m.cam_fovy[i]),
            "convention": "MuJoCo camera axes (columns): x right, y up, looks along -z",
        }
        if parent == "world":
            entry["role"] = "display"
        else:
            sid = m.site(name).id
            R_site = quat_matrix(m.site_quat[sid])
            link = next(
                k
                for k, v in link_map.items()
                if v["mujoco"] == "site" and v["name"] == name
            )
            R_link_cam = R_site.T @ R_parent
            roll = rig["cameras"][name]["optical_roll_deg"]
            convention = Rotation.from_euler(
                "z", roll, degrees=True
            ).as_matrix() @ np.diag([1.0, -1.0, -1.0])
            err = float(np.abs(R_link_cam - convention).max())
            worst_cam = max(
                worst_cam, err, float(np.linalg.norm(m.cam_pos[i] - m.site_pos[sid]))
            )
            entry.update(
                role="policy",
                urdf_link=link,
                axes_in_link=R_link_cam.tolist(),
                pos_in_link=(R_site.T @ (m.cam_pos[i] - m.site_pos[sid])).tolist(),
                optical_roll_deg=roll,
                rig_convention_error=err,
            )
        cameras[name] = entry
    if worst_cam > 1e-7:  # xyaxes are printed with 9 significant digits
        raise ValueError(
            f"camera_rig convention does not reproduce the MuJoCo cameras ({worst_cam})"
        )
    lights = [
        {
            "name": m.light(i).name,
            "dir": m.light_dir[i].tolist(),
            "diffuse": m.light_diffuse[i].tolist(),
            "specular": m.light_specular[i].tolist(),
            "castshadow": bool(m.light_castshadow[i]),
        }
        for i in range(m.nlight)
    ]
    sites = {
        f"{side}_grasp": {
            "body": body_names[m.site_bodyid[m.site(f"{side}_grasp").id]],
            "pos": m.site_pos[m.site(f"{side}_grasp").id].tolist(),
        }
        for side in ("left", "right")
    }

    # ----- world geoms and free bodies -------------------------------------------------------
    def geom_record(g):
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        return {
            "type": MJ_GEOM[int(m.geom_type[g])],
            "pos": m.geom_pos[g].tolist(),
            "quat_xyzw": xyzw(m.geom_quat[g]),
            "half_extents": m.geom_size[g].tolist(),
            "rgba": [float(v) for v in rgba],
            "mujoco_friction": m.geom_friction[g].tolist(),
            "condim": int(m.geom_condim[g]),
            "solref": m.geom_solref[g].tolist(),
            "solimp": m.geom_solimp[g].tolist(),
        }

    # Box geoms also get a one-link URDF each (box collision + visual), so an
    # engine can load them like the robot (PyBullet needs that for SAT features).
    if OBJECTS_DIR.exists():
        shutil.rmtree(OBJECTS_DIR)
    OBJECTS_DIR.mkdir(parents=True)

    def object_urdf(
        name, rec, mass=0.0, inertia=(0.0, 0.0, 0.0), half=None, offset=(0, 0, 0)
    ):
        robot = ET.Element("robot", name=name)
        robot.append(
            ET.Comment(" Generated by convert_pybullet_assets.py; do not edit. ")
        )
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.17g}")
        ixx, iyy, izz = (f"{v:.17g}" for v in inertia)
        ET.SubElement(
            inertial, "inertia", ixx=ixx, iyy=iyy, izz=izz, ixy="0", ixz="0", iyz="0"
        )
        size = g17(2 * np.asarray(rec["half_extents"] if half is None else half))
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=g17(offset), rpy="0 0 0")
        ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=size)
        material = ET.SubElement(visual, "material", name=f"{name}_color")
        ET.SubElement(material, "color", rgba=g17(rec["rgba"]))
        collision = ET.SubElement(link, "collision", name=name)
        ET.SubElement(collision, "origin", xyz=g17(offset), rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=size)
        ET.indent(robot)
        path = OBJECTS_DIR / f"{name}.urdf"
        path.write_text(
            '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"
        )
        return str(path)

    world = {}
    for key, o in objects.items():
        if o["kind"] == "world_static":
            (g,) = o["geoms"]
            world[key] = geom_record(g)
            if world[key]["type"] == "box":
                world[key]["urdf"] = object_urdf(key, world[key])
            elif world[key]["type"] == "plane":
                # A MuJoCo plane collides as an infinite half-space. Realised as a
                # box over the plane's 6 x 6 m extent, 1 m deep, top face on the
                # plane (Bullet's plane shape would inflate hull contacts by 1 mm).
                half = [*world[key]["half_extents"][:2], 0.5]
                world[key]["realised_as"] = {
                    "type": "box",
                    "half_extents": half,
                    "offset": [0, 0, -0.5],
                }
                world[key]["urdf"] = object_urdf(
                    key, world[key], half=half, offset=(0, 0, -0.5)
                )
            else:
                raise ValueError(
                    f"{key}: world geom type {world[key]['type']} not converted"
                )
    free = {}
    for body in free_bodies:
        bid = m.body(body).id
        (g,) = objects[body]["geoms"]
        jid = next(j for j in range(m.njnt) if m.jnt_bodyid[j] == bid)
        qadr = int(m.jnt_qposadr[jid])
        free[body] = {
            "joint": m.joint(jid).name,
            "geom": geom_record(g),
            "mass_kg": float(m.body_mass[bid]),
            "inertia_diag": m.body_inertia[bid].tolist(),
            "inertial_pos": m.body_ipos[bid].tolist(),
            "inertial_quat_xyzw": xyzw(m.body_iquat[bid]),
            "position": m.qpos0[qadr : qadr + 3].tolist(),
            "orientation_xyzw": xyzw(m.qpos0[qadr + 3 : qadr + 7]),
        }
        assert np.allclose(m.geom_pos[g], 0) and np.allclose(m.body_ipos[bid], 0), body
        assert np.allclose(m.body_iquat[bid], [1, 0, 0, 0]), body
        if free[body]["geom"]["type"] != "box":
            raise ValueError(f"{body}: only box parts are converted")
        free[body]["urdf"] = object_urdf(
            body, free[body]["geom"], free[body]["mass_kg"], free[body]["inertia_diag"]
        )

    scene_json = {
        "schema": 1,
        "generated_by": "sdk/convert_pybullet_assets.py",
        "generated_utc": report["generated_utc"],
        "urdf": str(URDF_OUT),
        "inputs": report["inputs"],
        "strict_inputs": STRICT_INPUTS,
        "physics": {
            "timestep_s": float(m.opt.timestep),
            "gravity": m.opt.gravity.tolist(),
            "mujoco_cone": "elliptic"
            if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
            else "pyramidal",
            "mujoco_iterations": int(m.opt.iterations),
            "mujoco_integrator": int(m.opt.integrator),
        },
        "base": {
            "link": root_link,
            "position": base_p.tolist(),
            "orientation_xyzw": xyzw(m.body_quat[base_id]),
            "fixed": True,
        },
        "world_geoms": world,
        "free_bodies": free,
        "robot_links": {
            k: {
                "kind": o["kind"],
                "geoms": [m.geom(g).name for g in o["geoms"]],
                "contype": o["contype"],
                "conaffinity": o["conaffinity"],
                "mujoco_friction": o["friction"],
                "condim": o["condim"],
            }
            for k, o in objects.items()
            if o["kind"].startswith("robot")
        },
        "collision_filter": {
            "bullet_mode": "default AND test: (groupA & maskB) and (groupB & maskA)",
            "self_collision": "URDF_USE_SELF_COLLISION (direct parent-child skipped; none is enabled)",
            "classes": classes,
            "object_class": {k: classes[member_class[k]]["name"] for k in names},
            "objects": {
                k: {"kind": o["kind"], "mujoco_body": o["mujoco_body"]}
                for k, o in objects.items()
            },
            "enabled_pairs": sorted([a, b] for a, b in enabled),
        },
        "friction": {
            "mujoco_rule": "max of each coefficient over the pair (equal geom priority); torsional only if max condim >= 4",
            "bullet_rule": "lateral f_a*f_b; spinning s_a*f_b + f_a*s_b; rolling r_a*f_b + f_a*r_b",
            "bullet": bullet,
            "pairs": pairs,
        },
        "actuators": actuators,
        "cameras": cameras,
        "sites": sites,
        "render": {
            "znear_m": float(m.vis.map.znear * m.stat.extent),
            "mujoco_zfar_m": float(m.vis.map.zfar * m.stat.extent),
            "lights": lights,
            "headlight": {
                "ambient": m.vis.headlight.ambient.tolist(),
                "diffuse": m.vis.headlight.diffuse.tolist(),
            },
        },
        "visuals": visuals,
        # The backend re-hashes these at load, so a hand-edited or partially
        # regenerated asset set is refused.
        "outputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                URDF_OUT,
                *sorted(COLLISION_DIR.glob("*.obj")),
                *sorted(OBJECTS_DIR.glob("*.urdf")),
            ]
        },
    }
    SCENE_OUT.write_text(json.dumps(scene_json, indent=1))

    report["colliders"] = {
        "robot_geoms_by_type": counts,
        "per_link": {k: len(v) for k, v in per_link.items()},
        "wrist_hulls": {
            s: sum(e["type"] == "mesh" for e in per_link[f"{s}_wrist_roll_Link"])
            for s in ("left", "right")
        },
        "finger_hulls": {
            k: {
                x: v[0][x]
                for x in (
                    "vertices",
                    "faces",
                    "volume_m3",
                    "hull_source",
                    "mesh_vertices",
                )
            }
            for k, v in per_link.items()
            if "finger" in k
        },
        "world_geoms": sorted(world),
        "free_bodies": sorted(free),
        "build_report": scene.setup["wrist_collision"],
    }
    report["contact_matrix"] = {
        "objects": len(names),
        "pairs_total": len(names) * (len(names) - 1) // 2,
        "enabled_pairs": len(enabled),
        "robot_robot_enabled": robot_robot,
        "classes": {
            c["name"]: {"members": c["members"], "group": c["group"], "mask": c["mask"]}
            for c in classes
        },
        "class_encoding_reproduces_mujoco": True,
        "mujoco_random_pose_contacts": {
            "poses": len(cases),
            "object_pairs_in_contact": len(seen),
            "outside_enabled_matrix": stray,
        },
    }
    report["friction"] = {
        "single_free_part_pairs_max_error": exact_free,
        "other_pairs_max_error": max(
            max(
                abs(p["mujoco_sliding"] - p["bullet_lateral"]),
                abs(p["mujoco_torsional"] - p["bullet_spinning"]),
            )
            for p in pairs
        ),
        "mismatched_pairs": [
            p
            for p in pairs
            if abs(p["mujoco_sliding"] - p["bullet_lateral"]) > 1e-12
            or abs(p["mujoco_torsional"] - p["bullet_spinning"]) > 1e-12
        ],
    }
    report["cameras"] = {
        "rig_convention_max_error": worst_cam,
        "names": sorted(cameras),
    }

    report["outputs"] = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in [
            URDF_OUT,
            SCENE_OUT,
            *sorted(COLLISION_DIR.glob("*.obj")),
            *sorted(OBJECTS_DIR.glob("*.urdf")),
        ]
    }
    report["wall_time_s"] = time.monotonic() - started
    REPORT_OUT.write_text(json.dumps(report, indent=1))
    summary = {
        "urdf": str(URDF_OUT.relative_to(ROOT)),
        "colliders": counts,
        "frame_check_max_m": worst["position_m"],
        "enabled_pairs": len(enabled),
        "classes": len(classes),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
