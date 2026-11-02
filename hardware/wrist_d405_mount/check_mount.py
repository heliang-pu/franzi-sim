"""Check the wrist D405 adapter against the gripper CAD and render it.

    MUJOCO_GL=egl python hardware/wrist_d405_mount/check_mount.py  (on the pro6000)

Writes, next to this file: mount_check.json (clearances, aim, where the pads
land in the image), part.png (the printed part), assembly.png (gripper, the
adapter, the D405 and its optical axis) and wrist_views.png (the simulated
wrist images: the D405 where the URDF has it, straight on the strip, against
the adapter). The gripper is the scene's (sim.build_task): the painted CAD with
its linkage posed at each closure, so the pads drop as the jaws close.
Clearances are nearest distances between dense surface samples; the D405 is
the 42 x 42 x 23 mm housing of the datasheet. The adapter sits on the strip's
pad, so its distance is to the gripper beyond the pad's top face.
"""

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from robot_kinematics import ARM_JOINTS, Kinematics
from sim import tabletop
from sim.build_task import build
from sim.camera_rig import install_cameras
from sim.gripper_linkage import JAW_SPAN, contact_drop, set_closure
from sim.wrist_mount import DEPTH, PAD, P, camera

KIT_PART_MM = 40.0  # the sort-and-kit parts: the pads close on them here
CLOSURES = (0.0, 0.55, 1.0)
LINKS = {"left": "left_gripper_link", "right": "right_gripper_Link"}


def scene():
    root, _ = build()
    install_cameras(root)
    size = root.find("visual").find("global")
    if size is None:
        size = ET.SubElement(root.find("visual"), "global")
    size.set("offwidth", "800")
    size.set("offheight", "600")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return m, mujoco.MjData(m)


def subtree(m, top):
    out = {top}
    for b in range(top + 1, m.nbody):  # bodies are numbered parents first
        if m.body_parentid[b] in out:
            out.add(b)
    return out


def geom_mesh(m, d, g, frame):
    """A mesh geom's surface in a body's frame (mm)."""
    mid = m.geom_dataid[g]
    v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    world = d.geom_xpos[g] + m.mesh_vert[v0 : v0 + nv] @ d.geom_xmat[g].reshape(3, 3).T
    R, t = d.xmat[frame].reshape(3, 3), d.xpos[frame]
    faces = m.mesh_face[f0 : f0 + nf]
    return trimesh.Trimesh((world - t) @ R * 1000, faces, process=False)


def samples(mesh, density=4.0):
    count = max(200, int(mesh.area * density))
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
    return np.vstack([points, mesh.vertices])


def housing(eye, axis, right, up, step=1.0):
    """The D405 housing's surface points (mm): 42 x 42 x 23 behind the glass."""
    g = np.arange(-21, 21.01, step)
    d = np.arange(0, DEPTH + 0.01, step)
    pts = [
        eye - axis * c + right * a + up * b for a in g for b in g for c in (0, DEPTH)
    ]
    for a in g:
        for c in d:
            for s in (-21, 21):
                pts += [
                    eye - axis * c + right * a + up * s,
                    eye - axis * c + right * s + up * a,
                ]
    return np.array(pts)


def check_side(m, d, side):
    link = m.body(LINKS[side]).id
    adapter = m.geom(f"{side}_wrist_d405_block").id
    bodies = subtree(m, link)
    cam = m.camera(f"{side}_wrist_d405").id
    tcp = m.site(f"{side}_grasp").id
    glass, axis, _, _ = camera()
    eye = PAD + glass
    out = {"clearance_mm": {}}
    for closure in CLOSURES:
        set_closure(m, d, side, closure)
        mujoco.mj_forward(m, d)
        R = d.xmat[link].reshape(3, 3)
        right, up = (R.T @ d.cam_xmat[cam].reshape(3, 3))[:, :2].T
        d405 = housing(eye, axis, right, up)
        part = samples(geom_mesh(m, d, adapter, link), 20)
        clouds = {"gripper": [], "fingers": []}
        for g in range(m.ngeom):
            visual = m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
            name = m.geom(g).name
            if (
                visual
                and m.geom_bodyid[g] in bodies
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and g != adapter
                and not name.startswith(f"real_{side}_D405")
            ):
                body = m.body(m.geom_bodyid[g]).name
                kind = "fingers" if "finger" in body else "gripper"
                clouds[kind].append(samples(geom_mesh(m, d, g, link)))
        for kind, points in clouds.items():
            points = np.vstack(points)
            near = cKDTree(points).query(d405)[0].min()
            beyond = cKDTree(points[points[:, 2] > PAD[2] + 0.5])  # past the pad
            c = out["clearance_mm"]
            c[f"d405_to_{kind}_closure_{closure}"] = round(float(near), 2)
            c[f"adapter_to_{kind}_closure_{closure}"] = round(
                float(beyond.query(part)[0].min()), 2
            )
        if closure == 0.0:
            frame = R.T @ d.site_xmat[tcp].reshape(3, 3)
            centre = R.T @ (d.site_xpos[tcp] - d.xpos[link]) * 1000
            jaws, down = frame[:, 0], -frame[:, 2]  # closing; toward the fingertips
    set_closure(m, d, side, 0.0)
    # Aim, and where pads land in the image, normalised to the D405 colour FOV
    # (84 x 58 degrees): +-1 is the image edge.
    fov = P["d405"]["color_fov_deg"]
    points = {
        "open_pads": centre + np.outer([-1, 1], jaws * JAW_SPAN * 500),
        "jaw_centre": centre[None],
    }
    for width in (55.0, KIT_PART_MM):
        held = centre + down * 1000 * contact_drop(width / 1000)
        points[f"closed_on_{width:.0f}mm_part"] = held + np.outer(
            [-1, 1], jaws * width / 2
        )
        points[f"grip_centre_{width:.0f}mm"] = held[None]
    for name, pts in points.items():
        rel = pts - eye
        z = rel @ axis
        u = (rel @ right) / z / np.tan(np.radians(fov["h"] / 2))
        v = (rel @ up) / z / np.tan(np.radians(fov["v"] / 2))
        out.setdefault("image_uv", {})[name] = np.round(np.c_[u, v], 2).tolist()
    to_jaw = centre - eye
    grip = points[f"grip_centre_{KIT_PART_MM:.0f}mm"][0] - eye

    def off_axis(v):
        return round(float(np.degrees(np.arccos(v @ axis / np.linalg.norm(v)))), 2)

    out["aim"] = {
        "jaw_centre_off_axis_deg": off_axis(to_jaw),
        "glass_to_jaw_centre_mm": round(float(np.linalg.norm(to_jaw)), 1),
        "grip_centre_40mm_off_axis_deg": off_axis(grip),
        "glass_to_grip_centre_40mm_mm": round(float(np.linalg.norm(grip)), 1),
        "lens_beside_jaw_plane_mm": round(float(abs(to_jaw @ np.cross(jaws, down))), 1),
        "lens_behind_grasp_point_mm": round(float(to_jaw @ down), 1),
    }
    return out


def free_camera(lookat, view, distance):
    view = view / np.linalg.norm(view)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = float(np.degrees(np.arctan2(view[1], view[0])) + 180)
    cam.elevation = float(-np.degrees(np.arcsin(np.clip(view[2], -1, 1))))
    return cam


def part_png(path):
    """The printed part alone (part frame, mm), three views, 10 mm scale bar."""
    tmp = Path(tempfile.mkdtemp())
    part = trimesh.load(HERE / "wrist_d405_mount.stl")
    trimesh.Trimesh(part.vertices / 1000, part.faces).export(tmp / "part.stl")
    bar = '<geom type="box" pos="0.005 -0.022 0" size="0.005 0.0004 0.0004" rgba="1 1 1 1"/>'
    xml = (
        f'<mujoco><compiler meshdir="{tmp}"/><asset><mesh name="part" file="part.stl"/>'
        '</asset><visual><global offwidth="600" offheight="600"/><headlight '
        'ambient=".5 .5 .5" diffuse=".5 .5 .5"/></visual><worldbody><light pos=".2 .2 .3" '
        'dir="-1 -1 -1.5"/><geom type="mesh" mesh="part" rgba="1 .55 .1 1"/>'
        f"{bar}</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 600, 600)
    tiles = []
    for label, view in (
        ("along the jaws (y)", [0, 1, 0]),
        ("from outboard (x)", [1, 0, 0.001]),
        ("3/4 view", [1, 0.8, 0.6]),
    ):
        r.update_scene(
            data, camera=free_camera([0.008, 0, 0.022], -np.array(view, float), 0.12)
        )
        im = Image.fromarray(r.render())
        text = f"{label}; white bar = 10 mm; the pad side down"
        ImageDraw.Draw(im).text((8, 8), text, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def assembly_png(m, d, path):
    """The left gripper with the adapter (orange), the D405 (white) and its
    optical axis (blue)."""
    link = m.body(LINKS["left"]).id
    adapter = m.geom("left_wrist_d405_block").id
    rgba, mat = m.geom_rgba.copy(), m.geom_matid.copy()
    m.geom_matid[adapter], m.geom_rgba[adapter] = -1, (1, 0.55, 0.1, 1)
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("real_left_D405"):
            m.geom_matid[g], m.geom_rgba[g] = -1, (0.95, 0.95, 0.95, 1)
    mujoco.mj_forward(m, d)
    R, t = d.xmat[link].reshape(3, 3), d.xpos[link]
    cam = m.camera("left_wrist_d405").id
    eye, ahead = d.cam_xpos[cam], -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    r = mujoco.Renderer(m, 600, 800)
    tiles = []
    for label, view in (
        ("along the jaws", R[:, 1]),
        ("from outboard", R[:, 0]),
        ("oblique", R[:, 0] + R[:, 1] - 0.4 * R[:, 2]),
    ):
        r.update_scene(d, camera=free_camera(t + R @ [0.03, 0, 0.09], -view, 0.34))
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
            np.array([0, 0.6, 1, 1], np.float32),
        )  # fmt: skip
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, eye, eye + ahead * 0.11
        )
        r.scene.ngeom += 1
        im = Image.fromarray(r.render())
        text = f"{label}: orange adapter, white D405, blue optical axis"
        ImageDraw.Draw(im).text((8, 8), text, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    m.geom_rgba[:], m.geom_matid[:] = rgba, mat
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def wrist_views_png(m, d, path):
    """Wrist images at the tabletop ready pose, the jaws open and closed on a
    40 mm part: the D405 straight down the hand on the strip, where the URDF
    has it, against the adapter."""
    kin = Kinematics()
    for side in ("left", "right"):
        rotation = tabletop.grasp_rotation(side)
        q = kin.ik(side, tabletop.READY[side], np.zeros(7), rotation=rotation)
        for j, v in zip(ARM_JOINTS, q):
            d.qpos[m.joint(f"{side}_{j}_joint").qposadr[0]] = v
    d.qpos[m.joint("head_pitch_joint").qposadr[0]] = 0.5
    tilt = np.radians(P["camera_tilt_deg"])
    # Turns the adapter's camera back up the hand, about the jaws.
    upright = np.array(
        [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]]
    )
    r = mujoco.Renderer(m, 480, 640)
    rows = []
    for label, closure in (
        ("jaws open", 0.0),
        ("closed on 40 mm", 1 - KIT_PART_MM / 1000 / JAW_SPAN),
    ):
        for side in ("left", "right"):
            set_closure(m, d, side, closure)
        for side in ("left", "right"):
            cid = m.camera(f"{side}_wrist_d405").id
            assert m.cam_bodyid[cid] == m.body(LINKS[side]).id
            saved = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
            frame = np.zeros(9)
            mujoco.mju_quat2Mat(frame, saved[1])
            straight = np.zeros(4)
            mujoco.mju_mat2Quat(straight, (upright @ frame.reshape(3, 3)).reshape(-1))
            tiles = []
            # The adapter and the D405 on it stay out of the URDF mount's view.
            mounted = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name == f"{side}_wrist_d405_block"
                or m.geom(g).name.startswith(f"real_{side}_D405")
            ]
            rgba = m.geom_rgba.copy()
            for name in ("D405 on the strip (URDF)", "on the adapter"):
                if name.startswith("D405"):
                    m.cam_pos[cid] = (PAD + [0, 0, DEPTH]) / 1000
                    m.cam_quat[cid] = straight
                    m.geom_rgba[mounted, 3] = 0
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=cid)
                im = Image.fromarray(r.render()).resize((400, 300))
                text = f"{side} wrist | {name} | {label}"
                ImageDraw.Draw(im).text((6, 6), text, fill=(255, 255, 0))
                tiles.append(np.asarray(im))
                m.cam_pos[cid], m.cam_quat[cid] = saved
                m.geom_rgba[:] = rgba
            rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


if __name__ == "__main__":
    m, d = scene()
    part = trimesh.load(HERE / "wrist_d405_mount.stl")
    report = {
        "params": {k: P[k] for k in ("camera_tilt_deg", "glass_centre_mm")},
        "part_volume_cm3": round(float(part.volume) / 1000, 2),
        "part_watertight": bool(part.is_watertight),
        "left": check_side(m, d, "left"),
        "right": check_side(m, d, "right"),
    }
    (HERE / "mount_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    part_png(HERE / "part.png")
    assembly_png(m, d, HERE / "assembly.png")
    wrist_views_png(m, d, HERE / "wrist_views.png")
    print("wrote mount_check.json, part.png, assembly.png, wrist_views.png")
