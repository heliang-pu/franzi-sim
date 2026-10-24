"""Check the tilted wrist D405 block against the gripper CAD and render it.

    /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python \
        hardware/wrist_d405_mount/check_mount.py

Writes, next to this file: mount_check.json (clearances, aim, and the camera
pose in each wrist_roll_Link frame for the simulation), assembly.png (gripper,
existing L bracket, new block, D405) and wrist_views.png (simulated wrist
images at the expert's align and grasp ticks: current mount vs new block).
Clearances are nearest distances between dense surface samples of the CAD
meshes; the D405 is the 42 x 42 x 23 mm housing of the datasheet. The pads
drop as the jaws close, along the CAD linkage (sim/gripper_linkage.py).
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from sim.gripper_linkage import STROKE, contact_drop, drop, set_closure
from sim.wrist_mount import HINGE, OLD_BLOCK_BOXES, P, camera, part_to_wrist, sign

MESHES = ROOT / "assets/franzi_source/meshes"
JAW_CENTRE = np.array([-30.45, -1.75, -241.0])  # centre of the open pads
KIT_PART_MM = 40.0  # the sort-and-kit parts: the pads close on them here


def part_mesh(side):
    """The printed block placed on the side's clevis pin (mm)."""
    m = trimesh.load(HERE / "wrist_d405_mount.stl")
    R, t = part_to_wrist(side)
    return trimesh.Trimesh(m.vertices @ R.T + t, m.faces)


def samples(mesh, count=60000, seed=0):
    pts, _ = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
    return np.vstack([pts, mesh.vertices])


def housing_points(back, axis, right, up, step=1.0):
    g = np.arange(-21, 21.01, step)
    pts = []
    for a in g:
        for b in g:
            for c in (0.0, 23.0):
                pts.append(back + right * a + up * b + axis * c)
    for a in g:
        for c in np.arange(0, 23.01, step):
            for b in (-21.0, 21.0):
                pts.append(back + right * a + up * b + axis * c)
                pts.append(back + right * b + up * a + axis * c)
    return np.array(pts)


# The L bracket in left_wrist_roll_Link.STL, by bounding box (mm); the right
# gripper's bracket side is the mirror image (its mechanism is not). The
# original camera block's boxes are in sim/wrist_mount.py.
BRACKET_BOX = ((-39.7, 36.6, -102.0), (-20.9, 70.2, -72.0))


def jaw_centre(side, width=None):
    """Pad centre with the jaws open, or closed on a part of width (mm): the
    pads drop along the fingers as they close."""
    centre = JAW_CENTRE * [1, sign(side), 1]
    if width is not None:
        centre = centre - [0, 0, 1000 * contact_drop(width / 1000)]
    return centre


def wrist_parts(side="left"):
    """(gripper, existing L bracket, original camera block) of one side, mm."""
    mesh = trimesh.load(MESHES / f"{side}_wrist_roll_Link.STL")
    parts = [
        trimesh.Trimesh(p.vertices * 1000, p.faces)
        for p in mesh.split(only_watertight=False)
    ]

    def match(box):
        lo, hi = np.array(box)
        if side == "right":
            lo, hi = lo * [1, -1, 1], hi * [1, -1, 1]
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        found = [
            i for i, p in enumerate(parts) if np.abs(p.bounds - [lo, hi]).max() < 0.6
        ]
        if len(found) != 1:
            raise ValueError(f"{side}: {len(found)} CAD parts match {box}")
        return found[0]

    bracket = match(BRACKET_BOX)
    old = {match(box) for box in OLD_BLOCK_BOXES}
    gripper = trimesh.util.concatenate(
        [p for i, p in enumerate(parts) if i not in old | {bracket}]
    )
    return gripper, parts[bracket], trimesh.util.concatenate([parts[i] for i in old])


def fingers(closure, side="left"):
    out = []
    y = -5.0 * sign(side)
    z = -229.04 - 1000 * drop(STROKE * closure)
    for n, origin, direction in ((1, -77.95, 1), (2, 17.05, -1)):
        m = trimesh.load(MESHES / f"{side}finger{n}_Link.STL")
        v = m.vertices * 1000 + [origin + direction * 47.5 * closure, y, z]
        out.append(trimesh.Trimesh(v, m.faces))
    return trimesh.util.concatenate(out)


def check_side(side):
    gripper, bracket, _ = wrist_parts(side)
    trees = {
        "gripper": cKDTree(samples(gripper, 150000)),
        "bracket": cKDTree(samples(bracket, 20000)),
    }
    part = part_mesh(side)
    p_pts = samples(part, 40000)
    # The 5 mm tongue sits in the 5.4 mm clevis gap and bears on one prong, as
    # the original block does; only the wider web must clear the prongs.
    in_clevis = np.abs(p_pts[:, 0] - HINGE[0]) <= P["tongue_thickness_mm"] / 2 + 1e-6
    back, axis, right, up = camera(side)
    h_pts = housing_points(back, axis, right, up)
    eye = back + 23 * axis
    to_jaw = jaw_centre(side) - eye
    to_grip = jaw_centre(side, KIT_PART_MM) - eye
    out = {
        "clearance_mm": {
            "block_to_gripper": round(float(trees["gripper"].query(p_pts)[0].min()), 2),
            "wide_web_to_bracket": round(
                float(trees["bracket"].query(p_pts[~in_clevis])[0].min()), 2
            ),
            "d405_to_gripper": round(float(trees["gripper"].query(h_pts)[0].min()), 2),
            "d405_to_bracket": round(float(trees["bracket"].query(h_pts)[0].min()), 2),
        },
        "aim": {
            "jaw_centre_off_axis_deg": round(
                float(np.degrees(np.arccos(to_jaw @ axis / np.linalg.norm(to_jaw)))), 2
            ),
            "glass_to_jaw_centre_mm": round(float(np.linalg.norm(to_jaw)), 1),
            "grip_centre_40mm_off_axis_deg": round(
                float(np.degrees(np.arccos(to_grip @ axis / np.linalg.norm(to_grip)))),
                2,
            ),
            "glass_to_grip_centre_40mm_mm": round(float(np.linalg.norm(to_grip)), 1),
        },
    }
    for closure in (0.0, 0.55, 1.0):
        f = cKDTree(samples(fingers(closure, side), 30000))
        out["clearance_mm"][f"d405_to_fingers_closure_{closure}"] = round(
            float(f.query(h_pts)[0].min()), 2
        )
        out["clearance_mm"][f"block_to_fingers_closure_{closure}"] = round(
            float(f.query(p_pts)[0].min()), 2
        )
    # Where the finger pads land in the image, normalised to the D405 colour
    # FOV (84 x 58 degrees): +-1 is the image edge.
    fov = P["d405"]["color_fov_deg"]
    sy = sign(side)
    pads = {
        "open_pads": np.array([[-84.0, -1.75 * sy, -241], [23.0, -1.75 * sy, -241]])
    }
    for width in (55.0, KIT_PART_MM):
        centre = jaw_centre(side, width)
        pads[f"closed_on_{width:.0f}mm_part"] = centre + np.outer(
            [-1, 1], [width / 2, 0, 0]
        )
        pads[f"grip_centre_{width:.0f}mm"] = centre[None]
    pads["jaw_centre"] = jaw_centre(side)[None]
    for name, pts in pads.items():
        d = pts - eye
        z = d @ axis
        u = (d @ right) / z / np.tan(np.radians(fov["h"] / 2))
        v = (d @ up) / z / np.tan(np.radians(fov["v"] / 2))
        out.setdefault("image_uv", {})[name] = np.round(np.c_[u, v], 2).tolist()
    rot = np.stack([right, up, -axis], axis=1)  # MuJoCo camera: x right, y up, z back
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.reshape(-1))
    out["sim_camera_pose"] = {
        "frame": f"{side}_wrist_roll_Link",
        "camera_pos_m": np.round(eye / 1000, 6).tolist(),
        "camera_quat_wxyz": np.round(q, 6).tolist(),
        "optical_axis": np.round(axis, 6).tolist(),
        "image_right": np.round(right, 6).tolist(),
        "image_up": np.round(up, 6).tolist(),
    }
    return out


def check():
    part = part_mesh("left")
    return {
        "params": {k: P[k] for k in ("camera_tilt_deg", "standoff_mm", "outward_mm")},
        "part_volume_cm3": round(float(part.volume) / 1000, 2),
        "part_watertight": bool(part.is_watertight),
        "left": check_side("left"),
        "right": check_side("right"),
    }


def assembly_png(path):
    gripper, bracket, _ = wrist_parts()
    meshes = {"gripper": (gripper, ".45 .45 .45 1"), "bracket": (bracket, ".85 .2 .2 1"),
              "block": (part_mesh("left"), "1 .55 .1 1")}  # fmt: skip
    back, axis, right, up = camera("left")
    corners = np.array([back + right * a + up * b + axis * c
                        for a in (-21, 21) for b in (-21, 21) for c in (0, 23)])  # fmt: skip
    meshes["d405"] = (trimesh.convex.convex_hull(corners), ".95 .95 .95 1")
    meshes["fingers"] = (fingers(0.0), ".7 .7 .72 1")
    tmp = HERE / "_render"
    tmp.mkdir(exist_ok=True)
    assets, geoms = "", ""
    for name, (mesh, rgba) in meshes.items():
        trimesh.Trimesh(mesh.vertices / 1000, mesh.faces).export(tmp / f"{name}.stl")
        assets += f'<mesh name="{name}" file="{name}.stl"/>'
        geoms += f'<geom type="mesh" mesh="{name}" rgba="{rgba}"/>'
    jaw = jaw_centre("left") / 1000
    eye = (back + 23 * axis) / 1000
    geoms += (
        f'<geom type="capsule" fromto="{" ".join(map(str, eye))} {" ".join(map(str, jaw))}" '
        'size="0.0007" rgba="0 .6 1 1"/>'
    )
    xml = (
        f'<mujoco><compiler meshdir="{tmp}"/><asset>{assets}</asset><visual>'
        '<global offwidth="800" offheight="600"/><headlight ambient=".45 .45 .45" '
        'diffuse=".55 .55 .55"/><map znear="0.001"/></visual><worldbody>'
        f'<light pos=".3 .3 .4" dir="-1 -1 -1"/>{geoms}</worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 600, 800)
    tiles = []
    views = (
        ("along the jaw axis", 0, 0),
        ("outboard side", 90, 0),
        ("from above", 120, 35),
    )
    for label, az, el in views:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [-0.03, 0.02, -0.16]
        cam.azimuth, cam.elevation, cam.distance = az, el, 0.36
        r.update_scene(data, camera=cam)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((8, 8), f"{label}: grey gripper, red existing L bracket, "
                                "orange new block, white D405, blue optical axis", fill=(255, 255, 255))  # fmt: skip
        tiles.append(np.asarray(im))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def wrist_views_png(path, run=ROOT / "outputs/expert_wrist_block"):
    from scenes.pick_place.run import PickPlaceScene

    tmp = HERE / "_scene"
    tmp.mkdir(exist_ok=True)
    s = PickPlaceScene(tmp)
    m, d = s.m, s.d
    traj = np.load(run / "trajectory.npz")
    head = m.joint("head_pitch_joint").qposadr[0]
    r = mujoco.Renderer(m, 480, 640)
    rows = []
    phases = list(traj["subtask"])
    ticks = [len(phases) - 1 - phases[::-1].index(p) for p in ("align", "grasp")]
    for k, label in zip(ticks, ("align, jaws open", "grasp, jaws closed")):
        d.qpos[:] = m.qpos0
        q = traj["state"][k]
        for i, side in enumerate(("left", "right")):
            d.qpos[s.qids[side]] = q[7 * i : 7 * i + 7]
            set_closure(m, d, side, q[14 + i])
        d.qpos[head] = 0.5
        for side in ("left", "right"):
            cid = m.camera(f"{side}_wrist_d405").id
            saved = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
            tiles = []
            # The scene mounts the D405 on the new block; the original block
            # is camera(side, 0, 17, 0) on the same clevis pin.
            for name, args in (("original block", (0, 17, 0)), ("new block, as the sim mounts it", None)):  # fmt: skip
                if args is not None:
                    back, axis, right, up = camera(side, *args)
                    q4 = np.zeros(4)
                    mujoco.mju_mat2Quat(q4, np.stack([right, up, -axis], 1).reshape(-1))
                    m.cam_pos[cid], m.cam_quat[cid] = (back + 23 * axis) / 1000, q4
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=f"{side}_wrist_d405")
                im = Image.fromarray(r.render()).resize((400, 300))
                ImageDraw.Draw(im).text(
                    (6, 6), f"{side} wrist | {name} | {label}", fill=(255, 255, 0)
                )
                tiles.append(np.asarray(im))
                m.cam_pos[cid], m.cam_quat[cid] = saved
            rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def part_png(path):
    """The printed block alone (part frame, mm), three views, 10 mm scale bar."""
    m = trimesh.load(HERE / "wrist_d405_mount.stl")
    tmp = HERE / "_render"
    tmp.mkdir(exist_ok=True)
    trimesh.Trimesh(m.vertices / 1000, m.faces).export(tmp / "part.stl")
    bar = '<geom type="box" pos="0.012 0 -0.036" size="0.005 0.0004 0.0004" rgba="1 1 1 1"/>'
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
    for label, az, el in (
        ("along the pin (x)", 0, 0),
        ("outboard (y)", 90, 0),
        ("3/4 view", 40, 25),
    ):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0.007, -0.012]
        cam.azimuth, cam.elevation, cam.distance = az, el, 0.11
        r.update_scene(data, camera=cam)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text(
            (8, 8), f"{label}; white bar = 10 mm", fill=(255, 255, 255)
        )
        tiles.append(np.asarray(im))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    (tmp / "part.stl").unlink()
    tmp.rmdir()


if __name__ == "__main__":
    report = check()
    (HERE / "mount_check.json").write_text(json.dumps(report, indent=2))
    brief = {
        k: {n: x for n, x in v.items() if n != "sim_camera_pose"}
        if isinstance(v, dict) and "sim_camera_pose" in v
        else v
        for k, v in report.items()
    }
    print(json.dumps(brief, indent=2))
    part_png(HERE / "part.png")
    assembly_png(HERE / "assembly.png")
    wrist_views_png(HERE / "wrist_views.png")
    print("wrote mount_check.json, part.png, assembly.png, wrist_views.png")
