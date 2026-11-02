"""Export scenes' compiled MuJoCo models for the Isaac Sim port.

    python -m isaac.export <scene> [<scene> ...] --output outputs/<new_dir>

Runs in the project env. Each scene is built as sim/bench_check.py builds it:
the task's nominal layout in a plain Scene, so the model is the one every
MuJoCo run compiles, with the cameras installed and the servo overrides
applied. isaac/stage.py turns the export into a USD stage in Isaac Sim's
Python. <output>/<scene>/ holds:
- scene.xml: the MuJoCo model, as every run writes it;
- model.json: bodies, joints, actuators, joint couplings, geoms, materials,
  textures, lights, cameras, sites, contact exclusions and options, plus a test
  posture with MuJoCo's world poses of every robot body, for the parity check;
- arrays.npz: the compiled mesh arrays (vertices, faces, normals, texture
  coordinates) as MuJoCo holds them;
- textures/: each 2D texture as a PNG, rows flipped so that USD's bottom-up
  texture coordinates sample what MuJoCo's do;
- mujoco/: the MuJoCo scene check (sim/bench_check.py), the reference that
  isaac/check.py compares its preview with.
"""

import argparse
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim import bench_check
from sim.build_task import ARM_JOINTS, FINGER_JOINTS
from sim.camera_rig import POLICY_CAMERAS
from sim.episode import Scene

# The base is fixed in every scene, so the wheel and steering joints are welded
# in Isaac: in MuJoCo their servos hold them at 0 and nothing touches them.
LOCKED_SUFFIXES = ("_wheel_joint", "_steering_joint")
# The robot's start posture (sim/episode.py): the source model's zero posture,
# with the head pitched to 0.50 rad over a 3.3 s minimum-jerk ramp.
HEAD_PITCH, HEAD_RAMP_S = 0.50, 3.3
JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}
MESH_ARRAYS = (
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_vert",
    "mesh_faceadr",
    "mesh_facenum",
    "mesh_face",
    "mesh_normaladr",
    "mesh_normalnum",
    "mesh_normal",
    "mesh_facenormal",
    "mesh_texcoordadr",
    "mesh_texcoordnum",
    "mesh_texcoord",
    "mesh_facetexcoord",
)


def floats(a):
    return [float(x) for x in np.ravel(a)]


def robot_body(m, b):
    # Scene.robot_bodies: scene objects are the bodies named <name>_part.
    return b > 0 and not m.body(b).name.endswith("_part")


def test_posture(m, couplings):
    """A seeded posture of every free robot joint, 20% inside its range, with
    the coupled joints set by their polynomials."""
    rng = np.random.default_rng(0)
    q = m.qpos0.copy()
    followers = {c["joint1"] for c in couplings}
    for j in range(m.njnt):
        name = m.joint(j).name
        if (
            not robot_body(m, m.jnt_bodyid[j])
            # int(): a numpy integer is never `in` a tuple of these enums,
            # which left every joint at zero here before 2026-10-25.
            or int(m.jnt_type[j])
            not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            or name.endswith(LOCKED_SUFFIXES)
            or j in followers
        ):
            continue
        a = m.jnt_qposadr[j]
        lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (q[a] - 0.5, q[a] + 0.5)
        q[a] = rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))
    for c in couplings:
        a1, a2 = m.jnt_qposadr[c["joint1"]], m.jnt_qposadr[c["joint2"]]
        x = q[a2] - m.qpos0[a2]
        q[a1] = m.qpos0[a1] + sum(k * x**i for i, k in enumerate(c["polycoef"]))
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return {
        "qpos": {
            m.joint(j).name: floats(q[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 1])[0]
            for j in range(m.njnt)
            if robot_body(m, m.jnt_bodyid[j])
            and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        },
        "bodies": {
            m.body(b).name: {"xpos": floats(d.xpos[b]), "xquat": floats(d.xquat[b])}
            for b in range(1, m.nbody)
            if robot_body(m, b)
        },
    }


def describe(m, d):
    nrole = m.mat_texid.shape[1] if m.mat_texid.ndim == 2 else 1
    texid = m.mat_texid.reshape(m.nmat, nrole)
    light_type = getattr(m, "light_type", None)
    couplings = [
        {
            "name": m.equality(e).name,
            "joint1": int(m.eq_obj1id[e]),
            "joint2": int(m.eq_obj2id[e]),
            "polycoef": floats(m.eq_data[e][:5]),
        }
        for e in range(m.neq)
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]
    ]
    if any(m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT for e in range(m.neq)):
        raise ValueError("only joint equalities are ported")
    return {
        "options": {
            "timestep": float(m.opt.timestep),
            "gravity": floats(m.opt.gravity),
            "znear": float(m.vis.map.znear * m.stat.extent),
        },
        "bodies": [
            {
                "name": m.body(b).name or "world",
                "parent": int(m.body_parentid[b]),
                "root": int(m.body_rootid[b]),
                "joints": list(
                    range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                ),
                "robot": robot_body(m, b),
                "pos": floats(m.body_pos[b]),
                "quat": floats(m.body_quat[b]),
                "xpos": floats(d.xpos[b]),
                "xquat": floats(d.xquat[b]),
                "mass": float(m.body_mass[b]),
                "ipos": floats(m.body_ipos[b]),
                "iquat": floats(m.body_iquat[b]),
                "inertia": floats(m.body_inertia[b]),
            }
            for b in range(m.nbody)
        ],
        "joints": [
            {
                "name": m.joint(j).name,
                "type": JOINT_TYPES[int(m.jnt_type[j])],
                "body": int(m.jnt_bodyid[j]),
                "pos": floats(m.jnt_pos[j]),
                "axis": floats(m.jnt_axis[j]),
                "limited": bool(m.jnt_limited[j]),
                "range": floats(m.jnt_range[j]),
                "qpos0": floats(
                    m.qpos0[
                        m.jnt_qposadr[j] : m.jnt_qposadr[j]
                        + (7 if m.jnt_type[j] == 0 else 1)
                    ]
                ),
                "stiffness": float(m.jnt_stiffness[j]),
                "springref": float(m.qpos_spring[m.jnt_qposadr[j]]),
                "damping": float(m.dof_damping[m.jnt_dofadr[j]]),
                "armature": float(m.dof_armature[m.jnt_dofadr[j]]),
                "frictionloss": float(m.dof_frictionloss[m.jnt_dofadr[j]]),
                "locked": m.joint(j).name.endswith(LOCKED_SUFFIXES),
            }
            for j in range(m.njnt)
        ],
        "actuators": [
            {
                "name": m.actuator(u).name,
                "joint": int(m.actuator_trnid[u][0]),
                "gain": float(m.actuator_gainprm[u][0]),
                "bias": floats(m.actuator_biasprm[u][:3]),
                "ctrlrange": floats(m.actuator_ctrlrange[u]),
                "forcerange": floats(m.actuator_forcerange[u])
                if m.actuator_forcelimited[u]
                else None,
            }
            for u in range(m.nu)
            if m.actuator_trntype[u] == mujoco.mjtTrn.mjTRN_JOINT
        ],
        "couplings": couplings,
        "geoms": [
            {
                "name": m.geom(g).name,
                "body": int(m.geom_bodyid[g]),
                "type": mujoco.mjtGeom(m.geom_type[g]).name[7:].lower(),
                "size": floats(m.geom_size[g]),
                "pos": floats(m.geom_pos[g]),
                "quat": floats(m.geom_quat[g]),
                "rgba": floats(m.geom_rgba[g]),
                "material": int(m.geom_matid[g]),
                "group": int(m.geom_group[g]),
                "contype": int(m.geom_contype[g]),
                "conaffinity": int(m.geom_conaffinity[g]),
                "friction": float(m.geom_friction[g][0]),
                "priority": int(m.geom_priority[g]),
                "mesh": int(m.geom_dataid[g])
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                else -1,
            }
            for g in range(m.ngeom)
        ],
        "materials": [
            {
                "name": m.material(k).name,
                "rgba": floats(m.mat_rgba[k]),
                "specular": float(m.mat_specular[k]),
                "shininess": float(m.mat_shininess[k]),
                "reflectance": float(m.mat_reflectance[k]),
                "emission": float(m.mat_emission[k]),
                # The rgb role (1) where MuJoCo has texture roles, else the one.
                "texture": int(
                    texid[k][1] if nrole > 1 and texid[k][1] >= 0 else texid[k][0]
                ),
                "texrepeat": floats(m.mat_texrepeat[k]),
            }
            for k in range(m.nmat)
        ],
        "textures": [
            {
                "name": m.texture(t).name,
                "type": mujoco.mjtTexture(m.tex_type[t]).name[11:].lower(),
                "width": int(m.tex_width[t]),
                "height": int(m.tex_height[t]),
            }
            for t in range(m.ntex)
        ],
        "lights": [
            {
                "name": m.light(i).name,
                "directional": bool(
                    light_type[i] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
                    if light_type is not None
                    else m.light_directional[i]
                ),
                "castshadow": bool(m.light_castshadow[i]),
                "xpos": floats(d.light_xpos[i]),
                "xdir": floats(d.light_xdir[i]),
                "diffuse": floats(m.light_diffuse[i]),
                "cutoff": float(m.light_cutoff[i]),
                "active": bool(m.light_active[i]),
            }
            for i in range(m.nlight)
        ],
        "headlight": {
            "ambient": floats(m.vis.headlight.ambient),
            "diffuse": floats(m.vis.headlight.diffuse),
        },
        "cameras": [
            {
                "name": m.camera(c).name,
                "body": int(m.cam_bodyid[c]),
                "pos": floats(m.cam_pos[c]),
                "quat": floats(m.cam_quat[c]),
                "fovy": float(m.cam_fovy[c]),
                "xpos": floats(d.cam_xpos[c]),
                "xmat": floats(d.cam_xmat[c]),
            }
            for c in range(m.ncam)
        ],
        "sites": [
            {
                "name": m.site(s).name,
                "body": int(m.site_bodyid[s]),
                "pos": floats(m.site_pos[s]),
                "quat": floats(m.site_quat[s]),
            }
            for s in range(m.nsite)
        ],
        "excludes": [[int(s >> 16), int(s & 0xFFFF)] for s in m.exclude_signature],
        "test": test_posture(m, couplings),
    }


def write_textures(m, folder):
    folder.mkdir()
    files = {}
    for t in range(m.ntex):
        if m.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_2D:
            continue
        w, h, c = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
        a = int(m.tex_adr[t])
        rgb = m.tex_data[a : a + w * h * c].reshape(h, w, c)[::-1]
        name = "".join(ch if ch.isalnum() else "_" for ch in m.texture(t).name)
        files[t] = f"textures/{t:03d}_{name}.png"
        Image.fromarray(np.ascontiguousarray(rgb)).save(folder.parent / files[t])
    return files


def write_model(m, d, folder, task):
    """model.json and textures/ in folder for model m at state d (after
    mj_forward), with the task's entry; returns the model and the mesh arrays
    (isaac/episode.py hands both to stage.Builder without arrays.npz)."""
    model = describe(m, d)
    files = write_textures(m, folder / "textures")
    for t, entry in enumerate(model["textures"]):
        entry["file"] = files.get(t)
    model["task"] = task
    model["robot"] = {
        "arm_joints": list(ARM_JOINTS),
        "policy_cameras": list(POLICY_CAMERAS),
        "head_pitch": HEAD_PITCH,
        "head_ramp_s": HEAD_RAMP_S,
        "fingers": {s: list(FINGER_JOINTS[s]) for s in ("left", "right")},
        # sim/episode.py: the two fingers get equal and opposite targets of
        # 0.0475 g, the first (it closes with a positive slide) the positive one.
        "finger_travel": 0.0475,
    }
    model["mujoco"] = mujoco.__version__
    (folder / "model.json").write_text(json.dumps(model, indent=1))
    return model, {k: getattr(m, k) for k in MESH_ARRAYS}


def export(name, output):
    task = importlib.import_module(f"scenes.{name}.task")
    output.mkdir(parents=True, exist_ok=False)

    class TaskScene(Scene):
        def build_root(self):
            return task.build()

    scene = TaskScene(output)
    m, d = scene.m, scene.d
    mujoco.mj_forward(m, d)
    entry = {
        "scene": task.NAME,
        "instruction": scene.setup["task"]["instruction"],
        "key_points": getattr(task, "KEY_POINTS", []),
    }
    _, arrays = write_model(m, d, output, entry)
    np.savez(output / "arrays.npz", **arrays)
    files = [
        t for t in range(m.ntex) if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_2D
    ]
    reference = bench_check.check(task, output / "mujoco")
    print(
        json.dumps(
            {
                "scene": name,
                "bodies": m.nbody,
                "geoms": m.ngeom,
                "textures": len(files),
                "mujoco_settle_error": reference["settle_error"],
                "mujoco_max_drift_mm": max(
                    (reference.get("free_drift_mm") or {"-": 0}).values()
                ),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export scenes for the Isaac Sim port."
    )
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in args.scenes:
        export(name, args.output / name)


if __name__ == "__main__":
    main()
