"""Kinematic cross-check of controller MDH tables against the source URDF.

For every sign assignment q_sdk = s * q_urdf + c, a least-squares fit looks for
offsets c and fixed base/flange transforms that make the controller's MDH chain
reproduce the URDF chain exactly. FK alone leaves a discrete family of exact
equivalents and cannot see the zero offsets of the first and last joint, so the
archive joint ranges then select among the exact solutions. Nothing here reads
a live controller: the result is evidence about archive files, not hardware.

Writes config/robot/kinematic_crosscheck.json.
"""

import argparse
import hashlib
import itertools
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from simbridge import ROOT
from simbridge.robot import urdf_limits

URDF = ROOT / "assets/franzi_source/wheel_robot_4.0.urdf"
IMAGE = ROOT / "sdk_review/controller_image_probe_v2/files/image"
PERSISTENT = IMAGE / "inocfg/inbc_for_humanoid"
APP_CONFIG = IMAGE / "inodata/config11"
ARM = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
]
WAIST = [
    "calf_pitch_joint",
    "thigh_pitch_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
]
EXACT_M = 1e-6
RANGE_TOLERANCE_DEG = 15.0


# ----- NumPy chain math (offline only; the runtime FK/IK is robot_kinematics) -----
X_AXIS = np.array([1.0, 0, 0])
Z_AXIS = np.array([0, 0, 1.0])


def axis_rotation(axis, angles):
    """Batched Rodrigues rotation: (N,) angles about a unit axis -> (N, 3, 3)."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    angles = np.atleast_1d(np.asarray(angles, float))
    c, s = np.cos(angles), np.sin(angles)
    t = 1 - c
    return np.stack(
        [
            np.stack([c + x * x * t, x * y * t - z * s, x * z * t + y * s], -1),
            np.stack([y * x * t + z * s, c + y * y * t, y * z * t - x * s], -1),
            np.stack([z * x * t - y * s, z * y * t + x * s, c + z * z * t], -1),
        ],
        -2,
    )


def rpy_matrix(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    return (
        axis_rotation(Z_AXIS, y)[0]
        @ axis_rotation([0, 1.0, 0], p)[0]
        @ axis_rotation(X_AXIS, r)[0]
    )


def rotvec(R):
    """Rotation vectors of (N, 3, 3) matrices (angle in [0, pi])."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    angle = np.arccos(cos)
    w = np.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1
    )
    out = np.zeros_like(w)
    small = angle < 1e-6
    out[small] = w[small] / 2
    near_pi = angle > np.pi - 1e-4
    regular = ~small & ~near_pi
    out[regular] = w[regular] * (angle[regular] / (2 * np.sin(angle[regular])))[:, None]
    for i in np.flatnonzero(near_pi):
        # Axis from the dominant column of (R + I) / 2 = n n^T.
        B = (R[i] + np.eye(3)) / 2
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / np.sqrt(B[k, k])
        out[i] = n * angle[i]
    return out


def urdf_chain(urdf_path, joint_names):
    """Serial chain [(origin_xyz, origin_R, axis)] for consecutive URDF joints."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    chain = []
    for name in joint_names:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        chain.append((xyz, rpy_matrix(rpy), axis / np.linalg.norm(axis)))
    return chain


def fk_urdf(chain, Q):
    """Poses of the last child link in the first parent's frame, batched over Q."""
    Q = np.atleast_2d(Q)
    R = np.tile(np.eye(3), (len(Q), 1, 1))
    p = np.zeros((len(Q), 3))
    for (xyz, R0, axis), q in zip(chain, Q.T):
        p = p + R @ xyz
        R = R @ R0 @ axis_rotation(axis, q)
    return R, p


def mdh_table(rows):
    """Controller rows {a mm, alpha deg, d mm, theta deg} -> SI (a, alpha, d, theta)."""
    return [
        (r["a"] / 1000, np.deg2rad(r["alpha"]), r["d"] / 1000, np.deg2rad(r["theta"]))
        for r in rows
    ]


def machine_params_mdh(path, unit):
    data = json.loads(path.read_text())
    cfg = next(m for m in data["mechunit_cfg"] if m["name"] == unit)
    return mdh_table(cfg["bio_para"])


def fk_mdh(table, Q):
    """Craig modified DH: T_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(theta_i + q_i)
    TransZ(d_i). Each row holds (a_{i-1}, alpha_{i-1}, d_i, theta_i)."""
    Q = np.atleast_2d(Q)
    n = len(Q)
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    for (a, alpha, d, theta), q in zip(table, Q.T):
        p = p + R @ np.array([a, 0, 0])
        R = (
            R
            @ axis_rotation(X_AXIS, np.full(n, alpha))
            @ axis_rotation(Z_AXIS, theta + q)
        )
        p = p + R @ np.array([0, 0, d])
    return R, p


def fk_standard_dh(table, Q):
    """Standard DH with the same rows, used only to show it does not fit."""
    Q = np.atleast_2d(Q)
    n = len(Q)
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    for (a, alpha, d, theta), q in zip(table, Q.T):
        R = R @ axis_rotation(Z_AXIS, theta + q)
        p = p + R @ np.array([0, 0, d]) + R @ np.array([a, 0, 0])
        R = R @ axis_rotation(X_AXIS, np.full(n, alpha))
    return R, p


def compose(R_base, p_base, R, p, R_tool, p_tool):
    """Base * chain * tool for batched chain poses."""
    R_out = R_base @ R @ R_tool
    p_out = p_base + (p + R @ p_tool) @ R_base.T
    return R_out, p_out


def pose_from_vector(x):
    """6-vector (rotation vector, translation) -> (R, p)."""
    angle = np.linalg.norm(x[:3])
    R = np.eye(3) if angle < 1e-12 else axis_rotation(x[:3] / angle, angle)[0]
    return R, np.asarray(x[3:6], float)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual(x, table, signs, free, Q, R_urdf, p_urdf, fk):
    offsets = np.zeros(len(signs))
    offsets[free] = x[: len(free)]
    k = len(free)
    R_b, p_b = pose_from_vector(x[k : k + 6])
    R_t, p_t = pose_from_vector(x[k + 6 : k + 12])
    R, p = fk(table, signs * Q + offsets)
    R, p = compose(R_b, p_b, R, p, R_t, p_t)
    rot = rotvec(np.transpose(R, (0, 2, 1)) @ R_urdf)
    # Rotation error weighted by a 0.1 m lever arm, so both terms are metres.
    return np.concatenate([(p - p_urdf).ravel(), (0.1 * rot).ravel()])


def search(chain, table, fk, n, attempts=4, seed=0):
    """All sign vectors; a few offset seeds each. Returns exact fits found."""
    rng = np.random.default_rng(seed)
    free = list(range(1, n - 1))
    Q = rng.uniform(-1.2, 1.2, size=(10, n))
    R_urdf, p_urdf = fk_urdf(chain, Q)
    results = []
    for signs in itertools.product([1.0, -1.0], repeat=n):
        signs = np.array(signs)
        best = None
        for attempt in range(attempts):
            x0 = np.zeros(len(free) + 12)
            if attempt:
                x0[: len(free)] = rng.choice(
                    [0, np.pi / 2, np.pi, -np.pi / 2], len(free)
                )
                x0[len(free) : len(free) + 3] = rng.normal(size=3)
            fit = least_squares(
                residual,
                x0,
                args=(table, signs, free, Q, R_urdf, p_urdf, fk),
                max_nfev=400,
            )
            rms = float(np.sqrt(np.mean(fit.fun**2)))
            if best is None or rms < best[0]:
                best = (rms, fit.x)
        offsets = np.zeros(n)
        offsets[free] = best[1][: len(free)]
        offsets = (offsets + np.pi) % (2 * np.pi) - np.pi
        offsets[np.isclose(offsets, -np.pi)] = np.pi
        results.append(
            {
                "signs": signs.astype(int).tolist(),
                "offsets_rad": offsets.tolist(),
                "rms_m": best[0],
                "x": best[1],
            }
        )
    return results


def validate(chain, table, fk, signs, offsets, x, limits, count=500):
    """Fresh random joint states inside the URDF limits, never seen by the fit."""
    n = len(signs)
    k = n - 2
    Q = np.random.default_rng(42).uniform(limits[:, 0], limits[:, 1], size=(count, n))
    R_u, p_u = fk_urdf(chain, Q)
    R_b, p_b = pose_from_vector(x[k : k + 6])
    R_t, p_t = pose_from_vector(x[k + 6 : k + 12])
    R, p = fk(table, np.asarray(signs) * Q + np.asarray(offsets))
    R, p = compose(R_b, p_b, R, p, R_t, p_t)
    rot = rotvec(np.transpose(R, (0, 2, 1)) @ R_u)
    return {
        "samples": count,
        "max_position_error_m": float(np.max(np.linalg.norm(p - p_u, axis=1))),
        "max_rotation_error_rad": float(np.max(np.linalg.norm(rot, axis=1))),
    }


def range_check(signs, offsets, sim_limits, controller_limits):
    rows = []
    for s, c, sim, ctrl in zip(signs, offsets, sim_limits, controller_limits):
        mapped = np.sort(np.rad2deg(s * np.asarray(sim) + c))
        diff = np.abs(mapped - np.asarray(ctrl, float))
        rows.append(
            {
                "sim_range_mapped_to_sdk_deg": mapped.round(3).tolist(),
                "controller_range_deg": list(ctrl),
                "endpoint_difference_deg": diff.round(3).tolist(),
                "consistent": bool(np.all(diff <= RANGE_TOLERANCE_DEG)),
            }
        )
    return rows


def transform(x, k):
    R_b, p_b = pose_from_vector(x[k : k + 6])
    R_t, p_t = pose_from_vector(x[k + 6 : k + 12])
    # Flange (MDH last frame) expressed in the URDF child link: inverse of tool.
    return {
        "mdh_base_in_parent": {
            "R": R_b.round(9).tolist(),
            "p_m": p_b.round(9).tolist(),
        },
        "mdh_flange_in_child_link": {
            "R": R_t.T.round(9).tolist(),
            "p_m": (-R_t.T @ p_t).round(9).tolist(),
        },
    }


def analyse(name, chain, table, sim_limits, controller_limits, joints):
    n = len(joints)
    started = time.monotonic()
    found = search(chain, table, fk_mdh, n)
    exact = [r for r in found if r["rms_m"] < EXACT_M]
    for r in exact:
        r["ranges"] = range_check(
            r["signs"], r["offsets_rad"], sim_limits, controller_limits
        )
        r["range_consistent"] = all(x["consistent"] for x in r["ranges"])
        r["range_score_deg"] = max(
            max(x["endpoint_difference_deg"]) for x in r["ranges"]
        )
    exact.sort(key=lambda r: r["range_score_deg"])
    consistent = [r for r in exact if r["range_consistent"]]
    standard = search(chain, table, fk_standard_dh, n, attempts=2)
    report = {
        "chain": joints,
        "mdh_convention_tested": "Craig modified DH, theta_i = q_i + theta_offset_i",
        "sign_assignments_searched": len(found),
        "exact_fk_solutions": len(exact),
        "exact_solutions": [
            {
                "signs": r["signs"],
                "offsets_deg": np.rad2deg(r["offsets_rad"]).round(3).tolist(),
                "rms_m": r["rms_m"],
                "range_consistent": r["range_consistent"],
                "worst_range_endpoint_difference_deg": r["range_score_deg"],
            }
            for r in exact
        ],
        "standard_dh_best_rms_m": min(r["rms_m"] for r in standard),
        "seconds": round(time.monotonic() - started, 1),
    }
    gap = (
        exact[1]["range_score_deg"] - exact[0]["range_score_deg"]
        if len(exact) > 1
        else 0
    )
    if len(consistent) == 1:
        report["status"] = (
            "unique: FK exact up to a discrete family; archive ranges select one"
        )
    elif not consistent and exact and gap > 2 * RANGE_TOLERANCE_DEG:
        report["status"] = (
            "best range match (no exact solution fits every archive range; "
            "see selected.range_conflicts)"
        )
    else:
        report["status"] = f"ambiguous: {len(consistent)} range-consistent exact fits"
    if not report["status"].startswith("ambiguous"):
        r = exact[0]
        report["selected"] = {
            "signs": r["signs"],
            "offsets_rad": np.round(r["offsets_rad"], 12).tolist(),
            "validation": validate(
                chain, table, fk_mdh, r["signs"], r["offsets_rad"], r["x"], sim_limits
            ),
            "ranges": r["ranges"],
            "range_conflicts": [
                joints[i] for i, x in enumerate(r["ranges"]) if not x["consistent"]
            ],
            "runner_up_worst_endpoint_difference_deg": (
                exact[1]["range_score_deg"] if len(exact) > 1 else None
            ),
            "frames": transform(r["x"], n - 2),
            "not_identifiable_by_fk": [
                f"{joints[0]} zero offset (absorbed by the base frame)",
                f"{joints[-1]} zero offset (absorbed by the flange frame)",
            ],
        }
    print(name, report["status"], f"({report['seconds']} s)", flush=True)
    return report


def fixture_check():
    """The mdh_repeat_exact files pair joint and Cartesian values; test them."""
    rows = []
    for path in sorted(APP_CONFIG.glob("mdh_repeat_exact*.json")):
        data = json.loads(path.read_text())
        for when in ("start", "end"):
            for side in ("left", "right"):
                entry = data[when][f"{side}_arm"]
                q = np.array(entry["joint_pos"]["values"])
                cart = np.array(entry["cartesian_pos"]["values"])
                table = machine_params_mdh(
                    PERSISTENT / "machine_params.json", f"{side}_arm"
                )
                reach = {
                    unit: float(np.linalg.norm(fk_mdh(table, values)[1][0]))
                    for unit, values in (("deg", np.deg2rad(q)), ("rad", q))
                }
                rows.append(
                    {
                        "file": path.name,
                        "sample": f"{when}/{side}",
                        "fk_flange_distance_m": reach,
                        "file_cartesian_distance_m": float(np.linalg.norm(cart[:3])),
                    }
                )
    consistent = any(
        abs(d - r["file_cartesian_distance_m"]) < 0.01
        for r in rows
        for d in r["fk_flange_distance_m"].values()
    )
    return {
        "samples": rows,
        "usable_as_fk_ground_truth": consistent,
        "note": "Start and end joint values are almost equal while the Cartesian "
        "values differ by decimetres; the files look like edited fixtures.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=ROOT / "config/robot/kinematic_crosscheck.json"
    )
    args = parser.parse_args()
    machine = PERSISTENT / "machine_params.json"
    sysctrl = json.loads((PERSISTENT / "sysctrl_params.json").read_text())
    app_mdh = json.loads((APP_CONFIG / "mdh.json").read_text())["mdh_config"]
    report = {
        "inputs": {
            "urdf": {"path": str(URDF.relative_to(ROOT)), "sha256": sha256(URDF)},
            "machine_params": {
                "path": str(machine.relative_to(ROOT)),
                "sha256": sha256(machine),
            },
            "sysctrl_params": {
                "path": str((PERSISTENT / "sysctrl_params.json").relative_to(ROOT)),
                "sha256": sha256(PERSISTENT / "sysctrl_params.json"),
            },
            "waist_config": {
                "path": str((PERSISTENT / "waist_config.json").relative_to(ROOT)),
                "sha256": sha256(PERSISTENT / "waist_config.json"),
            },
        },
        "scope": "archive files versus source URDF; no live controller readback",
        "arms": {},
    }
    for unit, side in ((0, "left"), (1, "right")):
        table = machine_params_mdh(machine, f"{side}_arm")
        same = np.allclose(table, mdh_table(app_mdh[f"{side}_arm"]["mdh"]))
        joints = [f"{side}_{j}_joint" for j in ARM]
        limits = sysctrl["mechunit_limit"][unit]["joint_limit"]
        controller = [[j["negative_limit"], j["positive_limit"]] for j in limits]
        result = analyse(
            f"{side} arm",
            urdf_chain(URDF, joints),
            table,
            urdf_limits(URDF, joints),
            controller,
            joints,
        )
        result["persistent_mdh_equals_app_config_mdh"] = bool(same)
        report["arms"][side] = result
    waist = json.loads((PERSISTENT / "waist_config.json").read_text())["waist_config"]
    report["waist"] = analyse(
        "waist",
        urdf_chain(URDF, WAIST),
        mdh_table(waist["mdh"]),
        urdf_limits(URDF, WAIST),
        [[j["position_min"], j["position_max"]] for j in waist["joint_limits"]],
        WAIST,
    )
    report["head"] = {
        "status": "uncalibrated: the archive head_config.json has limits but no MDH",
        "controller_range_deg": [[-180, 180], [-30, 30]],
    }
    report["fixtures"] = fixture_check()
    args.output = type(ROOT)(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
