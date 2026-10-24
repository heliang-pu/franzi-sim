"""Four body-mounted cameras derived from the URDF sensor sites.

The wrist D405s sit on the tilted wrist block (sim/wrist_mount.py): each is its
URDF site carried onto the block, i.e. the site's frame times a fixed bracket.
No real-device intrinsics were supplied. FOV values are editable simulation
defaults, not a claim of calibrated D435/D405 RGB or depth imaging.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim import wrist_mount
from sim.build_task import fmt

POLICY_CAMERAS = ("head_d435", "body_d435", "left_wrist_d405", "right_wrist_d405")


def install_cameras(root):
    config = json.loads(Path(__file__).with_name("camera_rig.json").read_text())
    result = {}
    for name in POLICY_CAMERAS:
        profile = config["cameras"][name]
        for body in root.findall(".//body"):
            site = body.find(f"site[@name='{name}']")
            if site is not None:
                break
        else:
            raise ValueError(f"No URDF sensor site: {name}")
        q = np.fromstring(site.get("quat", "1 0 0 0"), sep=" ")
        r = Rotation.from_quat(np.r_[q[1:], q[0]])
        pos = site.get("pos")
        mount = {"mount_source": "URDF fixed sensor joint"}
        if profile.get("mount") == "wrist_block":
            at = np.fromstring(pos, sep=" ")
            on_block, turned = wrist_mount.sensor_pose(
                name.split("_")[0], at, r.as_matrix()
            )
            sensor = Rotation.from_matrix(turned)
            x, y, z, w = (r.inv() * sensor).as_quat()
            mount = {
                "mount_source": profile["mount_source"],
                "bracket_in_site": {
                    "pos_m": r.inv().apply(on_block - at).tolist(),
                    "quat_wxyz": [w, x, y, z],
                },
            }
            pos, r = fmt(on_block), sensor
        optical = r * Rotation.from_euler(
            "z", profile["optical_roll_deg"], degrees=True
        )
        camera = optical * Rotation.from_euler("x", 180, degrees=True)
        xy = camera.as_matrix()[:, :2].T.reshape(-1)
        ET.SubElement(
            body,
            "camera",
            name=name,
            pos=pos,
            xyaxes=fmt(xy),
            fovy=str(profile["fovy_deg"]),
        )
        result[name] = dict(
            profile,
            parent_body=body.get("name"),
            position_in_parent_m=np.fromstring(pos, sep=" ").tolist(),
            camera_axes_in_parent=camera.as_matrix().tolist(),
            **mount,
            width=640,
            height=480,
            calibration_status=config["calibration_status"],
            K_simulation=pinhole_K(profile["fovy_deg"]).tolist(),
        )
    return result


def pinhole_K(fovy, width=640, height=480):
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def assert_mounts(model, data, metadata):
    """Check actual camera origins and view directions after forward kinematics:
    each camera sits on its URDF sensor site, or on the site times its bracket
    (metadata "bracket_in_site", the wrist block)."""
    for name in POLICY_CAMERAS:
        site = model.site(name).id
        cam = model.camera(name).id
        frame = data.site_xmat[site].reshape(3, 3)
        offset, turn = np.zeros(3), np.eye(3)
        bracket = metadata[name].get("bracket_in_site")
        if bracket:
            offset = np.array(bracket["pos_m"])
            w, x, y, z = bracket["quat_wxyz"]
            turn = Rotation.from_quat([x, y, z, w]).as_matrix()
        origin = data.site_xpos[site] + frame @ offset
        assert np.linalg.norm(origin - data.cam_xpos[cam]) < 1e-7, name
        sensor_forward = (frame @ turn)[:, 2]
        camera_forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
        assert np.dot(sensor_forward, camera_forward) > 1 - 1e-6, name
        assert model.cam_bodyid[cam] == model.site_bodyid[site], name
    return True
