"""The gripper's CAD linkage, so the jaws move as on the real robot.

The source model freezes the gripper's links and jaw blocks, open, inside the
wrist_roll_Link mesh and slides only the finger pads along x, so closed pads
hung in the air. In the wrist CAD each jaw block hangs on a parallelogram
(link 5 and the spring strut, 55 mm, about ground pivots C and B): it stays
parallel but swings on an arc, so the pads drop up to 16.3 mm as they close.
A slider driven by the gripper motor turns link 8 about C through link 14, and
link 12 carries that to the jaw block.

Modelled here: the finger x-slide stays the actuated joint (its value is still
the jaw travel, so the closure and the SDK gripper map are unchanged). Each
finger body gets a z-slide, the jaw block (with its pins and red screws) rides
in the finger body, and every link becomes a body on its own joint. Each of
these joints follows its finger's x-slide through a quartic joint equality
fitted to the exact linkage solution (solve()). Their gravity is compensated
like the servos' (the real lead screw does not back-drive).

Coordinates: wrist_roll_Link frame, metres, the jaw-closing plane x-z. The
pivots are the left jaw's (finger1); finger2 mirrors them about x = XC. Both
wrist meshes put the linkage at the same x-z (they mirror each other in y).
"""

import xml.etree.ElementTree as ET

import numpy as np

XC = -0.03045  # mirror plane of the two jaws, midway between the finger bodies
STROKE = 0.0475  # finger travel from open to closed
JAW_SPAN = 0.095  # pad gap at closure 0
# Hole centres fitted on the left wrist CAD (mm / 1000).
C = np.array([-0.04635, -0.17986])  # ground: link 5 and link 8
B = np.array([-0.06702, -0.17234])  # ground: spring strut (parallel to link 5)
J54 = np.array([-0.08545, -0.21854])  # jaw block: link 5
J53 = np.array([-0.09673, -0.21444])  # jaw block: link 12
E = np.array([-0.07572, -0.15735])  # link 8, link 12 and link 14
A = np.array([-0.05176, -0.12823])  # link 14 on the slider, which moves along z
L5, L8, L12, L14 = (
    np.linalg.norm(p - q) for p, q in ((J54, C), (E, C), (J53, E), (A, E))
)
THETA0 = np.arctan2(*(J54 - C)[::-1])
# CAD components that move, picked by bounding box (centre x and z, and size,
# metres) in the left jaw's half; finger2's are their mirror about XC. "jaw"
# rides in the finger body; the others get bodies named "{side}_gripper_{key}".
PARTS = {
    "jaw": [
        ((-0.09578, -0.21953), (0.0317, 0.0175, 0.028)),  # jaw block
        ((-0.09673, -0.21444), (0.0075, 0.029, 0.0075)),  # pin of link 12
        ((-0.08545, -0.21854), (0.0075, 0.029, 0.0075)),  # pin of link 5
        ((-0.0884, -0.229), (0.008, 0.0055, 0.0055)),  # two red screws
    ],
    "link5": [((-0.06562, -0.19895), (0.0505, 0.051, 0.0502))],
    "strut": [
        ((-0.08683, -0.19193), (0.0496, 0.018, 0.0492)),
        ((-0.08533, -0.19045), (0.0323, 0.009, 0.0321)),  # its spring rod
    ],
    "link8": [((-0.06101, -0.16859), (0.0413, 0.0397, 0.0345))],
    "link12": [((-0.086, -0.18566), (0.0324, 0.0286, 0.0685))],
    "link14": [((-0.06376, -0.1428), (0.0359, 0.04, 0.0411))],
}
SLIDER = ((-0.03041, -0.12823), (0.0546, 0.015, 0.012))  # one for both jaws
# Bodies: (parent, pivot in the wrist frame, joint axis for finger1). Hinge
# coordinates are counterclockwise in x-z (x right, z up), so finger1's axis is
# -y; the mirrored finger2 uses +y with the same coordinate.
BODIES = {
    "link5": (None, C),
    "strut": (None, B),
    "link8": (None, C),
    "link12": ("link8", E),
    "link14": ("link8", E),
}
FIT_RANGE = (-0.002, STROKE + 0.002)  # slide values the couplings must cover
SOLREF = "0.004 1"  # stiff couplings: twice the 2 ms physics step
LINK_MASS = 0.001  # kg; the links' real mass stays in the wrist's inertial


def angle(v):
    return np.arctan2(v[1], v[0])


def turn(a, b):
    """Counterclockwise rotation from direction b to direction a, in (-pi, pi]."""
    return (angle(a) - angle(b) + np.pi) % (2 * np.pi) - np.pi


def meet(p, rp, q, rq, near):
    """The intersection of circles (p, rp) and (q, rq) nearer to near."""
    d = np.linalg.norm(q - p)
    a = (rp**2 - rq**2 + d**2) / (2 * d)
    h = np.sqrt(max(rp**2 - a**2, 0.0))
    m = p + a * (q - p) / d
    perp = np.array([p[1] - q[1], q[0] - p[0]]) / d
    return min((m + h * perp, m - h * perp), key=lambda x: np.linalg.norm(x - near))


def solve(s):
    """Finger1's linkage with the jaw moved s (m) toward the centre along x.

    Returns the joint values that pose it: the jaw's z shift ("drop"), and the
    counterclockwise rotations of link 5 (and the strut, its parallel), link 8,
    and links 12 and 14 relative to link 8, plus the slider's z shift."""
    theta = -np.arccos(np.cos(THETA0) + s / L5)  # below C
    drop = L5 * (np.sin(theta) - np.sin(THETA0))
    j53 = J53 + [s, drop]
    e = meet(C, L8, j53, L12, E)
    a = np.array([A[0], e[1] + np.sqrt(L14**2 - (A[0] - e[0]) ** 2)])
    phi8 = turn(e - C, E - C)
    return {
        "drop": drop,
        "link5": theta - THETA0,
        "strut": theta - THETA0,
        "link8": phi8,
        "link12": turn(j53 - e, J53 - E) - phi8,
        "link14": turn(e - a, E - A) - phi8,
        "slider": a[1] - A[1],
    }


def fit_couplings():
    """Quartic coefficients (constant 0) of each joint as a function of s, and
    the worst fit error over FIT_RANGE (m or rad)."""
    s = np.linspace(*FIT_RANGE, 400)
    values = [solve(x) for x in s]
    powers = np.stack([s**k for k in range(1, 5)], 1)
    fits = {}
    for key in values[0]:
        y = np.array([v[key] for v in values])
        coef = np.linalg.lstsq(powers, y, rcond=None)[0]
        # Element-wise: Accelerate's matmul raises spurious FP warnings here.
        error = np.abs((powers * coef).sum(1) - y).max()
        fits[key] = (np.r_[0.0, coef], float(error))
    return fits


def drop(s):
    """How far (m, positive) the pads sit below their open height at travel s."""
    return -solve(s)["drop"]


def contact_drop(width):
    """Pad drop (m) when the jaws close on a part of width (m)."""
    return drop((JAW_SPAN - width) / 2)


def mirror(x):
    return 2 * XC - x


def group_of(bounds):
    """The moving group ("jaw_1", "link5_2", ..., "slider") a wrist CAD
    component belongs to, from its bounds (2x3, metres), or None for the
    static gripper body."""
    lo, hi = np.asarray(bounds)
    centre, size = (lo + hi) / 2, hi - lo
    for key, boxes in [*PARTS.items(), ("slider", [SLIDER])]:
        for (x, z), box in boxes:
            if np.abs(size - box).max() > 0.0006 or abs(centre[2] - z) > 0.0006:
                continue
            for finger, cx in ((1, x), (2, mirror(x))):
                if abs(centre[0] - cx) < 0.0006:
                    return "slider" if key == "slider" else f"{key}_{finger}"
    return None


def body_name(side, group):
    """The body a moving group rides in: jaws in their finger body."""
    if group.startswith("jaw_"):
        return f"{side}finger{group[-1]}_Link"
    return f"{side}_gripper_{group}"


def origin(key, finger):
    """Origin (x, 0, z) in the wrist frame of a link or slider body."""
    if key == "slider":
        return np.array([SLIDER[0][0], 0.0, SLIDER[0][1]])
    pivot = BODIES[key][1]
    x = pivot[0] if finger == 1 else mirror(pivot[0])
    return np.array([x, 0.0, pivot[1]])


def offset_in_wrist(root, side, name):
    """Origin of body name in its wrist_roll_Link frame, from the scene tree at
    closure 0 (no body below the wrist is rotated)."""
    wrist = root.find(f".//body[@name='{side}_wrist_roll_Link']")
    parent = {child: body for body in wrist.iter("body") for child in body}
    body = wrist.find(f".//body[@name='{name}']")
    total = np.zeros(3)
    while body is not wrist:
        assert body.get("quat") is None, name
        total += np.array(body.get("pos", "0 0 0").split(), float)
        body = parent[body]
    return total


def dependent_joints(side):
    """The joints the couplings drive, for gravity compensation."""
    names = [f"{side}_gripper_slider_joint"]
    for finger in (1, 2):
        names.append(f"{side}finger{finger}_drop_joint")
        names += [f"{side}_gripper_{key}_{finger}_joint" for key in BODIES]
    return names


def set_closure(m, d, side, closure):
    """Pose a gripper at closure (0 open, 1 closed) by writing the qpos of its
    finger slides and of every coupled joint, exactly. For static renders only:
    an episode drives the fingers through their actuators."""
    s = STROKE * closure
    exact = solve(s)
    for finger, sign in ((1, 1), (2, -1)):
        d.qpos[m.joint(f"{side}finger{finger}_joint").qposadr[0]] = sign * s
        d.qpos[m.joint(f"{side}finger{finger}_drop_joint").qposadr[0]] = exact["drop"]
        for key in BODIES:
            joint = m.joint(f"{side}_gripper_{key}_{finger}_joint")
            d.qpos[joint.qposadr[0]] = exact[key]
    d.qpos[m.joint(f"{side}_gripper_slider_joint").qposadr[0]] = exact["slider"]


def coupling(coef, finger):
    """Polynomial in the finger's own slide value: finger2's slide is -s."""
    sign = 1 if finger == 1 else -1
    return [c * sign**k for k, c in enumerate(coef)]


def add_linkage(root, fmt):
    """Add the moving linkage bodies, joints and couplings to the scene tree."""
    fits = fit_couplings()
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    def inertial(body):
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=fmt(LINK_MASS),
            diaginertia=fmt([1e-8] * 3),
        )

    def couple(joint, key, finger, slide):
        ET.SubElement(
            equality,
            "joint",
            name=joint + "_coupling",
            joint1=joint,
            joint2=slide,
            polycoef=fmt(coupling(fits[key][0], finger)),
            solref=SOLREF,
        )

    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_roll_Link']")
        slider = ET.SubElement(
            wrist,
            "body",
            name=body_name(side, "slider"),
            pos=fmt(origin("slider", None)),
        )
        inertial(slider)
        joint = f"{side}_gripper_slider_joint"
        ET.SubElement(
            slider,
            "joint",
            name=joint,
            type="slide",
            axis="0 0 1",
            limited="false",
            damping="0",
            armature="0.0001",
        )
        couple(joint, "slider", 1, f"{side}finger1_joint")
        for finger in (1, 2):
            slide = f"{side}finger{finger}_joint"
            body = wrist.find(f"body[@name='{side}finger{finger}_Link']")
            joint = f"{side}finger{finger}_drop_joint"
            ET.SubElement(
                body,
                "joint",
                name=joint,
                type="slide",
                axis="0 0 1",
                limited="false",
                damping="0",
                armature="0",
            )
            couple(joint, "drop", finger, slide)
            made = {}
            for key, (parent, _) in BODIES.items():
                here = origin(key, finger)
                outer = wrist if parent is None else made[parent]
                base = np.zeros(3) if parent is None else origin(parent, finger)
                name = body_name(side, f"{key}_{finger}")
                made[key] = ET.SubElement(
                    outer, "body", name=name, pos=fmt(here - base)
                )
                inertial(made[key])
                joint = name + "_joint"
                ET.SubElement(
                    made[key],
                    "joint",
                    name=joint,
                    type="hinge",
                    axis="0 -1 0" if finger == 1 else "0 1 0",
                    limited="false",
                    damping="0",
                    armature="1e-05",
                )
                couple(joint, key, finger, slide)
    return {
        "model": "CAD parallelogram linkage; finger x-slides actuated, the rest coupled",
        "pivots_m": {
            k: v.tolist() for k, v in (("C", C), ("B", B), ("E", E), ("A", A))
        },
        "link_lengths_m": [float(L5), float(L8), float(L12), float(L14)],
        "max_pad_drop_m": float(max(drop(s) for s in np.linspace(0, STROKE, 200))),
        "coupling_fit_error": {k: v[1] for k, v in fits.items()},
    }
