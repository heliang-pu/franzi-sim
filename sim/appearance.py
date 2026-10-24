"""Paint the source CAD to match the real robot, retaining all physical model data.

Visual meshes are split into connected CAD components (largest first) and, where
one component carries two finishes, into face regions. Each group becomes a
visual-only geom; collision geoms, inertia and the source files are untouched.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from sim import wrist_mount
from sim.fittings import attach_fittings, build_fittings
from sim.gripper_linkage import body_name, group_of, offset_in_wrist

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/appearance/painted"
WORDMARK_DIR = ROOT / "assets/appearance/wordmark"
FITTINGS_DIR = ROOT / "assets/appearance/fittings"
CAMERA_DIR = ROOT / "assets/appearance/cameras"
# Bump when a face rule changes; the cache key only sees the source meshes.
PAINT_VERSION = 10

# Finishes matched to the lab and factory videos of the real robot.
PALETTE = {
    "fa_pearl": ("0.84 0.85 0.87 1", "0.55", "0.7", "0"),
    "fa_silver": ("0.72 0.73 0.75 1", "0.75", "0.85", "0"),
    "fa_graphite": ("0.15 0.16 0.18 1", "0.3", "0.35", "0"),
    "fa_black": ("0.05 0.055 0.06 1", "0.3", "0.4", "0"),
    "fa_gloss_black": ("0.015 0.017 0.02 1", "0.9", "0.95", "0"),
    "fa_navy": ("0.06 0.07 0.095 1", "0.25", "0.3", "0"),
    # Lights get no specular, which would wash their emission towards white.
    "fa_led_green": ("0.1 0.9 0.3 1", "0", "0", "1"),
    "fa_estop_red": ("0.85 0.08 0.06 1", "0.5", "0.6", "0"),
    "fa_led_blue": ("0.45 0.75 1 1", "0", "0", "0.9"),
    # Anodised aluminium of the RealSense D435/D405 housings.
    "fa_camera_grey": ("0.66 0.67 0.69 1", "0.6", "0.7", "0"),
    # RealSense front: Menagerie's D435i colours with glossy glass and lenses.
    "fa_cam_glass": ("0.07 0.07 0.07 1", "0.9", "0.9", "0"),
    "fa_cam_gray": ("0.30 0.30 0.30 1", "0.3", "0.3", "0"),
    "fa_cam_lens": ("0.036 0.036 0.036 1", "0.9", "0.95", "0"),
    "fa_cam_emitter": ("0.29 0.67 0.33 1", "0.6", "0.8", "0"),
    "fa_cam_rim": ("0.80 0.81 0.80 1", "0.7", "0.8", "0"),
    "fa_cam_pupil": ("0.087 0.003 0.009 1", "0.9", "0.95", "0"),
    # Tinted chest-slot window: the D435's lenses show faintly through it.
    "fa_slot_window": ("0.02 0.022 0.026 0.86", "0.9", "0.95", "0"),
}
# Official RealSense meshes (see SOURCE.json beside them) replace the URDF's camera
# boxes. D435: Menagerie's coloured D435i parts, width along +x with the RGB at +x,
# glass at z=0. Into the URDF sensor frame: width to +y with the left IR imager at
# the origin (RGB 15 mm towards +y), glass 4.2 mm ahead of it, height along -x.
D435_PARTS = [
    ("d435i_0.stl", "fa_cam_lens"),
    ("d435i_1.stl", "fa_cam_emitter"),
    ("d435i_2.stl", "fa_cam_rim"),
    ("d435i_3.stl", "fa_cam_lens"),
    ("d435i_4.stl", "fa_cam_gray"),
    ("d435i_5.stl", "fa_cam_glass"),
    ("d435i_6.stl", "fa_cam_glass"),
    ("d435i_7.stl", "fa_cam_pupil"),
    ("d435i_8.stl", "fa_camera_grey"),
]
D435_ALIGN = (
    np.array([0.0, -0.0175, 0.0042]),
    Rotation.from_euler("z", 90, degrees=True),
)
REAL_CAMERAS = {
    "head_d435_Link": "d435",
    "body_d435_Link": "d435",
    "left_wrist_d405_Link": "d405",
    "right_wrist_d405_Link": "d405",
}
# Chest camera slot, in the body D435 sensor frame (x height, y width, z forward).
# The shell's slot is an 85.1 x 20.8 mm stadium around the camera, its walls
# parallel to the optical axis from the glass (z=4.2 mm) to past 6 mm; it shows the
# whole D435 front. The real robots show a narrow dark window in a light bezel, so a
# bezel plate with a tinted slit fills the slot just ahead of the glass. Both sit
# well inside the 45 mm render znear, so the body camera never sees them.
CHEST_SLOT = {
    "centre": (0.0001, -0.0175),  # aperture centre, x and y
    "aperture": (0.0851, 0.0208),  # width and height
    "margin": 0.001,  # the bezel reaches into the slot walls
    # Window width and height: on the lab video it is 82 mm wide, nearly the whole
    # slot, and half its height (10.5 mm in the slot's plane).
    "slit": (0.082, 0.0105),
    "bezel_z": 0.0052,
    "window_z": 0.0046,
    "segments": 16,  # per semicircular end
}
PEARL_SRGB = tuple(round(float(v) * 255) for v in PALETTE["fa_pearl"][0].split()[:3])
BLACK_SRGB = tuple(round(float(v) * 255) for v in PALETTE["fa_black"][0].split()[:3])
INK_SRGB = (52, 55, 62)
WORDMARK = {
    # Torso frame. Measured on the real robot against the 84.5 mm chest slot
    # (centre z=0.232): text 1.17x the slot width. The factory print sits it
    # 1.28 slot widths above the slot (the lab robot's INOVANCE-only print: 1.07).
    "center_z": 0.339,
    "width": 0.099,
    "cap_height": 0.0119,
    "margin": 0.004,
    "offset": 0.0004,
    "font": "/System/Library/Fonts/Avenir Next.ttc",
    "font_index": 2,  # Demi Bold
    "texture_width": 4096,
    "version": 3,
}
# Factory chest print under INOVANCE: "Powered by", the FMC³ letters and a tracked
# ROBOTICS line, measured on the factory video in INOVANCE widths (the I of
# INOVANCE is at y=-0.0495, its E at 0.0495). Text centres are ink centroids.
FACTORY_PRINT = {
    "logo": "assets/fmc3_logo.png",  # official FMC³ logo; supplies the small 3
    "fmc3_left": -0.0515,  # the F
    "letter_height": 0.0186,
    "baseline_z": 0.2715,
    "powered_by_width": 0.038,
    "powered_by_cap_height": 0.0043,  # tracked out to the width
    "powered_by_center_z": 0.312,
    "powered_by_font_index": 7,  # Regular
    "robotics_cap_height": 0.0035,  # tracked from the F to INOVANCE's E
    "robotics_center_z": 0.262,
    "version": 3,
}
# The printed FMC³ differs from the logo file. In letter heights from the F's left
# edge and the cap line (y down), read off a 30-frame stack of the factory video: a
# bar floating over a Γ, a V-shaped M, a C with two dot eyes and a small 3 standing
# on the cap line.
FMC3_GLYPHS = {
    "stroke": 0.22,
    "bar": (0.0, -0.43, 1.24, -0.24),
    "arm_right": 1.24,
    "m": (1.54, 3.11),
    "c": (3.35, 4.89),
    "eyes": (3.95, 4.62),
    "eye_diameter": 0.19,
    "three": (5.08, 0.27),  # left edge and height
}
# Elbow-cap logo in the 72 mm outer cap, sized from the lab photos.
ELBOW_LOGO = {
    "width": 0.048,
    "cap_height": 0.0058,
    "cap_radius": 0.0362,  # black cap, inside the CAD's 35.9 mm vertex ring
    "patch_radius": 0.044,  # decal reaches past the cap onto the pearl housing
    "offset": 0.0003,
    "ink": (236, 237, 239),
    "texture_width": 2048,
    "version": 1,
}


def thigh_crossbar(part, side):
    # The H-frame crossbar between the white rails is dark on the real robot.
    return np.abs(part.triangles_center[:, 1]) < 0.069


def grille_pad(part, side):
    # The textured upper-arm pad is the densely tessellated patch on the front;
    # only faces wholly inside it are taken, so the edge follows the CAD groove.
    c, v = part.triangles_center, part.vertices
    zbins = np.arange(-0.19, 0.015, 0.004)
    tbins = np.arange(-180, 181, 6)
    theta = np.degrees(np.arctan2(c[:, 1], c[:, 0] - 0.008))
    count, _, _ = np.histogram2d(c[:, 2], theta, bins=[zbins, tbins])
    dense = ndimage.binary_closing(count > np.quantile(count[count > 0], 0.8))
    labels, n = ndimage.label(dense)
    blob = labels == 1 + np.argmax(ndimage.sum(dense, labels, range(1, n + 1)))
    blob = ndimage.binary_dilation(ndimage.binary_fill_holes(blob))
    theta = np.degrees(np.arctan2(v[:, 1], v[:, 0] - 0.008))
    iz = np.clip(np.digitize(v[:, 2], zbins) - 1, 0, len(zbins) - 2)
    it = np.clip(np.digitize(theta, tbins) - 1, 0, len(tbins) - 2)
    inside = blob[iz, it] & (np.hypot(v[:, 0] - 0.008, v[:, 1]) > 0.026)
    return inside[part.faces].all(axis=1)


def ring_cap(part, axis, signs, radius, depth):
    # Faces wholly inside a vertex ring of a joint cap give a clean circular edge.
    # Only faces connected to the cap centre count: housing flanks that merely
    # fall inside the ring's cylinder stay unpainted.
    v = part.vertices
    u, w = [i for i in range(3) if i != axis]
    radial = np.hypot(v[:, u], v[:, w])
    keep = np.zeros(len(part.faces), bool)
    for s in signs:
        inside = (radial <= radius) & (s * v[:, axis] > depth)
        cand = inside[part.faces].all(axis=1)
        pairs = part.face_adjacency[cand[part.face_adjacency].all(axis=1)]
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(part.faces),) * 2,
        )
        _, labels = connected_components(graph, directed=False)
        centre = cand & (radial[part.faces].mean(axis=1) < 0.3 * radius)
        keep |= cand & np.isin(labels, labels[centre])
    return keep


def shoulder_caps(part, side):
    # Front and back ears of the roll yoke carry dark caps with a white rim.
    return ring_cap(part, 0, (1, -1), 0.0379, 0.045)


def elbow_inner_cap(part, side):
    # Domed inner elbow cap to its 30.5 mm vertex ring. The outer logo cap is
    # drawn by the elbow decal instead: its triangles cross any face-level cut.
    sign = 1 if side == "left" else -1
    return ring_cap(part, 1, (-sign,), 0.0307, 0.03)


def part_size(size, tol=0.0015):
    # Whole CAD parts picked by bounding-box size, which mirrored arms share.
    size = np.asarray(size)
    return lambda part: np.abs(np.ptp(part.bounds, axis=0) - size).max() < tol


def linkage_group(part, side):
    # Gripper links and jaw blocks ride in their own bodies (gripper_linkage).
    return group_of(part.bounds)


def old_wrist_block(part, side):
    # The original D405 block and its screws, replaced by the tilted block.
    return wrist_mount.is_old_block(part.bounds, side)


def d435_housing(part, side):
    # The head D435 box is merged with its bracket; the box matches body_d435 and
    # is dropped for the official D435 mesh, leaving the bracket.
    v = part.vertices
    inside = np.all(
        (v >= [-0.0126, -0.0626, -0.0209]) & (v <= [0.0126, 0.0276, 0.0043]), 1
    )
    return inside[part.faces].all(axis=1)


# Material per CAD component index (largest first); "rest" covers every other
# component, "parts" repaints whole components chosen by shape and "faces"
# repaints regions of one component. None drops the faces.
RECIPE = {
    "head_pitch_Link": {
        0: "fa_pearl",  # band over the crown
        1: "fa_graphite",  # rear shell
        2: "fa_gloss_black",  # face visor
        3: "fa_pearl",  # neck bracket
    },
    "head_yaw_Link": {"rest": "fa_graphite"},
    "torso_Link": {"rest": "fa_graphite", 2: "fa_pearl", 3: "fa_pearl"},
    "waist_Link": {"rest": "fa_graphite"},
    "thigh_Link": {"rest": "fa_pearl", "faces": [(0, thigh_crossbar, "fa_navy")]},
    "calf_Link": {"rest": "fa_pearl"},
    "base_link": {"rest": "fa_black", 1: "fa_pearl", 3: "fa_pearl", 4: "fa_pearl"},
    "{side}_shoulder_pitch_Link": {"rest": "fa_black", 0: "fa_pearl"},
    "{side}_shoulder_roll_Link": {
        "rest": "fa_black",
        0: "fa_pearl",
        "faces": [(0, shoulder_caps, "fa_black")],
    },
    "{side}_shoulder_yaw_Link": {
        "rest": "fa_pearl",
        "faces": [(0, grille_pad, "fa_black")],
    },
    "{side}_elbow_pitch_Link": {
        "rest": "fa_black",
        0: "fa_pearl",
        "faces": [(0, elbow_inner_cap, "fa_black")],
    },
    "{side}_wrist_yaw_Link": {"rest": "fa_pearl"},
    "{side}_wrist_pitch_Link": {"rest": "fa_black"},
    # Linkage gripper: black body, flange, links and camera bracket; aluminium
    # jaw blocks and pins, red screws on the jaw blocks, aluminium finger pads.
    # "detach" moves the links and jaw blocks into the bodies that move them;
    # "drop" leaves out the original D405 block (attach_wrist_blocks).
    "{side}_wrist_roll_Link": {
        "rest": "fa_black",
        "parts": [
            (part_size([0.0317, 0.0175, 0.028]), "fa_silver"),
            (part_size([0.0075, 0.029, 0.0075]), "fa_silver"),
            (part_size([0.008, 0.0055, 0.0055]), "fa_estop_red"),
        ],
        "detach": linkage_group,
        "drop": old_wrist_block,
    },
    "{side}finger1_Link": {"rest": "fa_silver"},
    "{side}finger2_Link": {"rest": "fa_silver"},
    "head_d435_Link": {"rest": "fa_black", "faces": [(0, d435_housing, None)]},
    "{wheel}_wheel_Link": {"rest": "fa_silver", 0: "fa_black"},
}


def expanded_recipe():
    recipes = {}
    for key, recipe in RECIPE.items():
        if "{side}" in key:
            for side in ("left", "right"):
                recipes[key.format(side=side)] = (recipe, side)
        elif "{wheel}" in key:
            for wheel in ("left_front", "right_front", "rear"):
                recipes[key.format(wheel=wheel)] = (recipe, None)
        else:
            recipes[key] = (recipe, None)
    return recipes


def paint_groups(path, recipe, side):
    """Painted meshes keyed by (detached group or None, material)."""
    mesh = trimesh.load(path, process=True, force="mesh")
    parts = sorted(
        mesh.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True
    )
    groups = {}
    for k, part in enumerate(parts):
        material = recipe.get(k, recipe.get("rest"))
        for match, override in recipe.get("parts", []):
            if match(part):
                material = override
        assert material, f"{path.name}: component {k} has no paint; CAD changed?"
        if recipe.get("drop") and recipe["drop"](part, side):
            continue
        detach = recipe.get("detach")
        group = detach(part, side) if detach else None
        paint = np.full(len(part.faces), material, dtype=object)
        for index, rule, override in recipe.get("faces", []):
            if index == k:
                paint[rule(part, side)] = override
        for name in set(paint) - {None}:
            faces = np.nonzero(paint == name)[0]
            groups.setdefault((group, name), []).append(
                part.submesh([faces], append=True)
            )
    return {key: trimesh.util.concatenate(g) for key, g in groups.items()}


def prepare_paint(source):
    recipes = expanded_recipe()
    key = hashlib.sha256(f"paint-v{PAINT_VERSION}".encode())
    for name in sorted(recipes):
        key.update((source / "meshes" / f"{name}.STL").read_bytes())
    digest = key.hexdigest()
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        files = [e["file"] for groups in old["meshes"].values() for e in groups]
        if old["key"] == digest and all((OUTPUT / f).exists() for f in files):
            return old
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"key": digest, "meshes": {}}
    for name, (recipe, side) in recipes.items():
        groups = paint_groups(source / "meshes" / f"{name}.STL", recipe, side)
        manifest["meshes"][name] = []
        for (part, material), group in sorted(
            groups.items(), key=lambda item: (item[0][0] or "", item[0][1])
        ):
            stem = name if part is None else f"{name}__{part}"
            filename = f"{stem}__{material}.stl"
            group.export(OUTPUT / filename)
            entry = {"file": filename, "material": material, "faces": len(group.faces)}
            manifest["meshes"][name].append(entry | ({"part": part} if part else {}))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def compiled_mesh(mesh, path):
    mesh.export(path)
    return compiled_file(path)


def compiled_file(path):
    """Vertices, faces and per-corner normals exactly as MuJoCo renders a mesh."""
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="s" file="{path}" inertia="shell"/></asset>'
        '<worldbody><geom type="mesh" mesh="s" contype="0" conaffinity="0"/>'
        "</worldbody></mujoco>"
    )
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[0])
    rot = rot.reshape(3, 3)
    # Element-wise products: Accelerate's matmul raises spurious FP warnings here.
    vert = (rot * model.mesh_vert[: model.nmeshvert, None]).sum(-1)
    normal = (rot * model.mesh_normal[: model.nmeshnormal, None]).sum(-1)
    face = model.mesh_face[: model.nmeshface]
    facenormal = model.mesh_facenormal[: model.nmeshface]
    return vert + model.mesh_pos[0], face, facenormal, normal


def tracked_mask(text, width_px, cap_px, index, inverted=()):
    """Tracked capitals whose ink measures exactly width_px by cap_px.

    Glyphs at the positions in inverted are drawn upside down.
    """
    font = ImageFont.truetype(WORDMARK["font"], 400, index=index)
    top, bottom = font.getbbox("I")[1], font.getbbox("I")[3]
    cap = bottom - top
    advances = [font.getlength(c) for c in text]
    left = font.getbbox(text[0])[0]
    right = advances[-1] - font.getbbox(text[-1])[2]
    natural = sum(advances) - left - right
    gaps = len(text) - 1
    track = (width_px * cap / cap_px - natural) / gaps
    canvas = Image.new("L", (round(natural + gaps * track) + 400, cap + 400), 0)
    draw = ImageDraw.Draw(canvas)
    x = 200 - left
    for i, (c, advance) in enumerate(zip(text, advances)):
        if i in inverted:
            l, t, r, b = font.getbbox(c)
            glyph = Image.new("L", (r + 10, b + 10), 0)
            ImageDraw.Draw(glyph).text((0, 0), c, font=font, fill=255)
            glyph = glyph.crop((l, t, r, b)).rotate(180)
            canvas.paste(glyph, (round(x + advance - r), 200), glyph)
        else:
            draw.text((x, 200 - top), c, font=font, fill=255)
        x += advance + track
    canvas = canvas.crop((200, 200, round(x - track - right), 200 + cap))
    return canvas.resize((round(width_px), round(cap_px)), Image.LANCZOS)


def wordmark_mask(width_px, cap_px):
    # The INOVANCE logo draws its A without a crossbar, so A is an inverted V.
    return tracked_mask("INOVVNCE", width_px, cap_px, WORDMARK["font_index"], (4,))


def text_mask(text, width_px, index, cap_px=None):
    """Text cropped to its ink, width_px wide: at its own spacing, or with capitals
    cap_px tall and the spacing tracked out to the width. Returns the mask and its
    ink centroid row."""
    font = ImageFont.truetype(WORDMARK["font"], 400, index=index)
    cap = font.getbbox("I")[3] - font.getbbox("I")[1]
    l, t, r, b = font.getbbox(text)
    track = 0.0
    if cap_px is not None:
        track = (width_px * cap / cap_px - (r - l)) / (len(text) - 1)
    canvas = Image.new("L", (round(r - l + track * len(text)) + 40, b - t + 40), 0)
    draw = ImageDraw.Draw(canvas)
    x = 20 - l
    for ch in text:
        draw.text((x, 20 - t), ch, font=font, fill=255)
        x += font.getlength(ch) + track
    canvas = canvas.crop(canvas.getbbox())
    height = round(canvas.height * width_px / canvas.width)
    mask = canvas.resize((round(width_px), height), Image.LANCZOS)
    ink = np.asarray(mask, float).sum(1)
    return mask, (ink * np.arange(len(ink))).sum() / ink.sum()


def fmc3_mask(path, letter_px):
    """FMC³ as printed on the factory robot (FMC3_GLYPHS), letters letter_px tall.

    Drawn at 4x and filtered down; the small 3 is the logo file's. Returns the mask
    and the F's left edge and the baseline row in it.
    """
    g, k = FMC3_GLYPHS, 4
    u, s = letter_px * k, FMC3_GLYPHS["stroke"]  # u: one letter height at 4x
    logo = Image.open(path).convert("L").transpose(Image.Transpose.ROTATE_270)
    a = np.asarray(logo, float)
    ink = np.uint8(np.clip((np.median(a) - a) / (np.median(a) - a.min()), 0, 1) * 255)
    rows = np.nonzero(ink.max(1) > 127)[0]
    band = rows[: np.argmax(np.diff(rows) > 1) + 1]  # letters; ROBOTICS is below
    cols = np.nonzero(ink[band].max(0) > 127)[0]
    three = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)[-1]
    r = np.nonzero(ink[:, three[0] : three[-1] + 1].max(1) > 127)[0]
    glyph = Image.fromarray(ink).crop((three[0], r[0], three[-1] + 1, r[-1] + 1))
    left, height = g["three"]
    size = (round(glyph.width * height * u / glyph.height), round(height * u))
    glyph = glyph.resize(size, Image.LANCZOS)
    pad = 0.1 * u
    x0, cap = pad, pad - g["bar"][1] * u  # the F's left edge, the cap line
    big = Image.new(
        "L", (round(left * u + glyph.width + 2 * pad), round(cap + u + pad))
    )
    draw = ImageDraw.Draw(big)

    def at(x, y):
        return x0 + x * u, cap + y * u

    def bar(xa, ya, xb, yb):
        r = min(xb - xa, yb - ya) * u / 2
        draw.rounded_rectangle([*at(xa, ya), *at(xb, yb)], radius=r, fill=255)

    def dot(x, y, d):
        draw.ellipse([*at(x - d / 2, y - d / 2), *at(x + d / 2, y + d / 2)], fill=255)

    bar(*g["bar"])  # F: a floating bar over a Γ
    bar(0, 0, g["arm_right"], s)
    bar(0, 0, s, 1)
    m0, m1 = g["m"]  # M: two stems and a V down to the baseline
    bar(m0, 0, m0 + s, 1)
    bar(m1 - s, 0, m1, 1)
    for x in (m0 + s / 2, m1 - s / 2):
        ends = (x, s / 2), ((m0 + m1) / 2, 1 - s / 2)
        draw.line([at(*e) for e in ends], fill=255, width=round(s * u))
        for e in ends:
            dot(*e, s)
    c0, c1 = g["c"]  # C: a half ring, straight strokes and two dot eyes
    draw.ellipse([*at(c0, 0), *at(c0 + 1, 1)], fill=255)
    draw.ellipse([*at(c0 + s, s), *at(c0 + 1 - s, 1 - s)], fill=0)
    draw.rectangle([*at(c0 + 0.5, 0), *at(c1, 1)], fill=0)
    for y in (0, 1 - s):  # strokes meet the ring square, end round
        draw.rectangle([*at(c0 + 0.5, y), *at(c1 - s / 2, y + s)], fill=255)
        dot(c1 - s / 2, y + s / 2, s)
    for x in g["eyes"]:
        dot(x, 0.5, g["eye_diameter"])
    big.paste(255, (round(x0 + left * u), round(cap - glyph.height)), glyph)
    mask = big.resize((round(big.width / k), round(big.height / k)), Image.LANCZOS)
    return mask, x0 / k, (cap + u) / k


def build_chest_decal(source):
    """Factory chest print (INOVANCE, Powered by, FMC³, ROBOTICS) on the torso shell.

    MuJoCo ignores texture alpha, so the decal copies the shell's own triangles,
    vertices and compiled normals, lifted along the normals, with a pearl texture
    background: only the print shows and the background shades like the shell.
    """
    WORDMARK_DIR.mkdir(parents=True, exist_ok=True)
    stl = (source / "meshes/torso_Link.STL").read_bytes()
    logo = (source / FACTORY_PRINT["logo"]).read_bytes()
    params = [WORDMARK, FACTORY_PRINT, FMC3_GLYPHS, PEARL_SRGB, INK_SRGB]
    digest = hashlib.sha256(stl + logo + json.dumps(params, sort_keys=True).encode())
    digest = digest.hexdigest()
    manifest = WORDMARK_DIR / "chest_decal.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["digest"] == digest and all(
            (WORDMARK_DIR / info[k]).exists() for k in ("mesh", "texture")
        ):
            return info
    p, f = WORDMARK, FACTORY_PRINT
    mesh = trimesh.load(source / "meshes/torso_Link.STL", process=True, force="mesh")
    shell = sorted(
        mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True
    )[2]
    scratch = WORDMARK_DIR / "front_shell.stl"
    vert, face, facenormal, normal = compiled_mesh(shell, scratch)
    scratch.unlink()
    # From INOVANCE on top to the ROBOTICS line, plus the margin.
    y_lo = min(-p["width"] / 2, f["fmc3_left"]) - p["margin"]
    y_hi = p["width"] / 2 + p["margin"]
    z_lo = f["robotics_center_z"] - f["robotics_cap_height"] / 2 - p["margin"]
    z_hi = p["center_z"] + p["cap_height"] / 2 + p["margin"]
    tri = vert[face]
    lo, hi = tri.min(1), tri.max(1)
    outward = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 0] > 0
    keep = (
        outward
        & (hi[:, 1] > y_lo)
        & (lo[:, 1] < y_hi)
        & (hi[:, 2] > z_lo)
        & (lo[:, 2] < z_hi)
    )
    # One decal vertex per (vertex, normal) pair keeps every compiled normal, so
    # MuJoCo's per-vertex lighting reproduces the shell exactly.
    pairs = np.stack([face[keep].reshape(-1), facenormal[keep].reshape(-1)], 1)
    pairs, corner = np.unique(pairs, axis=0, return_inverse=True)
    nrm = normal[pairs[:, 1]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    pos = vert[pairs[:, 0]] + p["offset"] * nrm
    (y0, z0), (y1, z1) = pos[:, 1:].min(0), pos[:, 1:].max(0)
    uv = np.c_[(pos[:, 1] - y0) / (y1 - y0), (pos[:, 2] - z0) / (z1 - z0)]
    lines = ["# factory chest print decal on the torso front shell"]
    lines += [f"v {a:.7f} {b:.7f} {c:.7f}" for a, b, c in pos]
    lines += [f"vt {a:.7f} {b:.7f}" for a, b in uv]
    lines += [f"vn {a:.7f} {b:.7f} {c:.7f}" for a, b, c in nrm]
    for a, b, c in corner.reshape(-1, 3) + 1:
        lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
    (WORDMARK_DIR / "chest_decal.obj").write_text("\n".join(lines) + "\n")
    w = p["texture_width"]
    h = round(w * (z1 - z0) / (y1 - y0))
    sy, sz = w / (y1 - y0), h / (z1 - z0)
    image = Image.new("RGB", (w, h), PEARL_SRGB)

    def stamp(mask, y, z, dx, dy):
        # Ink through the mask, with its anchor (dx, dy) at the torso point (y, z).
        at = (round((y - y0) * sy - dx), round((z1 - z) * sz - dy))
        image.paste(Image.new("RGB", mask.size, INK_SRGB), at, mask)

    mask = wordmark_mask(p["width"] * sy, p["cap_height"] * sz)
    stamp(mask, 0.0, p["center_z"], mask.width / 2, mask.height / 2)
    mask, centroid = text_mask(
        "Powered by",
        f["powered_by_width"] * sy,
        f["powered_by_font_index"],
        f["powered_by_cap_height"] * sz,
    )
    stamp(mask, f["fmc3_left"], f["powered_by_center_z"], 0, centroid)
    mask, left, base = fmc3_mask(source / f["logo"], f["letter_height"] * sz)
    stamp(mask, f["fmc3_left"], f["baseline_z"], left, base)
    span = p["width"] / 2 - f["fmc3_left"]
    mask = tracked_mask(
        "ROBOTICS", span * sy, f["robotics_cap_height"] * sz, p["font_index"]
    )
    stamp(mask, f["fmc3_left"], f["robotics_center_z"], 0, mask.height / 2)
    image.save(WORDMARK_DIR / "chest_decal.png")
    info = {
        "digest": digest,
        "mesh": "chest_decal.obj",
        "texture": "chest_decal.png",
        "center_m": [0.0, p["center_z"]],
        "text_size_m": [p["width"], p["cap_height"]],
        "factory_print_m": {
            "powered_by": [f["powered_by_width"], f["powered_by_center_z"]],
            "fmc3": [f["fmc3_left"], f["letter_height"], f["baseline_z"]],
            "robotics": [f["robotics_cap_height"], f["robotics_center_z"]],
        },
        "faces": int(keep.sum()),
        "font": f"{p['font']}#{p['font_index']}",
    }
    manifest.write_text(json.dumps(info, indent=2))
    return info


def attach_chest_decal(root, info):
    asset = root.find("asset")
    _, specular, shininess, _ = PALETTE["fa_pearl"]
    ET.SubElement(
        asset,
        "texture",
        name="fa_wordmark",
        type="2d",
        file=str(WORDMARK_DIR / info["texture"]),
    )
    ET.SubElement(
        asset,
        "material",
        name="fa_wordmark",
        texture="fa_wordmark",
        rgba="1 1 1 1",
        texuniform="false",
        texrepeat="1 1",
        specular=specular,
        shininess=shininess,
    )
    ET.SubElement(
        asset,
        "mesh",
        name="inovance_chest_wordmark",
        file=str(WORDMARK_DIR / info["mesh"]),
        inertia="shell",
    )
    ET.SubElement(
        root.find(".//body[@name='torso_Link']"),
        "geom",
        name="inovance_chest_wordmark",
        type="mesh",
        mesh="inovance_chest_wordmark",
        material="fa_wordmark",
        contype="0",
        conaffinity="0",
        mass="0",
        group="1",
    )


def build_elbow_logos():
    """Black outer elbow caps with a white INOVANCE, printed like the chest wordmark.

    The decal copies the outer faces of the painted pearl elbow mesh as MuJoCo
    compiles it. Its texture draws the black cap as an exact circle on a pearl
    background, so the cap edge stays round whatever the triangulation.
    """
    p = ELBOW_LOGO
    infos = {}
    for side, sign in (("left", 1), ("right", -1)):
        group = OUTPUT / f"{side}_elbow_pitch_Link__fa_pearl.stl"
        key = (
            group.read_bytes()
            + json.dumps([p, PEARL_SRGB, BLACK_SRGB, WORDMARK]).encode()
        )
        digest = hashlib.sha256(key).hexdigest()
        manifest = WORDMARK_DIR / f"elbow_logo_{side}.json"
        if manifest.exists():
            info = json.loads(manifest.read_text())
            if info["digest"] == digest and all(
                (WORDMARK_DIR / info[k]).exists() for k in ("mesh", "texture")
            ):
                infos[side] = info
                continue
        vert, face, facenormal, normal = compiled_file(group)
        tri = vert[face]
        outward = (
            sign * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 1] > 0
        )
        keep = outward & (np.hypot(tri[..., 0], tri[..., 2]) <= p["patch_radius"]).all(
            1
        )
        keep &= (sign * tri[..., 1] > 0.03).all(1)
        pairs = np.stack([face[keep].reshape(-1), facenormal[keep].reshape(-1)], 1)
        pairs, corner = np.unique(pairs, axis=0, return_inverse=True)
        nrm = normal[pairs[:, 1]]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        pos = vert[pairs[:, 0]] + p["offset"] * nrm
        # Reads left to right seen from outside the arm, upright towards the shoulder.
        s, t = -sign * pos[:, 0], pos[:, 2]
        (s0, t0), (s1, t1) = (s.min(), t.min()), (s.max(), t.max())
        uv = np.c_[(s - s0) / (s1 - s0), (t - t0) / (t1 - t0)]
        lines = [f"# INOVANCE logo on the {side} outer elbow cap"]
        lines += [f"v {a:.7f} {b:.7f} {c:.7f}" for a, b, c in pos]
        lines += [f"vt {a:.7f} {b:.7f}" for a, b in uv]
        lines += [f"vn {a:.7f} {b:.7f} {c:.7f}" for a, b, c in nrm]
        for a, b, c in corner.reshape(-1, 3) + 1:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        (WORDMARK_DIR / f"elbow_logo_{side}.obj").write_text("\n".join(lines) + "\n")
        w = p["texture_width"]
        h = round(w * (t1 - t0) / (s1 - s0))
        image = Image.new("RGB", (w, h), PEARL_SRGB)
        cx, cy = -s0 / (s1 - s0) * w, t1 / (t1 - t0) * h
        rx, ry = p["cap_radius"] / (s1 - s0) * w, p["cap_radius"] / (t1 - t0) * h
        ImageDraw.Draw(image).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), BLACK_SRGB)
        mask = wordmark_mask(
            p["width"] / (s1 - s0) * w, p["cap_height"] / (t1 - t0) * h
        )
        at = (
            round(-s0 / (s1 - s0) * w - mask.width / 2),
            round(t1 / (t1 - t0) * h - mask.height / 2),
        )
        image.paste(Image.new("RGB", mask.size, p["ink"]), at, mask)
        image.save(WORDMARK_DIR / f"elbow_logo_{side}.png")
        infos[side] = {
            "digest": digest,
            "mesh": f"elbow_logo_{side}.obj",
            "texture": f"elbow_logo_{side}.png",
            "faces": int(keep.sum()),
        }
        manifest.write_text(json.dumps(infos[side], indent=2))
    return infos


def attach_elbow_logos(root, infos):
    asset = root.find("asset")
    # Pearl specular so the decal's pearl surround shades like the housing.
    _, specular, shininess, _ = PALETTE["fa_pearl"]
    for side, info in infos.items():
        name = f"inovance_elbow_logo_{side}"
        ET.SubElement(
            asset,
            "texture",
            name=name,
            type="2d",
            file=str(WORDMARK_DIR / info["texture"]),
        )
        ET.SubElement(
            asset,
            "material",
            name=name,
            texture=name,
            rgba="1 1 1 1",
            texuniform="false",
            texrepeat="1 1",
            specular=specular,
            shininess=shininess,
        )
        ET.SubElement(
            asset,
            "mesh",
            name=name,
            file=str(WORDMARK_DIR / info["mesh"]),
            inertia="shell",
        )
        ET.SubElement(
            root.find(f".//body[@name='{side}_elbow_pitch_Link']"),
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            material=name,
            contype="0",
            conaffinity="0",
            mass="0",
            group="1",
        )


def vec(values):
    return " ".join(f"{v:.9g}" for v in values)


def rotate(rotation, vectors):
    # Element-wise products: Accelerate's matmul raises spurious FP warnings here.
    return (rotation.as_matrix() * np.asarray(vectors)[..., None, :]).sum(-1)


def prepare_d405():
    # The realsense-ros D405 has no lens geometry: its flat front is the glass.
    src = CAMERA_DIR / "realsense_d405/d405.stl"
    out = {
        "fa_camera_grey": CAMERA_DIR / "realsense_d405/d405_housing.stl",
        "fa_cam_glass": CAMERA_DIR / "realsense_d405/d405_glass.stl",
    }
    if all(
        p.exists() and p.stat().st_mtime >= src.stat().st_mtime for p in out.values()
    ):
        return out
    mesh = trimesh.load(src, process=True, force="mesh")
    glass = (mesh.face_normals[:, 2] > 0.9) & (mesh.triangles_center[:, 2] > -0.0005)
    for name, keep in (("fa_cam_glass", glass), ("fa_camera_grey", ~glass)):
        mesh.submesh([np.nonzero(keep)[0]], append=True).export(out[name])
    return out


def attach_real_cameras(root):
    """Official RealSense meshes replace the URDF camera boxes on the same mounts.

    The head keeps its URDF geom so the recipe can paint the bracket merged into it.
    """
    assets = root.find("asset")
    d405 = prepare_d405()
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            mesh = geom.get("mesh")
            kind = REAL_CAMERAS.get(mesh)
            if geom.get("class") != "visual" or kind is None:
                continue
            pos = np.array(geom.get("pos", "0 0 0").split(), float)
            w, x, y, z = map(float, geom.get("quat", "1 0 0 0").split())
            frame = Rotation.from_quat([x, y, z, w])
            if mesh != "head_d435_Link":
                body.remove(geom)
            if kind == "d435":
                offset, turn = D435_ALIGN
                parts = [(CAMERA_DIR / "realsense_d435i" / f, m) for f, m in D435_PARTS]
                pos, frame = pos + rotate(frame, offset), frame * turn
            else:
                # On the tilted wrist block, as its camera (camera_rig.py).
                side = mesh.split("_")[0]
                pos, turned = wrist_mount.sensor_pose(side, pos, frame.as_matrix())
                frame = Rotation.from_matrix(turned)
                parts = [(path, m) for m, path in d405.items()]
            qx, qy, qz, qw = frame.as_quat()
            for i, (path, material) in enumerate(parts):
                name = f"real_{mesh}_{i}"
                ET.SubElement(
                    assets, "mesh", name=name, file=str(path), inertia="shell"
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="mesh",
                    mesh=name,
                    material=material,
                    pos=vec(pos),
                    quat=vec([qw, qx, qy, qz]),
                    mass="0",
                    **{"class": "visual"},
                )


def attach_wrist_blocks(root):
    """The tilted D405 block (hardware/wrist_d405_mount) on each clevis pin, in
    place of the original block that the recipe drops."""
    assets = root.find("asset")
    ET.SubElement(
        assets,
        "mesh",
        name="wrist_d405_block",
        file=str(wrist_mount.STL),
        scale="0.001 0.001 0.001",
        inertia="shell",
    )
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_wrist_roll_Link']")
        R, t = wrist_mount.part_to_wrist(side)
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        ET.SubElement(
            body,
            "geom",
            name=f"{side}_wrist_d405_block",
            type="mesh",
            mesh="wrist_d405_block",
            material="fa_black",
            pos=vec(t / 1000),
            quat=vec([w, x, y, z]),
            mass="0",
            **{"class": "visual"},
        )


def stadium(width, height, segments):
    # Closed outline (u along the width, v up) with semicircular ends; outlines
    # with the same segments correspond point for point.
    r, half = height / 2, width / 2 - height / 2
    a = np.linspace(-np.pi / 2, np.pi / 2, segments + 1)
    right = np.c_[half + r * np.cos(a), r * np.sin(a)]
    return np.r_[right, -right]


def attach_chest_slot(root):
    """A light bezel with a narrow tinted window in the chest camera slot."""
    p = CHEST_SLOT
    cx, cy = p["centre"]
    w, h = p["aperture"]
    outer = stadium(w + 2 * p["margin"], h + 2 * p["margin"], p["segments"])
    slit = stadium(*p["slit"], p["segments"])
    n = len(outer)

    def points(outline, z):
        # (u, v) to the sensor frame: u is the width (y), v the height (x).
        return np.c_[cx + outline[:, 1], cy + outline[:, 0], np.full(len(outline), z)]

    window = np.r_[points(outer, p["window_z"]), [[cx, cy, p["window_z"]]]]
    window_faces = [[i, (i + 1) % n, n] for i in range(n)]
    bezel = np.r_[points(outer, p["bezel_z"]), points(slit, p["bezel_z"])]
    bezel_faces = []
    for i in range(n):
        j = (i + 1) % n
        bezel_faces += [[i, j, n + j], [i, n + j, n + i]]
    site = root.find(".//site[@name='body_d435']")
    body = next(b for b in root.iter("body") if site in list(b))
    assets = root.find("asset")
    for name, vert, faces, material in (
        ("chest_slot_window", window, window_faces, "fa_slot_window"),
        ("chest_slot_bezel", bezel, bezel_faces, "fa_pearl"),
    ):
        faces = np.array(faces)
        tri = vert[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        faces[normal[:, 2] < 0] = faces[normal[:, 2] < 0][:, ::-1]  # face outwards
        ET.SubElement(
            assets,
            "mesh",
            name=name,
            vertex=" ".join(f"{v:.7f}" for v in vert.ravel()),
            face=" ".join(str(i) for i in faces.ravel()),
            inertia="shell",
        )
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            material=material,
            pos=site.get("pos"),
            quat=site.get("quat"),
            mass="0",
            **{"class": "visual"},
        )


def apply_appearance(root, source):
    assets = root.find("asset")
    for name, (rgba, specular, shininess, emission) in PALETTE.items():
        ET.SubElement(
            assets,
            "material",
            name=name,
            rgba=rgba,
            specular=specular,
            shininess=shininess,
            emission=emission,
        )
    attach_real_cameras(root)
    attach_wrist_blocks(root)
    attach_chest_slot(root)
    manifest = prepare_paint(source)
    painted = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            mesh = geom.get("mesh")
            if geom.get("class") != "visual" or mesh not in manifest["meshes"]:
                continue
            body.remove(geom)
            frame = {k: geom.get(k) for k in ("pos", "quat") if geom.get(k)}
            for entry in manifest["meshes"][mesh]:
                name = f"paint_{mesh}_{entry['material']}"
                parent, where = body, frame
                if "part" in entry:
                    # A gripper link or jaw block: same place at closure 0, in
                    # the body that moves it (no rotation below the wrist).
                    assert "quat" not in frame, mesh
                    side = mesh.split("_")[0]
                    name = f"paint_{mesh}_{entry['part']}_{entry['material']}"
                    target = body_name(side, entry["part"])
                    parent = root.find(f".//body[@name='{target}']")
                    pos = np.array(frame.get("pos", "0 0 0").split(), float)
                    where = {"pos": vec(pos - offset_in_wrist(root, side, target))}
                ET.SubElement(
                    assets,
                    "mesh",
                    name=name,
                    file=str(OUTPUT / entry["file"]),
                    inertia="shell",
                )
                ET.SubElement(
                    parent,
                    "geom",
                    name=name,
                    type="mesh",
                    mesh=name,
                    material=entry["material"],
                    mass="0",
                    **where,
                    **{"class": "visual"},
                )
                painted += 1
    # The printed chest decal replaces the source's flat FMC3 logo box.
    for parent in root.iter("body"):
        logo = parent.find("geom[@name='chest_logo']")
        if logo is not None:
            parent.remove(logo)
    # URDF sensor sites are frames, not hardware: hide their marker spheres.
    for site in root.iter("site"):
        site.set("group", site.get("group", "4"))
    wordmark = build_chest_decal(source)
    attach_chest_decal(root, wordmark)
    attach_elbow_logos(root, build_elbow_logos())
    fitted = attach_fittings(root, build_fittings(source, FITTINGS_DIR))
    keys = ("center_m", "text_size_m", "factory_print_m", "font")
    return {
        "paint_key": manifest["key"],
        "painted_meshes": len(manifest["meshes"]),
        "visual_geoms": painted,
        "wordmark": {k: wordmark[k] for k in keys},
        "fittings": fitted,
        "physical_model_modified": False,
        "head_camera_mount": "source URDF mount retained behind the front cover",
    }
