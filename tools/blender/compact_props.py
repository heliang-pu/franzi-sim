"""Compact delivered prop files without changing mesh topology or source art.

Run with the inovance-fineart Python, e.g. compact_props.py --apply. Builders
call this after saving their sources. Only visually approved PNGs use palettes. Five
decimal metre positions are used wherever possible; individual vertices keep
more decimals when needed to stay within 5 micrometres or preserve triangles.
Collision files additionally retain enough precision to stay strictly convex.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from PIL import Image
from PIL.PngImagePlugin import PngInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "assets" / "props"
QA = HERE / "_qa" / "task13"
POSITION_ERROR_M = 0.000005
OBJ_MARKER = "# Prop compact format v1; topology preserved"


def category(path):
    if path.name == "source.blend":
        return "source.blend"
    if path.suffix == ".blend1":
        return "blend1"
    if path.suffix == ".obj":
        return "obj"
    if path.suffix == ".png":
        return "previews" if path.name.startswith("preview") else "textures"
    return "other"


def inventory(root=ASSETS):
    categories = dict.fromkeys(("textures", "obj", "previews", "source.blend", "blend1", "other"), 0)
    props = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        categories[category(path)] += size
        name = path.relative_to(root).parts[0]
        row = props.setdefault(name, {"total": 0, "without_source": 0})
        row["total"] += size
        if path.name != "source.blend":
            row["without_source"] += size
    return {"props": props, "categories": categories}


def save_png(path, *, resize=None):
    """Lossless RGB/RGBA re-encoding; resizing only when explicitly requested."""
    path = Path(path)
    with Image.open(path) as source:
        source.load()
        original_size = source.size
        info = dict(source.info)
        image = source.copy()
    if image.mode in ("RGBA", "LA") or "transparency" in info:
        image = image.convert("RGBA")
        if image.getchannel("A").getextrema() == (255, 255):
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")
    if resize is not None and image.size != resize:
        image = image.resize(resize, Image.Resampling.LANCZOS)
    metadata = PngInfo()
    for key, value in info.items():
        if isinstance(value, str) and key not in ("transparency", "icc_profile"):
            metadata.add_text(key, value)
    kwargs = {"optimize": True, "pnginfo": metadata}
    for key in ("icc_profile", "dpi", "exif"):
        if key in info:
            kwargs[key] = info[key]
    before = path.stat().st_size
    temporary = path.with_name(path.name + ".compact-tmp")
    image.save(temporary, format="PNG", **kwargs)
    with Image.open(temporary) as verified:
        if verified.mode != image.mode or verified.size != image.size or verified.tobytes() != image.tobytes():
            raise ValueError(f"PNG re-encoding changed pixels: {path}")
    temporary.replace(path)
    return {"file": str(path), "before": before, "after": path.stat().st_size,
            "size_before": original_size, "size_after": image.size, "mode": image.mode}


def png_size(path):
    with Path(path).open("rb") as stream:
        return struct.unpack(">II", stream.read(24)[16:24])


def quantize_png(path):
    """Apply only the separately reviewed 256-colour Floyd-Steinberg policy."""
    with Image.open(path) as image:
        if image.mode == "P":
            result = image.copy()
        else:
            rgb = image.convert("RGB")
            palette = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            result = rgb.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        result.save(path, optimize=True, bits=8)


def apply_palette_policy(root, selected=None):
    policy = HERE / "compact_texture_policy.json"
    if not policy.is_file():
        return []
    changed = []
    for relative in json.loads(policy.read_text())["images"]:
        path = root / relative
        if path.is_file() and (selected is None or path.parent in selected):
            quantize_png(path)
            changed.append(relative)
    return changed


def is_baked(path):
    return path.suffix.lower() == ".png" and "albedo" in path.name


def downscale(paths):
    changes = []
    for path in paths:
        # The floor atlas is a 1 m repeat tile: keep both it and its source art.
        if is_baked(path) and path.parent.name != "factory_floor" and max(png_size(path)) > 1024:
            changes.append(save_png(path, resize=(1024, 1024)))
    return changes


def fix_texture_sizes(data, folder):
    """Include replacement-part manifests inside variants, not just top-level parts."""
    if isinstance(data, dict):
        if data.get("texture") and "texture_px" in data:
            width, height = png_size(folder / data["texture"])
            if width != height:
                raise ValueError(f"Non-square atlas: {folder / data['texture']}")
            data["texture_px"] = width
        for child in data.values():
            fix_texture_sizes(child, folder)
    elif isinstance(data, list):
        for child in data:
            fix_texture_sizes(child, folder)


def write_json_if_changed(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.read_text() != text:
        path.write_text(text)


def update_manifests(root):
    for path in sorted(root.glob("*/prop.json")):
        data = json.loads(path.read_text())
        fix_texture_sizes(data, path.parent)
        note = (" Task 13 delivery: baked albedos are Lanczos-downsampled to at most 1024 px "
                "except the unchanged-resolution factory-floor repeat tile; previews are 512 square. "
                "PNG encoding uses RGB8 or reviewed 256-colour Floyd-Steinberg palettes; source artwork pixels "
                "and source.blend are retained. Shared PNG paths resolve from this prop folder.")
        # Replace the delivery clause without losing the precision-exception clause.
        prefix, _, remainder = data["notes"].partition(" Task 13 delivery:")
        precision_marker = " Task 13 compact OBJ exception:"
        _, has_exception, exception_text = remainder.partition(precision_marker)
        tail = precision_marker + exception_text if has_exception else ""
        data["notes"] = prefix + note + tail
        write_json_if_changed(path, data)


def canonical_order(path):
    # The library/source owners must win over dependent instances, so a rebuild
    # cannot accidentally overwrite a shared texture with an unrelated atlas.
    priorities = {"panel_devices": 0, "small_tote": 1, "parcel_box_s": 1,
                  "parcel_box_m": 1, "background_rack": 9, "cnc_front": 4,
                  "control_console": 3}
    return priorities.get(path.parent.name, 2), path.as_posix()


def deduplicate(root):
    groups = {}
    for path in sorted(root.glob("*/*.png")):
        if is_baked(path) and path.parent.name != "factory_floor":
            groups.setdefault(hashlib.sha256(path.read_bytes()).digest(), []).append(path.resolve())
    replacements = {}
    for paths in groups.values():
        canonical = min(paths, key=canonical_order)
        for path in paths:
            if path != canonical:
                replacements[path] = canonical

    def rewrite(value, folder):
        if isinstance(value, dict):
            return {key: child if key == "notes" else rewrite(child, folder) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child, folder) for child in value]
        if isinstance(value, str) and value.endswith(".png"):
            resolved = (folder / value).resolve()
            if resolved in replacements:
                return Path(os.path.relpath(replacements[resolved], folder)).as_posix()
        return value

    for path in sorted(root.glob("*/prop.json")):
        write_json_if_changed(path, rewrite(json.loads(path.read_text()), path.parent))
    records = []
    for path, canonical in sorted(replacements.items()):
        records.append({"removed": str(path.relative_to(root)),
                        "shared": str(canonical.relative_to(root)), "bytes": path.stat().st_size})
        path.unlink()
    return records


def compact_visual_indices(text):
    """Losslessly share equal rounded UVs and use shorter legal OBJ indices.

    All attribute declarations precede faces, so relative indices resolve the
    same way in Blender, MuJoCo and trimesh (no interleaved-declaration trick).
    Position order and face/corner order are unchanged. Collision files never
    pass through this function.
    """
    lines = text.splitlines()
    counts = {tag: sum(line.startswith(tag + " ") for line in lines) for tag in ("v", "vt", "vn")}
    seen, uv_map, filtered = {}, {}, []
    uv_index = 0
    found_face = False
    for line in lines:
        if line.startswith("f "):
            found_face = True
        if line.startswith(("v ", "vt ", "vn ")) and found_face:
            return text  # Leave valid but interleaved OBJ files alone.
        if line.startswith("vt "):
            uv_index += 1
            if line in seen:
                uv_map[uv_index] = seen[line]
                continue
            seen[line] = len(seen) + 1
            uv_map[uv_index] = seen[line]
        filtered.append(line)

    def index(value, tag):
        original = int(value)
        positive = original if original > 0 else counts[tag] + original + 1
        if tag == "vt":
            positive = uv_map[positive]
            total = len(seen)
        else:
            total = counts[tag]
        negative = positive - total - 1
        return min((str(positive), str(negative)), key=len)

    output = []
    for line in filtered:
        if line.startswith("f "):
            corners = []
            for token in line.split()[1:]:
                attributes = token.split("/")
                corners.append("/".join(index(value, tag) if value else "" for value, tag
                                        in zip(attributes, ("v", "vt", "vn"))))
            line = "f " + " ".join(corners)
        output.append(line)
    return "\n".join(output) + "\n"


def round_obj(path, root=ASSETS):
    """Keep every face, its order, vertex index and object/material directive.

    Independent five-decimal rounding can move a vertex by sqrt(3)*5 um and
    collapse sub-10 um bevel triangles. Promote only necessary points. A
    collision piece may need uniform higher precision to retain coplanar hull
    faces; never change its topology to repair a rounding artefact.
    """
    import numpy as np
    from validate_props import read_obj

    data = read_obj(path)
    original = data.vertices
    already_compact = OBJ_MARKER in path.read_text().splitlines()
    decimals = np.full(len(original), 5, dtype=int)
    points = np.round(original, 5)
    promote = np.linalg.norm(points - original, axis=1) > POSITION_ERROR_M + 1e-14
    decimals[promote] = 6
    points[promote] = np.round(original[promote], 6)
    collision = "_col_" in path.name
    if already_compact:
        points = original.copy()
        tokens = [line.split()[1:] for line in path.read_text().splitlines() if line.startswith("v ")]
        decimals = np.asarray([max(len(token.partition(".")[2]) for token in row) for row in tokens])
    if collision and not already_compact:
        base_mesh = data.mesh()

        def valid(candidate):
            # base_mesh may weld normals/UV seams; use unprocessed indices to
            # preserve the exact original face topology before welding.
            import trimesh
            mesh = trimesh.Trimesh(vertices=candidate, faces=data.faces, process=True)
            return (len(mesh.vertices) == len(base_mesh.vertices) and mesh.is_watertight
                    and mesh.is_winding_consistent and mesh.is_convex
                    and (mesh.area_faces > 1e-18).all())

        if not valid(points):
            for precision in range(6, 13):
                candidate = np.round(original, precision)
                if valid(candidate):
                    points = candidate
                    decimals[:] = precision
                    break
            else:
                raise ValueError(f"Cannot preserve collision geometry by rounding: {path}")
    elif not already_compact:
        for precision in range(6, 13):
            tris = points[data.faces]
            area2 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
            bad = area2 <= 2e-18
            if not bad.any():
                break
            indices = np.unique(data.faces[bad])
            decimals[indices] = precision
            points[indices] = np.round(original[indices], precision)
        else:
            raise ValueError(f"Cannot preserve visual triangles by rounding: {path}")
    displacement = np.linalg.norm(points - original, axis=1)
    if displacement.max() > POSITION_ERROR_M + 1e-14:
        raise ValueError(f"Position movement exceeds 0.005 mm: {path}")
    source = path.read_text().splitlines()
    output = []
    vertex_index = 0
    preserved = []
    for line in source:
        fields = line.split()
        if fields and fields[0] in ("v", "vt", "vn"):
            tag = fields[0]
            if tag == "v":
                precision = int(decimals[vertex_index])
                coordinates = points[vertex_index]
                vertex_index += 1
            else:
                precision = 4
                coordinates = [float(value) for value in fields[1:]]
            formatted = []
            for value in coordinates:
                token = f"{value:.{precision}f}"
                if float(token) == 0:
                    token = token.lstrip("-")
                formatted.append(token)
            output.append(tag + " " + " ".join(formatted))
        else:
            output.append(line)
            preserved.append(line)
    before = path.stat().st_size
    result = "\n".join(output) + "\n"
    if not already_compact:
        result = OBJ_MARKER + "\n" + result
    if not collision:
        result = compact_visual_indices(result)
    if result != path.read_text():
        path.write_text(result)
    final = read_obj(path)
    assert np.array_equal(final.vertices, points) and np.array_equal(final.faces, data.faces)
    if len(data.uv):
        assert np.array_equal(final.uv[final.uv_faces], np.round(data.uv, 4)[data.uv_faces])
    if len(data.normals):
        assert np.array_equal(final.normals[final.normal_faces], np.round(data.normals, 4)[data.normal_faces])
    # Face/index topology and non-coordinate directives are byte-for-byte equal.
    assert preserved == [line for line in output if not line.split() or line.split()[0] not in ("v", "vt", "vn")]
    return {"file": str(path.relative_to(root)),
            "before": before, "after": path.stat().st_size, "vertices": len(original),
            "collision": collision, "max_vertex_change_mm": float(displacement.max() * 1000),
            "higher_precision_vertices": int((decimals > 5).sum()),
            "max_decimals": int(decimals.max()),
            "preserved_topology_sha256": hashlib.sha256("\n".join(preserved).encode()).hexdigest()}


def record_precision_notes(root, records):
    by_prop = {}
    for record in records:
        if record["higher_precision_vertices"]:
            by_prop.setdefault(Path(record["file"]).parts[0], []).append(record)
    for name, rows in by_prop.items():
        path = root / name / "prop.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        marker = " Task 13 compact OBJ exception:"
        data["notes"] = data["notes"].partition(marker)[0]
        data["notes"] += (marker + " positions use 5 decimal metres where safe; necessary vertices "
                          "retain up to " + str(max(row["max_decimals"] for row in rows)) +
                          " decimals to preserve nonzero triangles, strict collision convexity and "
                          "the 0.005 mm Euclidean movement bound. Independent fixed-5 rounding "
                          "cannot satisfy all three. Normals/UVs use 4 decimals; every original triangle, "
                          "position index and per-corner attribute value is retained. Equal rounded UV "
                          "records share indices and visual OBJ indices use shorter equivalent relative "
                          "notation. No collision remeshing, reindexing or simplification.")
        write_json_if_changed(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--root", type=Path, default=ASSETS)
    parser.add_argument("--props", nargs="+")
    parser.add_argument("--step", choices=("all", "downscale", "deduplicate", "optimize", "previews", "obj", "palette"), default="all")
    parser.add_argument("--png", type=Path, help="Optimise one freshly exported PNG (used by prop_lib)")
    args = parser.parse_args()
    if args.png:
        path = args.png.resolve()
        size = (1024, 1024) if is_baked(path) and path.parent.name != "factory_floor" and max(png_size(path)) > 1024 else None
        save_png(path, resize=size)
        return
    root = args.root.resolve()
    if not args.apply:
        print(json.dumps(inventory(root), indent=2))
        return
    selected = {root / name for name in args.props} if args.props else set(p.parent for p in root.glob("*/prop.json"))
    paths = sorted(path for folder in selected for path in folder.rglob("*") if path.is_file())
    report = {}
    if args.step in ("all", "downscale"):
        report["downscaled"] = downscale(paths)
        update_manifests(root)
    if args.step in ("all", "deduplicate"):
        report["deduplicated"] = deduplicate(root)
    if args.step in ("all", "optimize"):
        report["optimized_pngs"] = [save_png(path) for path in paths if path.suffix == ".png" and path.exists()]
    if args.step in ("all", "previews"):
        report["previews"] = [save_png(path, resize=(512, 512)) for path in paths
                              if path.suffix == ".png" and path.name.startswith("preview") and path.exists()]
    if args.step in ("all", "obj"):
        report["obj"] = [round_obj(path, root) for path in paths if path.suffix == ".obj"]
        record_precision_notes(root, report["obj"])
    if args.step in ("all", "palette"):
        report["palette_approved"] = apply_palette_policy(root, selected)
    if args.step == "all":
        report["deduplicated_after_encoding"] = deduplicate(root)
    update_manifests(root)
    QA.mkdir(parents=True, exist_ok=True)
    destination = QA / ("compression.json" if not args.props else "last_rebuild_compression.json")
    if args.step != "all":
        destination = QA / f"{args.step}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in report.items()}))
    print(json.dumps(inventory(root)["categories"]))


if __name__ == "__main__":
    main()
