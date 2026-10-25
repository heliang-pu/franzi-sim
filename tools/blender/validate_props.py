"""Validate exported props with numpy/trimesh, without importing Blender.

Run with /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python.
Convexity is checked in both directions: hull/piece >= .99 alone also passes
every concave solid, so we additionally require piece/hull >= .99 and convex
face half-spaces. OBJ geometry is read directly to keep the prop frame intact.
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import trimesh
from PIL import Image
from prop_schema import TRIANGLE_BUDGETS, budget_class, referenced_files, schema_errors

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ObjData:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    uv_faces: np.ndarray
    normals: np.ndarray
    normal_faces: np.ndarray

    def mesh(self):
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=True)


def read_obj(path):
    vertices, uv, normals, faces, uv_faces, normal_faces = [], [], [], [], [], []

    def index(token, count):
        value = int(token)
        result = value - 1 if value > 0 else count + value
        if value == 0 or not 0 <= result < count:
            raise ValueError("OBJ index is out of range")
        return result

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(v) for v in fields[1:4]])
            elif fields[0] == "vt":
                uv.append([float(v) for v in fields[1:3]])
            elif fields[0] == "vn":
                normals.append([float(v) for v in fields[1:4]])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError("OBJ must contain triangles only (no quads or n-gons)")
                corners = [corner.split("/") for corner in fields[1:]]
                faces.append([index(c[0], len(vertices)) for c in corners])
                uv_faces.append([index(c[1], len(uv)) if len(c) > 1 and c[1] else -1 for c in corners])
                normal_faces.append(
                    [index(c[2], len(normals)) if len(c) > 2 and c[2] else -1 for c in corners]
                )
    verts, tris = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or tris.size == 0 or not np.isfinite(verts).all():
        raise ValueError("OBJ needs finite XYZ vertices and nonempty triangles")
    return ObjData(
        verts,
        tris,
        np.asarray(uv, dtype=float),
        np.asarray(uv_faces),
        np.asarray(normals, dtype=float),
        np.asarray(normal_faces),
    )


def check_png(path, size):
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    if (width, height) != (size, size):
        raise ValueError(f"{width}x{height}, expected {size}x{size}")
    if bit_depth != 8 or color_type not in (2, 3, 6):
        raise ValueError("expected 8-bit RGB, indexed RGB or RGBA PNG")
    with Image.open(path) as image:
        image.verify()


@dataclass
class Result:
    prop_id: str
    errors: list = field(default_factory=list)
    bbox: str = "-"
    triangles: int = 0
    budget: int = 0
    collisions: int = 0
    max_vertices: int = 0
    textures: set = field(default_factory=set)

    def row(self):
        textures = ",".join(f"{s}x{s}" for s in sorted(self.textures)) or "-"
        return [
            self.prop_id,
            "FAIL" if self.errors else "PASS",
            self.bbox,
            f"{self.triangles:,}/{self.budget:,}",
            str(self.collisions),
            str(self.max_vertices),
            textures,
        ]


def validate_prop(folder):
    folder = Path(folder).resolve()
    result = Result(folder.name)
    try:
        data = json.loads((folder / "prop.json").read_text(encoding="utf-8"))
        result.errors.extend(schema_errors(data))
        if result.errors:
            return result
        if data["id"] != folder.name:
            result.errors.append("id does not match folder name")
        result.budget = TRIANGLE_BUDGETS[budget_class(data)]
        required = {
            data.get("source", "source.blend"),
            data.get("preview", "preview.png"),
            *referenced_files(data),
        }
        # Git leaves out the source .blend files (the build scripts rebuild
        # them), so they are checked only where they exist.
        required = {
            ref
            for ref in required
            if not ref.lower().endswith(".blend") or (folder / ref).exists()
        }
        for reference in sorted(required):
            target = (folder / reference).resolve()
            # Task 13 shares sibling PNG atlases. Meshes/sources still belong
            # to their own prop, and even symlinks cannot escape the props root.
            allowed = folder.parent if Path(reference).suffix.lower() == ".png" else folder
            if not target.is_relative_to(allowed) or not target.is_file():
                result.errors.append(f"missing/unsafe file: {reference}")
        if result.errors:
            return result
        check_png(folder / data.get("preview", "preview.png"), 512)
        for preview in folder.glob("preview*.png"):
            check_png(preview, 512)
        visual_points = []
        texture_sizes = {}
        all_collisions = set()
        for part in data["parts"]:
            name = part["name"]
            try:
                visual = read_obj(folder / part["visual"])
                visual_points.append(visual.vertices)
                result.triangles += len(visual.faces)
                if len(visual.normals) == 0 or (visual.normal_faces < 0).any():
                    result.errors.append(f"{name}: missing exported normals")
                elif (
                    not np.isfinite(visual.normals).all()
                    or (np.linalg.norm(visual.normals, axis=1) < 0.9).any()
                ):
                    result.errors.append(f"{name}: invalid normals")
                if part["texture"]:
                    if len(visual.uv) == 0 or (visual.uv_faces < 0).any():
                        result.errors.append(f"{name}: missing UV coordinates")
                    elif (
                        not np.isfinite(visual.uv).all()
                        or visual.uv.min() < -1e-5
                        or visual.uv.max() > 1 + 1e-5
                    ):
                        result.errors.append(f"{name}: UV coordinates outside [0,1]")
                    else:
                        corners = visual.uv[visual.uv_faces]
                        a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
                        area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2
                        if area <= 1e-8:
                            result.errors.append(f"{name}: degenerate UV atlas (zero total area)")
                    texture_sizes[part["texture"]] = part["texture_px"]
                mesh = visual.mesh()
                if (mesh.area_faces <= 1e-18).any():
                    result.errors.append(f"{name}: zero-area visual triangle")
                if len(np.unique(np.sort(visual.faces, axis=1), axis=0)) != len(visual.faces):
                    result.errors.append(f"{name}: duplicate visual triangle")
                for primitive in part.get("collision_primitives", []):
                    result.collisions += 1
                    # Schema has already checked type, static placement and
                    # finite dimensions. Its support surface must coincide
                    # with the visual top, to the general 1 mm tolerance.
                    normal = np.asarray(primitive["normal"])
                    distances = (visual.vertices - np.asarray(primitive["pos_m"])) @ normal
                    if abs(float(distances.max())) > 0.001 + 1e-9:
                        result.errors.append(f"{name}: collision plane does not coincide with visual surface")
            except Exception as exc:  # noqa: BLE001 - report every bad asset, continue other parts.
                result.errors.append(f"{name}: visual: {exc}")
            for reference in part["collision"]:
                if reference in all_collisions:
                    result.errors.append(f"{name}: collision shared by multiple rigid bodies: {reference}")
                all_collisions.add(reference)
                result.collisions += 1
                try:
                    obj = read_obj(folder / reference)
                    piece = obj.mesh()
                    vertices = len(piece.vertices)
                    result.max_vertices = max(result.max_vertices, vertices)
                    if vertices > 64:
                        result.errors.append(f"{reference}: {vertices} vertices > 64")
                    if not piece.is_watertight or not piece.is_winding_consistent:
                        result.errors.append(f"{reference}: not a closed, consistently oriented solid")
                        continue
                    volume = abs(float(piece.volume))
                    hull_volume = abs(float(piece.convex_hull.volume))
                    if min(volume, hull_volume) < 1e-16:
                        result.errors.append(f"{reference}: zero collision volume")
                        continue
                    ratio = hull_volume / volume
                    if not 0.99 <= ratio <= 1 / 0.99 or not piece.is_convex:
                        result.errors.append(f"{reference}: nonconvex; hull/piece volume={ratio:.6f}")
                except Exception as exc:  # noqa: BLE001 - report every bad collision file.
                    result.errors.append(f"{reference}: collision: {exc}")
        if visual_points:
            points = np.concatenate(visual_points)
            lower, upper = points.min(axis=0) * 1000, points.max(axis=0) * 1000
            actual = upper - lower
            result.bbox = " x ".join(f"{v:.3f}" for v in actual)
            tolerance = 0.1 if "precision_mm" in data else 1.0
            delta = np.abs(actual - np.asarray(data["bbox_mm"]))
            if np.any(delta > tolerance + 1e-6):
                result.errors.append(f"bbox differs by {delta.tolist()} mm (tolerance {tolerance})")
            if data["origin"] == "bottom_center":
                offset = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]])
                if np.any(np.abs(offset) > tolerance + 1e-6):
                    result.errors.append(f"bottom-centre origin offset is {offset.tolist()} mm")
        if result.triangles != data["triangles"]:
            result.errors.append(f"triangles metadata={data['triangles']}, exported={result.triangles}")
        if result.triangles > result.budget:
            result.errors.append(f"triangle budget exceeded ({result.budget})")
        for variant in data["variants"]:
            for name, texture in variant.get("textures", {}).items():
                part = next(p for p in data["parts"] if p["name"] == name)
                if part["texture_px"] is None:
                    result.errors.append(f"variant texture assigned to untextured part {name}")
                else:
                    texture_sizes[texture] = part["texture_px"]
        for reference, size in texture_sizes.items():
            try:
                check_png(folder / reference, size)
                result.textures.add(size)
            except Exception as exc:  # noqa: BLE001 - image decoders have several exception classes.
                result.errors.append(f"{reference}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a malformed prop must not prevent later rows.
        result.errors.append(f"{type(exc).__name__}: {exc}")
    return result


def print_table(results):
    rows = [
        ["Prop", "Status", "BBox mm (X x Y x Z)", "Triangles / budget", "Collision", "Max verts", "Textures"]
    ]
    rows.extend(result.row() for result in results)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))
    for result in results:
        for error in result.errors:
            print(f"  {result.prop_id}: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props", nargs="+", help="Only these prop IDs (default: every prop.json)")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "assets" / "props",
        help="Prop directory, also useful for isolated validator tests",
    )
    args = parser.parse_args(argv)
    folders = (
        [args.root / prop_id for prop_id in args.props]
        if args.props
        else sorted(p.parent for p in args.root.glob("*/prop.json"))
    )
    if not folders:
        print("No prop.json files found.")
        return 1
    results = [validate_prop(folder) for folder in folders]
    print_table(results)
    passed = sum(not result.errors for result in results)
    print(f"\n{passed}/{len(results)} props passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
