"""Blender-independent checks for benchmark_props.md section 1.8.

Extensions: budget_class is small/medium/large; a spring uses stiffness and
reference (N/m and m for slides, N m/rad and rad for hinges). Unknown metadata
is retained so task-specific placement, dimensions and behaviour can be added.
"""

import math
import re
from pathlib import PurePosixPath

TRIANGLE_BUDGETS = {"small": 10_000, "medium": 50_000, "large": 150_000}
SMALL_PROPS = {
    "_test_cube",
    "flange_part",
    "abs_housing",
    "panel_devices",
    "hex_blank",
    "flanged_bushing",
    "l_bracket",
    "spur_gear",
    "round_blank",
    "machined_part",
    "bearing_6204",
    "dowel_pin",
    "pin_fixture",
}
LARGE_PROPS = {
    "belt_conveyor",
    "roller_conveyor",
    "cnc_front",
    "scissor_lift_table",
    "workbench",
    "factory_floor",
    "safety_fence",
    "background_rack",
}
FICTIONAL_BRANDS = {"QF Express", "NOVA Logistics", "IMC Controls", "IMC-CNC 850", "NVK"}


def budget_class(data):
    """Use the prescribed tier for known props, or an explicit tier for new ones."""
    prop_id = data.get("id", "")
    if prop_id in SMALL_PROPS:
        return "small"
    if prop_id.startswith("bearing_housing_"):
        return "small"
    if prop_id in LARGE_PROPS:
        return "large"
    if "budget_class" in data:
        return data["budget_class"]
    return "medium"


def collision_budget(data):
    # The task-specific section 3.10 overrides section 1.6's 128-piece limit.
    return 150 if data["id"].startswith("bearing_housing_") else 64


def referenced_files(data):
    """Find references including variant textures/meshes and source/preview."""
    refs = set()

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "notes":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.lower().endswith((".obj", ".png", ".blend")):
            refs.add(value)

    visit(data)
    return sorted(refs)


def schema_errors(data):
    """Return all useful errors, without requiring bpy or jsonschema."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def vector(value, length):
        return isinstance(value, list) and len(value) == length and all(number(v) for v in value)

    def filename(value, suffix=None):
        if not isinstance(value, str) or not value or "\\" in value:
            return False
        parsed = PurePosixPath(value)
        shared_texture = (
            parsed.suffix.lower() == ".png"
            and len(parsed.parts) == 3
            and parsed.parts[0] == ".."
            and re.fullmatch(r"[A-Za-z0-9_-]+", parsed.parts[1])
            and parsed.parts[2] not in (".", "..")
        )
        return (
            not parsed.is_absolute()
            and (".." not in parsed.parts or shared_texture)
            and (suffix is None or parsed.suffix.lower() == suffix)
        )

    if not isinstance(data, dict):
        return ["prop.json must be an object"]
    for key in ("id", "task", "units", "origin", "bbox_mm", "triangles", "parts", "variants", "notes"):
        require(key in data, f"missing {key}")
    require(isinstance(data.get("id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", data["id"]), "invalid id")
    require(isinstance(data.get("task"), str) and bool(data["task"]), "task must be a nonempty string")
    require(data.get("units") == "m", "units must be m")
    require(
        isinstance(data.get("origin"), str) and bool(data["origin"]), "origin must describe the prop frame"
    )
    bbox = data.get("bbox_mm")
    require(vector(bbox, 3) and all(v > 0 for v in bbox), "bbox_mm must contain three positive dimensions")
    count = data.get("triangles")
    require(
        isinstance(count, int) and not isinstance(count, bool) and count > 0,
        "triangles must be a positive integer",
    )
    require(isinstance(data.get("notes"), str), "notes must be a string")
    if "budget_class" in data:
        require(data["budget_class"] in TRIANGLE_BUDGETS, "invalid budget_class")
    if "precision_mm" in data:
        require(number(data["precision_mm"]) and data["precision_mm"] > 0, "invalid precision_mm")
    for key, suffix in (("source", ".blend"), ("preview", ".png")):
        if key in data:
            require(filename(data[key], suffix), f"invalid {key} filename")

    parts = data.get("parts")
    if not isinstance(parts, list) or not parts:
        return errors + ["parts must be a nonempty list"]
    names = [p.get("name") for p in parts if isinstance(p, dict)]
    require(
        all(isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names), "invalid part name"
    )
    require(len({n for n in names if isinstance(n, str)}) == len(names), "duplicate part name")
    name_set = {n for n in names if isinstance(n, str)}
    parent_map = {}
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            errors.append(f"parts[{index}] must be an object")
            continue
        prefix = str(part.get("name", index))
        for key in (
            "name",
            "visual",
            "texture",
            "texture_px",
            "collision",
            "mass_kg",
            "friction",
            "mujoco_material",
            "parent",
            "joint",
        ):
            require(key in part, f"{prefix}: missing {key}")
        require(filename(part.get("visual"), ".obj"), f"{prefix}: invalid visual filename")
        texture = part.get("texture")
        if texture is None:
            require(
                vector(part.get("rgba"), 4) and all(0 <= c <= 1 for c in part["rgba"]),
                f"{prefix}: untextured parts need rgba",
            )
            require(part.get("texture_px") is None, f"{prefix}: untextured texture_px must be null")
        else:
            require(filename(texture, ".png"), f"{prefix}: invalid texture filename")
            require(
                type(part.get("texture_px")) is int and part["texture_px"] in (1024, 2048),
                f"{prefix}: texture_px must be 1024 or 2048",
            )
        collisions = part.get("collision")
        require(
            isinstance(collisions, list), f"{prefix}: collision must be a list (empty for background props)"
        )
        if isinstance(collisions, list):
            require(all(filename(f, ".obj") for f in collisions), f"{prefix}: invalid collision filename")
            require(
                len({f for f in collisions if isinstance(f, str)}) == len(collisions),
                f"{prefix}: duplicate collision reference",
            )
            if isinstance(data.get("id"), str):
                require(len(collisions) <= collision_budget(data), f"{prefix}: too many collision pieces")
        # Environment floors use MuJoCo's analytic plane, not a degenerate
        # zero-volume mesh or an artificial thin collision box. Mesh collision
        # references retain all the existing solid/convex validation rules.
        primitives = part.get("collision_primitives", [])
        require(isinstance(primitives, list), f"{prefix}: collision_primitives must be a list")
        if isinstance(primitives, list):
            if isinstance(collisions, list) and isinstance(data.get("id"), str):
                require(len(collisions) + len(primitives) <= collision_budget(data),
                        f"{prefix}: too many collision pieces including primitives")
            for primitive in primitives:
                if not isinstance(primitive, dict):
                    errors.append(f"{prefix}: collision primitive must be an object")
                    continue
                require(primitive.get("type") == "plane", f"{prefix}: unsupported collision primitive")
                require(data.get("static") is True and part.get("joint") is None
                        and part.get("parent") is None,
                        f"{prefix}: analytic planes require a static root part")
                require(vector(primitive.get("pos_m"), 3), f"{prefix}: plane needs finite pos_m")
                normal = primitive.get("normal")
                require(vector(normal, 3) and abs(sum(v * v for v in normal) - 1) <= 1e-5,
                        f"{prefix}: plane normal must be a unit vector")
                size = primitive.get("size_m")
                require(vector(size, 3) and all(v > 0 for v in size),
                        f"{prefix}: plane size_m must contain three positive dimensions")
        require(number(part.get("mass_kg")) and part["mass_kg"] > 0, f"{prefix}: mass_kg must be positive")
        require(
            number(part.get("friction")) and part["friction"] >= 0, f"{prefix}: friction must be nonnegative"
        )
        material = part.get("mujoco_material")
        require(isinstance(material, dict), f"{prefix}: missing MuJoCo material")
        if isinstance(material, dict):
            for key in ("specular", "shininess", "reflectance"):
                require(
                    number(material.get(key)) and 0 <= material[key] <= 1, f"{prefix}: invalid material {key}"
                )
        if "emission" in part:
            require(number(part["emission"]) and 0 <= part["emission"] <= 1, f"{prefix}: invalid emission")
        parent = part.get("parent")
        require(
            parent is None or isinstance(parent, str) and parent in name_set and parent != part.get("name"),
            f"{prefix}: parent must name another part or be null",
        )
        if isinstance(part.get("name"), str):
            parent_map[part["name"]] = parent
        joint = part.get("joint")
        if joint is None:
            continue
        if not isinstance(joint, dict):
            errors.append(f"{prefix}: joint must be an object or null")
            continue
        require(parent is not None, f"{prefix}: joint needs a parent part")
        require(joint.get("type") in ("hinge", "slide"), f"{prefix}: invalid joint type")
        if "parent" in joint:
            require(joint["parent"] == parent, f"{prefix}: joint.parent disagrees with part.parent")
        axis = joint.get("axis")
        require(
            vector(axis, 3) and abs(sum(v * v for v in axis) - 1) <= 1e-5,
            f"{prefix}: axis must be a unit vector",
        )
        require(vector(joint.get("anchor_m"), 3), f"{prefix}: missing/invalid joint anchor_m")
        limits = joint.get("range")
        require(
            vector(limits, 2) and limits[0] < limits[1] and limits[0] <= 0 <= limits[1],
            f"{prefix}: range must be ordered and include initial zero",
        )
        for key in ("damping", "frictionloss"):
            require(number(joint.get(key)) and joint[key] >= 0, f"{prefix}: missing/invalid joint {key}")
        if "spring" in joint:
            spring = joint["spring"]
            require(
                isinstance(spring, dict)
                and number(spring.get("stiffness"))
                and spring["stiffness"] >= 0
                and number(spring.get("reference")),
                f"{prefix}: spring needs nonnegative stiffness and reference",
            )
    for name in parent_map:
        seen = set()
        current = name
        while isinstance(current, str) and current in parent_map:
            if current in seen:
                errors.append(f"{name}: cyclic parent chain")
                break
            seen.add(current)
            current = parent_map[current]

    variants = data.get("variants")
    require(isinstance(variants, list), "variants must be a list")
    if isinstance(variants, list):
        variant_names = set()
        for variant in variants:
            if not isinstance(variant, dict) or not isinstance(variant.get("name"), str):
                errors.append("each variant needs a name")
                continue
            require(variant["name"] not in variant_names, "duplicate variant name")
            variant_names.add(variant["name"])
            textures = variant.get("textures", {})
            require(isinstance(textures, dict), "variant textures must map part names to PNG files")
            if isinstance(textures, dict):
                for name, texture in textures.items():
                    require(
                        name in name_set and filename(texture, ".png"), f"invalid variant texture for {name}"
                    )
    for reference in referenced_files(data):
        require(filename(reference), f"unsafe file reference: {reference}")
    return errors


def check_schema(data):
    errors = schema_errors(data)
    if errors:
        raise ValueError("Invalid prop.json:\n  " + "\n  ".join(errors))
