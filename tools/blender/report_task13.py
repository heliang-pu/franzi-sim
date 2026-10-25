"""Write the size comparison and complete validator table for Task 13."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from compact_props import ASSETS, HERE, QA, inventory
from validate_props import validate_prop


def table(rows):
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    result = []
    for index, row in enumerate(rows):
        result.append(" | ".join(str(value).ljust(width) for value, width in zip(row, widths, strict=True)))
        if index == 0:
            result.append("-+-".join("-" * width for width in widths))
    return "\n".join(result)


before = json.loads((QA / "inventory_before.json").read_text())
after = inventory()
audit = json.loads((QA / "verification.json").read_text())["audit"]
loader = json.loads((QA / "loader_checks.json").read_text())
results = [validate_prop(path.parent) for path in sorted(ASSETS.glob("*/prop.json"))]
assert not any(result.errors for result in results)
size_rows = [["Prop", "Before MB", "After MB", "Validator"]]
details = []
for result in results:
    name = result.prop_id
    old = before["props"][name]
    new = after["props"][name]
    size_rows.append([name, f"{old['without_source'] / 1e6:.3f}", f"{new['without_source'] / 1e6:.3f}", "PASS"])
    details.append({"prop": name, "before": old, "after": new, "validator": "PASS"})
categories = [["Category", "Before MB", "After MB"]]
for key in ("textures", "obj", "previews", "source.blend", "blend1", "other"):
    categories.append([key, f"{before['categories'][key] / 1e6:.3f}", f"{after['categories'][key] / 1e6:.3f}"])
old_public = sum(before["categories"].values()) - before["categories"]["source.blend"]
new_public = sum(after["categories"].values()) - after["categories"]["source.blend"]
assert new_public <= 90_000_000
summary = f"Public asset size (source.blend excluded): {old_public:,} -> {new_public:,} bytes; {100 * (1 - new_public / old_public):.1f}% smaller. Target <= 90,000,000 bytes: PASS."
text = "\n\n".join([
    "Task 13: public prop compaction",
    summary,
    "MB means 1,000,000 bytes. Per-prop sizes below exclude source.blend, include all other files, and count shared textures once in their owning folder.",
    table(size_rows),
    table(categories),
    f"All files including source.blend: {sum(before['categories'].values()):,} -> {sum(after['categories'].values()):,} bytes.",
    f"Validation: {len(results)}/{len(results)} props PASS; {audit['references_checked']} paths resolve; {audit['source_blends_unchanged']} source.blend SHA-256 hashes unchanged. The real repository MuJoCo loader compiled all {len(loader)} defaults/variants, including indexed PNGs and relative OBJ/texture indices.",
    "Geometry: all 1,505 OBJ files processed. Position movement is bounded by 0.005 mm in Euclidean distance; face order and position indices are preserved. Normals and UVs use four decimal places. Equal rounded UVs share records; visual files use shorter equivalent OBJ relative indices. Collision files keep their original topology/indices and strict convexity; there was no collision remeshing or simplification.",
    "Required exception: fixed five-decimal metre positions would collapse fine triangles, break strict collision convexity and exceed the Euclidean movement bound. Necessary coordinates retain 6-12 decimals; each affected prop.json notes this. The factory floor repeat tile and source artwork retain their original dimensions and RGB pixels.",
    "Images: 81 baked albedos downsampled using Lanczos; 38 byte-identical atlas copies removed; all 86 previews are 512 square. Only 69 reviewed albedos and 84 reviewed previews use 256-colour Floyd-Steinberg palettes. The two floor previews remain RGB. Parcel/carton labels, legend plates, zone signs, CNC screen/panel, stacking-plan sheets and station-plate print atlases remain RGB.",
    "Visual QA: 69 before/after actual OBJ/UV close-render pairs at 640x480 showed no visible banding; 84 preview comparisons were also reviewed. Parcel-label and CNC-screen renders at 0.5 m remain readable. Review sheets and metrics are under tools/blender/_qa/task13/.",
    "Rebuild: all twelve build_task entry points call the shared finalizer; source saving has save_version=0. A real isolated Task 0 rebuild exercised 2048 Cycles baking, Lanczos output at 1024, 512 previews, approved palette output and repeated source saving without backups. Its validator passes. All 45 production props were validated without rebaking their preserved sources.",
    "Reproduce: /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -B tools/blender/compact_props.py --apply\nThen: /Users/puheliang/miniforge3/envs/inovance-fineart/bin/python -B tools/blender/validate_props.py",
    "Full validator table:\n" + (HERE / "task13_validation.txt").read_text().strip(),
    "Unfinished required work: none. Precision-format exceptions are recorded above and in the individual prop notes.",
]) + "\n"
(HERE / "task13_report.txt").write_text(text)
(QA / "size_comparison.json").write_text(json.dumps({"props": details, "before": before["categories"],
                                                     "after": after["categories"], "public_bytes": new_public}, indent=2) + "\n")
print(summary)
print(table(categories))
