"""Compile every default and variant using the repository's real MuJoCo loader."""

import gc
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
from sim.prop_assets import add

records = []
for path in sorted((ROOT / "assets" / "props").glob("*/prop.json")):
    data = json.loads(path.read_text())
    for variant in [None] + [variant["name"] for variant in data["variants"]]:
        root = ET.Element("mujoco", model="task13_loader_check")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "asset")
        ET.SubElement(root, "worldbody")
        add(root, data["id"], "prop", variant_name=variant)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        records.append({"prop": data["id"], "variant": variant or "default", "status": "PASS",
                        "meshes": model.nmesh, "textures": model.ntex})
        del model
        gc.collect()
    print("LOADER_PASS", data["id"], flush=True)
(HERE / "_qa" / "task13" / "loader_checks.json").write_text(json.dumps(records, indent=2) + "\n")
print("LOADER_COMPLETE", len(records), "defaults/variants")
