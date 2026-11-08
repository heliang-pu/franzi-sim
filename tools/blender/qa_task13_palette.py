"""Prepare (do not install) Floyd-Steinberg 256-colour PNG candidates for QA."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from compact_props import QA
from prop_paths import PROPS, label


def candidate(path_string):
    path = Path(path_string)
    target = QA / "palette_candidates" / label(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        palette = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        quantized = rgb.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        delta = np.abs(np.asarray(rgb, dtype=float) - np.asarray(quantized.convert("RGB"), dtype=float))
        quantized.save(target, optimize=True, bits=8)
    return {"file": label(path), "before": path.stat().st_size,
            "candidate": target.stat().st_size, "saved": path.stat().st_size - target.stat().st_size,
            "max_rgb_error": float(delta.max()), "mean_rgb_error": float(delta.mean()),
            "rmse_rgb": float(np.sqrt((delta ** 2).mean()))}


if __name__ == "__main__":
    paths = [str(path) for path in sorted(PROPS.glob("*/*.png"))
             if path.parent.name != "factory_floor" and ("albedo" in path.name or path.name.startswith("preview"))]
    with ProcessPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(candidate, paths))
    (QA / "palette_candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Candidate savings (not installed):", sum(row["saved"] for row in rows))
