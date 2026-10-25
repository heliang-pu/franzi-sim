"""Pillow worker for prop_lib.make_label_png; JSON on stdin, no shell quoting.

Fields: text/title, lines, brand, tracking, barcode (bool or seed text), seed,
zone, zone_color, background, foreground, checker, checker_cells. A barcode
is decorative random bars, never a real shipping or product identifier.
"""

import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from PIL import Image, ImageDraw, ImageFont
from prop_schema import FICTIONAL_BRANDS


def validate_fields(fields):
    if not isinstance(fields, dict):
        raise TypeError("label fields must be a dictionary")
    if fields.get("brand") and fields["brand"] not in FICTIONAL_BRANDS:
        raise ValueError("Only the section 1.10 fictional brands are permitted")


def _font(size):
    for font in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font).is_file():
            return ImageFont.truetype(font, max(1, int(size)))
    return ImageFont.load_default(size=max(1, int(size)))


def draw_label(path, width, height, fields):
    validate_fields(fields)
    if width < 1 or height < 1:
        raise ValueError("label dimensions must be positive")
    bg, fg = fields.get("background", "#F5F3E9"), fields.get("foreground", "#15191E")
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    pad = max(2, int(min(width, height) * 0.035))
    if fields.get("checker"):
        cells = int(fields.get("checker_cells", 8))
        if not 1 <= cells <= 128:
            raise ValueError("checker_cells must be between 1 and 128")
        for row in range(cells):
            for col in range(cells):
                draw.rectangle(
                    (
                        col * width // cells,
                        row * height // cells,
                        (col + 1) * width // cells - 1,
                        (row + 1) * height // cells - 1,
                    ),
                    fill=fields.get("checker_dark", "#285575") if (row + col) % 2 else bg,
                )
    title = str(fields.get("text", fields.get("title", fields.get("brand", ""))))
    if title:
        target_width = width * (0.85 if not fields.get("zone") else 0.62)
        size = min(height * 0.26, width * 0.3)
        while size > 6 and draw.textbbox((0, 0), title, font=_font(size))[2] > target_width:
            size *= 0.92
        font = _font(size)
        left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
        x = (width - (right - left)) / 2 - left
        y = height * (0.48 if fields.get("checker") else 0.18) - (bottom - top) / 2 - top
        if fields.get("checker"):
            draw.rounded_rectangle((pad, y + top - pad, width - pad, y + bottom + pad), radius=pad, fill=bg)
        draw.text((x, y), title, fill=fg, font=font)
    if fields.get("zone"):
        zone_size = int(min(width * 0.23, height * 0.3))
        draw.rectangle((pad, pad, pad + zone_size, pad + zone_size), fill=fields.get("zone_color", "#D32F2F"))
        draw.text(
            (pad + zone_size / 2, pad + zone_size / 2),
            str(fields["zone"]),
            fill="white",
            font=_font(zone_size * 0.73),
            anchor="mm",
        )
    lines = fields.get("lines", [])
    if isinstance(lines, str):
        lines = lines.splitlines()
    if fields.get("tracking"):
        lines = [*lines, fields["tracking"]]
    line_size = min(height * 0.065, width * 0.04)
    for index, line in enumerate(lines):
        draw.text((pad, height * 0.4 + index * line_size * 1.35), str(line), fill=fg, font=_font(line_size))
    if fields.get("barcode"):
        rng = random.Random(str(fields.get("seed", fields.get("barcode"))))
        x, stop = pad, width - pad
        base = max(1, width // 240)
        while x < stop:
            bar_width = rng.randint(1, 3) * base
            draw.rectangle((x, int(height * 0.77), min(x + bar_width - 1, stop), height - pad), fill=fg)
            x += bar_width + rng.randint(1, 3) * base
    # Explicit sRGB rendering intent; Pillow writes an sRGB chunk.
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, pnginfo=info)


if __name__ == "__main__":
    if "--probe" in sys.argv:
        print("Pillow available")
    else:
        payload = json.load(sys.stdin)
        draw_label(payload["path"], payload["width"], payload["height"], payload["fields"])
