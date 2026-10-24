"""H.264 video through an ffmpeg pipe, so no engine-specific encoder is needed."""

import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
)


def font(size):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Video:
    def __init__(self, path, width, height, fps=25):
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if not shutil.which(ffmpeg):
            raise FileNotFoundError("--record needs ffmpeg on PATH")
        self.size = (width, height)
        self.frames = 0
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        assert rgb.shape == (self.size[1], self.size[0], 3), rgb.shape
        self.process.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        return self.process.wait()


class SessionVideo:
    """Display camera plus the four SDK camera streams, with a status banner."""

    def __init__(self, path, backend_label, cameras, size=(1280, 720), fps=25):
        self.video = Video(path, *size, fps=fps)
        self.label = backend_label
        self.cameras = cameras
        self.size = size
        self.font = font(20)
        self.small = font(15)

    def frame(self, overview, insets, lines):
        w, h = self.size
        canvas = Image.new("RGB", (w, h), (20, 27, 38))
        canvas.paste(Image.fromarray(overview).resize((960, 720)), (0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, name in enumerate(self.cameras):
            image = insets.get(name)
            y = 8 + i * 178
            if image is not None:
                canvas.paste(Image.fromarray(image).resize((213, 160)), (1000, y))
            draw.text((1004, y + 160), name, font=self.small, fill="white")
        draw.rectangle((0, 0, 960, 62), fill=(20, 27, 38))
        draw.text((14, 6), self.label, font=self.font, fill="white")
        draw.text((14, 34), lines, font=self.small, fill=(128, 218, 233))
        self.video.write(np.asarray(canvas))

    def close(self):
        return self.video.close()
