import math
import numpy as np
from PIL import Image, ImageOps

DOWNSAMPLING_RATIO = 16
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int, max_pixels: int = MATCH_MAX_PIXELS) -> tuple[tuple[int, int], tuple[int, int]]:
    scale = min(1.0, math.sqrt(max_pixels / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= max_pixels:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image, max_pixels: int = MATCH_MAX_PIXELS) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height, max_pixels=max_pixels)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    padded = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image, max_pixels: int = MULTI_REFERENCE_MAX_PIXELS) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))

