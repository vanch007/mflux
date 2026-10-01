import hashlib
import math

import mlx.core as mx
from PIL import Image


class QwenImage21LatentCreator:
    @staticmethod
    def edit_noise_seed(seed: int, images: list[Image.Image]) -> int:
        if not images:
            return seed
        # Reusing a generated reference's noise causes oversharpening in self-edits.
        digest = hashlib.sha256(f"mflux-qwen21-edit-noise-v1:{seed}".encode())
        for image in images:
            rgba = image.convert("RGBA")
            digest.update(rgba.width.to_bytes(4, "big"))
            digest.update(rgba.height.to_bytes(4, "big"))
            digest.update(rgba.tobytes())
        mixed = int.from_bytes(digest.digest()[:4], "big")
        return mixed if mixed != (seed & 0xFFFFFFFF) else mixed ^ 0xFFFFFFFF

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0]
        b, c, h, w = latents.shape
        return latents.transpose(0, 2, 3, 1).reshape(b, h * w, c)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents.reshape(latents.shape[0], height // 16, width // 16, -1).transpose(0, 3, 1, 2)[:, :, None]

    @staticmethod
    def dimensions(resolution: int, ratio: float) -> tuple[int, int]:
        width = math.sqrt(resolution * resolution * ratio)
        return max(32, round(width / 32) * 32), max(32, round(width / ratio / 32) * 32)

    @staticmethod
    def validate(width: int, height: int, steps: int, image_count: int) -> None:
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Qwen-Image-2.1 dimensions must be positive multiples of 32.")
        if steps < 2:
            raise ValueError("Qwen-Image-2.1 requires at least two steps for terminal sigma shifting.")
        if image_count > 10:
            raise ValueError("Qwen-Image-2.1 supports at most 10 reference images.")

    @staticmethod
    def validate_resolution(resolution: int) -> None:
        if resolution < 32 or resolution % 32:
            raise ValueError("output_resolution must be a positive multiple of 32.")
