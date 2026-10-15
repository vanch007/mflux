import math
import mlx.core as mx

IMAGE_ROPE_SCALE = 0.5
DOWNSAMPLING_RATIO = 16
LATENT_CHANNELS = 128


class Flux3LatentCreator:
    @staticmethod
    def latent_shape(height: int, width: int) -> tuple[int, int]:
        if height % DOWNSAMPLING_RATIO != 0 or width % DOWNSAMPLING_RATIO != 0:
            raise ValueError(f"width ({width}) and height ({height}) must be multiples of {DOWNSAMPLING_RATIO}")
        return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO

    @staticmethod
    def latent_ids(h: int, w: int) -> mx.array:
        rows, cols = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
        zeros = mx.zeros_like(rows)
        return mx.reshape(mx.stack([zeros, rows, cols, zeros], axis=-1), (h * w, 4)).astype(mx.float32)

    @staticmethod
    def scaled_ids(ids: mx.array) -> mx.array:
        # Scale row and col axes by 0.5
        ids_scaled = ids.astype(mx.float32)
        r = ids_scaled[:, 1:3] * IMAGE_ROPE_SCALE
        return mx.concatenate([ids_scaled[:, :1], r, ids_scaled[:, 3:]], axis=-1)

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> tuple[mx.array, mx.array, mx.array]:
        h, w = Flux3LatentCreator.latent_shape(height, width)
        key = mx.random.key(seed)
        noise = mx.random.normal((h * w, LATENT_CHANNELS), key=key, dtype=mx.float32)
        raw_ids = Flux3LatentCreator.latent_ids(h, w)
        scaled = Flux3LatentCreator.scaled_ids(raw_ids)
        return noise, raw_ids, scaled

