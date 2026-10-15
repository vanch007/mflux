import mlx.core as mx

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class Flux3VAE(Flux2VAE):
    """FLUX.3 image autoencoder with 128-channel packed token interface."""

    def encode_to_tokens(self, image: mx.array) -> tuple[mx.array, mx.array]:
        """image: (1, 3, H, W) or (3, H, W) in [-1, 1] -> tokens: (h*w, 128), ids: (h*w, 4)"""
        if image.ndim == 3:
            image = mx.expand_dims(image, axis=0)
        # encoder output: (1, 32, H/8, W/8)
        mean = self.encode(image)
        b, c, h, w = mean.shape
        h_p, w_p = h // 2, w // 2
        z = mx.reshape(mean, (b, c, h_p, 2, w_p, 2))
        z = mx.transpose(z, (0, 1, 3, 5, 2, 4))
        z = mx.reshape(z, (b, c * 4, h_p, w_p))

        bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self.bn.running_var.reshape(1, -1, 1, 1) + self.bn.eps)
        z = (z - bn_mean) / bn_std

        tokens = mx.reshape(mx.transpose(z[0], (1, 2, 0)), (h_p * w_p, 128))

        rows, cols = mx.meshgrid(mx.arange(h_p), mx.arange(w_p), indexing="ij")
        zeros = mx.zeros_like(rows)
        ids = mx.reshape(mx.stack([zeros, rows, cols, zeros], axis=-1), (h_p * w_p, 4)).astype(mx.float32)
        return tokens, ids

    def decode_tokens(self, tokens: mx.array, ids: mx.array, tiling_config: TilingConfig | None = None) -> mx.array:
        """tokens: (h*w, 128) -> image: (3, H, W) in [-1, 1]"""
        h = int(mx.max(ids[:, 1]).item()) + 1
        w = int(mx.max(ids[:, 2]).item()) + 1
        packed = mx.transpose(mx.reshape(tokens, (h, w, 128)), (2, 0, 1))
        packed = mx.expand_dims(packed, axis=0)
        return self.decode_packed_latents(packed, tiling_config=tiling_config)[0]

