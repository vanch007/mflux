import math
import mlx.core as mx

AXES_DIM = (32, 32, 32, 32)
THETA = 10000


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10000) -> mx.array:
    t = 1000.0 * t
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(t.dtype)


def compute_rope_freqs(ids: mx.array, axes_dim: tuple[int, ...] = AXES_DIM, theta: int = THETA) -> tuple[mx.array, mx.array]:
    """ids: (B, L, 4) -> cos_all, sin_all of shape (B, 1, L, 64)"""
    cos_parts = []
    sin_parts = []
    for i, dim_i in enumerate(axes_dim):
        scale_i = mx.arange(0, dim_i, 2, dtype=mx.float32) / dim_i
        theta_scale = 1.0 / (theta ** scale_i)
        angles = ids[..., i : i + 1] * theta_scale
        cos_parts.append(mx.cos(angles))
        sin_parts.append(mx.sin(angles))
    cos_all = mx.expand_dims(mx.concatenate(cos_parts, axis=-1), axis=1)
    sin_all = mx.expand_dims(mx.concatenate(sin_parts, axis=-1), axis=1)
    return cos_all, sin_all


def rmsnorm_rope(x: mx.array, scale: mx.array, cos_all: mx.array, sin_all: mx.array, eps: float = 1e-6) -> mx.array:
    """x: (B, H, L, D), scale: (D,), cos_all/sin_all: (B, 1, L, D//2)"""
    orig_dtype = x.dtype
    x_f = x.astype(mx.float32)
    norm = x_f * mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + eps) * scale.astype(mx.float32)

    b, h, l, d = norm.shape
    x_pairs = mx.reshape(norm, (b, h, l, d // 2, 2))
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    rot0 = x0 * cos_all - x1 * sin_all
    rot1 = x0 * sin_all + x1 * cos_all
    rotated = mx.reshape(mx.stack([rot0, rot1], axis=-1), (b, h, l, d))
    return rotated.astype(orig_dtype)

