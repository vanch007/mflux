import math
import mlx.core as mx
from mlx import nn

from mflux.models.flux3.model.flux3_transformer.modulation import layernorm_modulation
from mflux.models.flux3.model.flux3_transformer.pos_embed import rmsnorm_rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = mx.ones((dim,), dtype=mx.float32)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(self.head_dim)

    def __call__(
        self,
        x: mx.array,
        cos_all: mx.array,
        sin_all: mx.array,
        bounds: list[tuple[int, int]],
        mod: tuple[mx.array, mx.array, mx.array],
    ) -> mx.array:
        shift, scale, gate = mod
        x_mod = layernorm_modulation(x, scale, shift)

        length = x.shape[0]
        # (L, hidden) -> (1, L, heads, head_dim) -> (1, heads, L, head_dim)
        q = mx.transpose(mx.reshape(self.q_proj(x_mod), (1, length, self.heads, self.head_dim)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(self.k_proj(x_mod), (1, length, self.heads, self.head_dim)), (0, 2, 1, 3))
        v = mx.transpose(mx.reshape(self.v_proj(x_mod), (1, length, self.heads, self.head_dim)), (0, 2, 1, 3))

        q = rmsnorm_rope(q, self.norm.query_norm.scale, cos_all, sin_all)
        k = rmsnorm_rope(k, self.norm.key_norm.scale, cos_all, sin_all)

        scale_val = 1.0 / math.sqrt(self.head_dim)
        if len(bounds) == 1:
            start, end = bounds[0]
            attn = mx.fast.scaled_dot_product_attention(
                q[:, :, start:end, :],
                k[:, :, start:end, :],
                v[:, :, start:end, :],
                scale=scale_val,
            )
            # (1, heads, L_s, head_dim) -> (L_s, heads, head_dim) -> (L_s, hidden)
            attn_flat = mx.reshape(mx.transpose(attn[0], (1, 0, 2)), (end - start, self.heads * self.head_dim))
            attn_out = attn_flat
        else:
            chunks = []
            for start, end in bounds:
                if start == end:
                    continue
                attn_s = mx.fast.scaled_dot_product_attention(
                    q[:, :, start:end, :],
                    k[:, :, start:end, :],
                    v[:, :, start:end, :],
                    scale=scale_val,
                )
                chunks.append(mx.reshape(mx.transpose(attn_s[0], (1, 0, 2)), (end - start, self.heads * self.head_dim)))
            attn_out = mx.concatenate(chunks, axis=0) if chunks else mx.zeros((length, self.heads * self.head_dim), dtype=q.dtype)

        chunks_mlp = mx.split(self.mlp_in(x_mod), 2, axis=-1)
        gate_in, up = chunks_mlp[0], chunks_mlp[1]
        mlp_act = nn.silu(gate_in) * up

        return x + gate * (self.attn_out(attn_out) + self.mlp_out(mlp_act))

