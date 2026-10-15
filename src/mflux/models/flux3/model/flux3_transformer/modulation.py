import mlx.core as mx
from mlx import nn


def layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    u = mx.mean(x, axis=-1, keepdims=True)
    s = mx.var(x, axis=-1, keepdims=True)
    return (x - u) * mx.rsqrt(s + eps)


def layernorm_modulation(x: mx.array, scale: mx.array, shift: mx.array, eps: float = 1e-6) -> mx.array:
    return layer_norm(x, eps=eps) * (1.0 + scale) + shift


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        chunks = mx.split(self.lin(nn.silu(vec)), 3, axis=-1)
        return chunks[0], chunks[1], chunks[2]


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False)]

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        h = self.adaLN_modulation[0](vec)
        h = self.adaLN_modulation[1](h)
        chunks = mx.split(h, 2, axis=-1)
        shift, scale = chunks[0], chunks[1]
        return self.linear(layernorm_modulation(x, scale, shift))

