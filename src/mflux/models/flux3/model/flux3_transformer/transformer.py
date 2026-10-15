from dataclasses import dataclass
import mlx.core as mx
from mlx import nn

from mflux.models.flux3.model.flux3_transformer.attention import Block
from mflux.models.flux3.model.flux3_transformer.modulation import LastLayer, MLPEmbedder, ModeModulation
from mflux.models.flux3.model.flux3_transformer.pos_embed import (
    AXES_DIM,
    THETA,
    compute_rope_freqs,
    timestep_embedding,
)

MODES = ("image", "image_cond", "text")
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
HIDDEN = 3072
HEADS = 24
DEPTH = 5
DEPTH_SINGLE = 28
MLP_RATIO = 3.0


class ModuleContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class Packed:
    data: dict[str, mx.array]
    ids: dict[str, mx.array]
    timesteps: dict[str, mx.array]
    mode_indices: dict[str, list[tuple[int, int]]]

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


class Flux3Transformer(nn.Module):
    def __init__(
        self,
        hidden: int = HIDDEN,
        heads: int = HEADS,
        depth: int = DEPTH,
        depth_single: int = DEPTH_SINGLE,
        mlp_ratio: float = MLP_RATIO,
        flow_map: bool = True,
    ):
        super().__init__()
        self.flow_map = flow_map
        self.hidden = hidden
        self.heads = heads
        self.depth = depth
        self.depth_single = depth_single

        self.emb_in = ModuleContainer(
            image=nn.Linear(IN_CHANNELS["image"], hidden, bias=False),
            image_cond=nn.Linear(IN_CHANNELS["image_cond"], hidden, bias=False),
        )
        self.txt_in = nn.Linear(IN_CHANNELS["text"], hidden, bias=False)
        self.time_in = MLPEmbedder(256, hidden)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, hidden)

        self.early_stream_modulations = ModuleContainer(
            image=ModeModulation(hidden),
            image_cond=ModeModulation(hidden),
            txt=ModeModulation(hidden),
        )
        self.single_stream_modulations = ModuleContainer(
            image=ModeModulation(hidden),
            image_cond=ModeModulation(hidden),
            txt=ModeModulation(hidden),
        )
        self.content_mode_blocks = ModuleContainer(
            image=[Block(hidden, heads, mlp_ratio) for _ in range(depth)],
            image_cond=[Block(hidden, heads, mlp_ratio) for _ in range(depth)],
        )
        self.txt_mode_blocks = [Block(hidden, heads, mlp_ratio) for _ in range(depth)]
        self.single_blocks = [Block(hidden, heads, mlp_ratio) for _ in range(depth_single)]

        last_layer = LastLayer(hidden, IN_CHANNELS["image"])
        self.final_layer = ModuleContainer(image=last_layer)

    def __call__(self, packed: Packed, timesteps_ref: mx.array | None = None) -> mx.array:
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a flow-map DiT")

        data = dict(packed.data)
        ids = dict(packed.ids)
        timesteps = dict(packed.timesteps)

        for m in MODES:
            if data[m].shape[0] == 0:
                data[m] = mx.zeros((1, data[m].shape[1]), dtype=data[m].dtype)
                ids[m] = mx.zeros((1, 4), dtype=ids[m].dtype)
                timesteps[m] = mx.zeros((1,), dtype=timesteps[m].dtype)

        mod_key = lambda m: "txt" if m == "text" else m

        vec = {m: self.time_in(timestep_embedding(timesteps[m])) for m in MODES}
        if timesteps_ref is not None:
            ref = self.ref_time_in(timestep_embedding(timesteps_ref.astype(timesteps["image"].dtype)))
            for m in MODES:
                parts = []
                for s, (a, b) in enumerate(packed.mode_indices[m]):
                    if b > a:
                        parts.append(mx.broadcast_to(ref[s : s + 1], (b - a, ref.shape[-1])))
                if parts:
                    vec[m] = vec[m] + mx.concatenate(parts, axis=0)

        early = {
            m: getattr(self.early_stream_modulations, mod_key(m))(vec[m])
            for m in MODES
        }
        single = {
            m: getattr(self.single_stream_modulations, mod_key(m))(vec[m])
            for m in MODES
        }
        x = {
            m: (self.txt_in if m == "text" else getattr(self.emb_in, m))(data[m])
            for m in MODES
        }

        freqs = {m: compute_rope_freqs(ids[m][None]) for m in MODES}

        for d in range(self.depth):
            for m in ("image", "image_cond"):
                block = getattr(self.content_mode_blocks, m)[d]
                x[m] = block(x[m], freqs[m][0], freqs[m][1], packed.mode_indices[m], early[m])
            x["text"] = self.txt_mode_blocks[d](x["text"], freqs["text"][0], freqs["text"][1], packed.mode_indices["text"], early["text"])

        chunks = []
        id_chunks = []
        mods = ([], [], [])
        bounds = []
        placement = []
        offset = 0

        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))

        h = mx.concatenate(chunks, axis=0)
        all_ids = mx.concatenate(id_chunks, axis=0)[None]
        cos_sm, sin_sm = compute_rope_freqs(all_ids)
        combined_mods = (
            mx.concatenate(mods[0], axis=0),
            mx.concatenate(mods[1], axis=0),
            mx.concatenate(mods[2], axis=0),
        )

        for block in self.single_blocks:
            h = block(h, cos_sm, sin_sm, bounds, combined_mods)

        image_chunks = []
        for m, a, b, start in placement:
            if m == "image":
                image_chunks.append(h[start : start + b - a])
        image = mx.concatenate(image_chunks, axis=0) if image_chunks else mx.zeros_like(x["image"])

        return self.final_layer.image(image, vec["image"])

