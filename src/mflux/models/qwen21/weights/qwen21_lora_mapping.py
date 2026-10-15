import mlx.core as mx

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class Qwen21LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping(*, reference: bool = True) -> list[LoRATarget]:
        paths = [
            "img_in",
            "txt_in.in_layer",
            "txt_in.out_layer",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "modulation.1",
            "norm_out.linear",
            "proj_out",
            *[
                f"transformer_blocks.{{block}}.{suffix}"
                for suffix in (
                    "attn.to_q",
                    "attn.to_k",
                    "attn.to_v",
                    "attn.to_out.0",
                    "img_mlp.proj",
                    "img_mlp.gate_layer",
                    "img_mlp.out",
                )
            ],
        ]
        targets = []
        for path in paths:
            aliases = [f"{prefix}{path}" for prefix in ("", "transformer.", "diffusion_model.", "base_model.model.")]
            kohya = f"lora_unet_{path.replace('.', '_')}"
            targets.append(
                LoRATarget(
                    model_path="modulation.layers.1" if path == "modulation.1" else path,
                    possible_up_patterns=[
                        f"{alias}.{suffix}"
                        for alias in aliases
                        for suffix in ("lora_B.weight", "lora_B.default.weight", "lora_up.weight", "lora.up.weight")
                    ]
                    + [f"{kohya}.lora_up.weight"],
                    possible_down_patterns=[
                        f"{alias}.{suffix}"
                        for alias in aliases
                        for suffix in ("lora_A.weight", "lora_A.default.weight", "lora_down.weight", "lora.down.weight")
                    ]
                    + [f"{kohya}.lora_down.weight"],
                    possible_alpha_patterns=[f"{alias}.alpha" for alias in [*aliases, kohya]],
                )
            )
        for name, transform in (
            ("gate_layer", Qwen21LoRAMapping._gate_up_gate),
            ("proj", Qwen21LoRAMapping._gate_up_proj),
        ):
            root = "transformer_blocks.{block}.img_mlp.gate_up"
            aliases = [f"{prefix}{root}" for prefix in ("", "transformer.", "diffusion_model.", "base_model.model.")]
            targets.append(
                LoRATarget(
                    model_path=f"transformer_blocks.{{block}}.img_mlp.{name}",
                    possible_up_patterns=[
                        f"{alias}.{suffix}"
                        for alias in aliases
                        for suffix in ("lora_B.weight", "lora_B.default.weight")
                    ],
                    possible_down_patterns=[
                        f"{alias}.{suffix}"
                        for alias in aliases
                        for suffix in ("lora_A.weight", "lora_A.default.weight")
                    ],
                    possible_alpha_patterns=[f"{alias}.alpha" for alias in aliases],
                    up_transform=transform,
                )
            )
        return targets

    @staticmethod
    def _gate_up_gate(weight: mx.array) -> mx.array:
        return mx.split(weight, 2, axis=0)[0]

    @staticmethod
    def _gate_up_proj(weight: mx.array) -> mx.array:
        return mx.split(weight, 2, axis=0)[1]
