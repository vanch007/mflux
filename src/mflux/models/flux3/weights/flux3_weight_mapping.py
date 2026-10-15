import re
from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms


def map_bfl_ae_to_vae(k: str) -> str:
    if k.startswith("bn."):
        return k
    if k.startswith("encoder.quant_conv."):
        return k.replace("encoder.quant_conv.", "quant_conv.")
    if k.startswith("decoder.post_quant_conv."):
        return k.replace("decoder.post_quant_conv.", "post_quant_conv.")

    k = k.replace(".norm_out.", ".conv_norm_out.")
    k = k.replace(".nin_shortcut.", ".conv_shortcut.")

    k = k.replace("encoder.mid.block_1.", "encoder.mid_block.resnets.0.")
    k = k.replace("encoder.mid.block_2.", "encoder.mid_block.resnets.1.")
    k = k.replace("encoder.mid.attn_1.norm.", "encoder.mid_block.attentions.0.group_norm.")
    k = k.replace("encoder.mid.attn_1.q.", "encoder.mid_block.attentions.0.to_q.")
    k = k.replace("encoder.mid.attn_1.k.", "encoder.mid_block.attentions.0.to_k.")
    k = k.replace("encoder.mid.attn_1.v.", "encoder.mid_block.attentions.0.to_v.")
    k = k.replace("encoder.mid.attn_1.proj_out.", "encoder.mid_block.attentions.0.to_out.")

    k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
    k = k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
    k = k.replace("decoder.mid.attn_1.norm.", "decoder.mid_block.attentions.0.group_norm.")
    k = k.replace("decoder.mid.attn_1.q.", "decoder.mid_block.attentions.0.to_q.")
    k = k.replace("decoder.mid.attn_1.k.", "decoder.mid_block.attentions.0.to_k.")
    k = k.replace("decoder.mid.attn_1.v.", "decoder.mid_block.attentions.0.to_v.")
    k = k.replace("decoder.mid.attn_1.proj_out.", "decoder.mid_block.attentions.0.to_out.")

    k = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", k)
    k = re.sub(r"encoder\.down\.(\d+)\.downsample\.conv\.", r"encoder.down_blocks.\1.downsamplers.0.conv.", k)

    m_block = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.*)", k)
    if m_block:
        lvl, blk, rest = m_block.groups()
        diffusers_lvl = 3 - int(lvl)
        return f"decoder.up_blocks.{diffusers_lvl}.resnets.{blk}.{rest}"

    m_up = re.match(r"decoder\.up\.(\d+)\.upsample\.conv\.(.*)", k)
    if m_up:
        lvl, rest = m_up.groups()
        diffusers_lvl = 3 - int(lvl)
        return f"decoder.up_blocks.{diffusers_lvl}.upsamplers.0.conv.{rest}"

    return k


class Flux3WeightMapping(WeightMapping):
    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        targets = [
            WeightTarget(to_pattern="emb_in.image.weight", from_pattern=["emb_in.image.weight"]),
            WeightTarget(to_pattern="emb_in.image_cond.weight", from_pattern=["emb_in.image_cond.weight"]),
            WeightTarget(to_pattern="txt_in.weight", from_pattern=["txt_in.weight"]),
            WeightTarget(to_pattern="time_in.in_layer.weight", from_pattern=["time_in.in_layer.weight"]),
            WeightTarget(to_pattern="time_in.out_layer.weight", from_pattern=["time_in.out_layer.weight"]),
            WeightTarget(to_pattern="ref_time_in.in_layer.weight", from_pattern=["ref_time_in.in_layer.weight"]),
            WeightTarget(to_pattern="ref_time_in.out_layer.weight", from_pattern=["ref_time_in.out_layer.weight"]),
            WeightTarget(to_pattern="final_layer.image.linear.weight", from_pattern=["final_layer.image.linear.weight"]),
            WeightTarget(to_pattern="final_layer.image.adaLN_modulation.1.weight", from_pattern=["final_layer.image.adaLN_modulation.1.weight"]),
        ]

        for mod in ("image", "image_cond", "txt"):
            targets.append(
                WeightTarget(
                    to_pattern=f"early_stream_modulations.{mod}.lin.weight",
                    from_pattern=[f"early_stream_modulations.{mod}.lin.weight"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"single_stream_modulations.{mod}.lin.weight",
                    from_pattern=[f"single_stream_modulations.{mod}.lin.weight"],
                )
            )

        for block in range(5):
            for mode in ("image", "image_cond"):
                pfx = f"content_mode_blocks.{mode}.{block}"
                for part in ("attn_out", "k_proj", "q_proj", "v_proj", "mlp_in", "mlp_out"):
                    targets.append(WeightTarget(to_pattern=f"{pfx}.{part}.weight", from_pattern=[f"{pfx}.{part}.weight"]))
                targets.append(WeightTarget(to_pattern=f"{pfx}.norm.key_norm.scale", from_pattern=[f"{pfx}.norm.key_norm.scale"]))
                targets.append(WeightTarget(to_pattern=f"{pfx}.norm.query_norm.scale", from_pattern=[f"{pfx}.norm.query_norm.scale"]))

            pfx_txt = f"txt_mode_blocks.{block}"
            for part in ("attn_out", "k_proj", "q_proj", "v_proj", "mlp_in", "mlp_out"):
                targets.append(WeightTarget(to_pattern=f"{pfx_txt}.{part}.weight", from_pattern=[f"{pfx_txt}.{part}.weight"]))
            targets.append(WeightTarget(to_pattern=f"{pfx_txt}.norm.key_norm.scale", from_pattern=[f"{pfx_txt}.norm.key_norm.scale"]))
            targets.append(WeightTarget(to_pattern=f"{pfx_txt}.norm.query_norm.scale", from_pattern=[f"{pfx_txt}.norm.query_norm.scale"]))

        for block in range(28):
            pfx_single = f"single_blocks.{block}"
            for part in ("attn_out", "k_proj", "q_proj", "v_proj", "mlp_in", "mlp_out"):
                targets.append(WeightTarget(to_pattern=f"{pfx_single}.{part}.weight", from_pattern=[f"{pfx_single}.{part}.weight"]))
            targets.append(WeightTarget(to_pattern=f"{pfx_single}.norm.key_norm.scale", from_pattern=[f"{pfx_single}.norm.key_norm.scale"]))
            targets.append(WeightTarget(to_pattern=f"{pfx_single}.norm.query_norm.scale", from_pattern=[f"{pfx_single}.norm.query_norm.scale"]))

        return targets

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mlx import nn

        vae = Flux2VAE()
        vae_params = dict(nn.utils.tree_flatten(vae.parameters()))
        targets = []

        for vae_k, p in vae_params.items():
            is_conv = len(p.shape) == 4 and vae_k.endswith(".weight")
            transform = WeightTransforms.transpose_conv2d_weight if is_conv else None
            
            # Map back to BFL key
            bfl_k = vae_k
            if bfl_k.startswith("quant_conv."):
                bfl_k = bfl_k.replace("quant_conv.", "encoder.quant_conv.")
            elif bfl_k.startswith("post_quant_conv."):
                bfl_k = bfl_k.replace("post_quant_conv.", "decoder.post_quant_conv.")
            bfl_k = bfl_k.replace(".conv_norm_out.", ".norm_out.")
            bfl_k = bfl_k.replace(".conv_shortcut.", ".nin_shortcut.")

            bfl_k = bfl_k.replace("encoder.mid_block.resnets.0.", "encoder.mid.block_1.")
            bfl_k = bfl_k.replace("encoder.mid_block.resnets.1.", "encoder.mid.block_2.")
            bfl_k = bfl_k.replace("encoder.mid_block.attentions.0.group_norm.", "encoder.mid.attn_1.norm.")
            bfl_k = bfl_k.replace("encoder.mid_block.attentions.0.to_q.", "encoder.mid.attn_1.q.")
            bfl_k = bfl_k.replace("encoder.mid_block.attentions.0.to_k.", "encoder.mid.attn_1.k.")
            bfl_k = bfl_k.replace("encoder.mid_block.attentions.0.to_v.", "encoder.mid.attn_1.v.")
            bfl_k = bfl_k.replace("encoder.mid_block.attentions.0.to_out.", "encoder.mid.attn_1.proj_out.")

            bfl_k = bfl_k.replace("decoder.mid_block.resnets.0.", "decoder.mid.block_1.")
            bfl_k = bfl_k.replace("decoder.mid_block.resnets.1.", "decoder.mid.block_2.")
            bfl_k = bfl_k.replace("decoder.mid_block.attentions.0.group_norm.", "decoder.mid.attn_1.norm.")
            bfl_k = bfl_k.replace("decoder.mid_block.attentions.0.to_q.", "decoder.mid.attn_1.q.")
            bfl_k = bfl_k.replace("decoder.mid_block.attentions.0.to_k.", "decoder.mid.attn_1.k.")
            bfl_k = bfl_k.replace("decoder.mid_block.attentions.0.to_v.", "decoder.mid.attn_1.v.")
            bfl_k = bfl_k.replace("decoder.mid_block.attentions.0.to_out.", "decoder.mid.attn_1.proj_out.")

            bfl_k = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", bfl_k)
            bfl_k = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.", r"encoder.down.\1.downsample.conv.", bfl_k)

            m_res = re.match(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", bfl_k)
            if m_res:
                lvl, blk, rest = m_res.groups()
                bfl_lvl = 3 - int(lvl)
                bfl_k = f"decoder.up.{bfl_lvl}.block.{blk}.{rest}"
            m_ups = re.match(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.*)", bfl_k)
            if m_ups:
                lvl, rest = m_ups.groups()
                bfl_lvl = 3 - int(lvl)
                bfl_k = f"decoder.up.{bfl_lvl}.upsample.conv.{rest}"

            from_list = [bfl_k]
            if vae_k != bfl_k:
                from_list.append(vae_k)

            targets.append(
                WeightTarget(
                    to_pattern=vae_k,
                    from_pattern=from_list,
                    transform=transform,
                )
            )

        return targets

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        targets = [
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=[
                    "model.language_model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                ],
            ),
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=[
                    "model.language_model.norm.weight",
                    "model.norm.weight",
                ],
            ),
        ]
        for layer in range(36):
            targets.extend([
                WeightTarget(
                    to_pattern=f"layers.{layer}.input_layernorm.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.input_layernorm.weight",
                        f"model.layers.{layer}.input_layernorm.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.post_attention_layernorm.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.post_attention_layernorm.weight",
                        f"model.layers.{layer}.post_attention_layernorm.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.q_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.q_proj.weight",
                        f"model.layers.{layer}.self_attn.q_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.k_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.k_proj.weight",
                        f"model.layers.{layer}.self_attn.k_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.v_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.v_proj.weight",
                        f"model.layers.{layer}.self_attn.v_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.o_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                        f"model.layers.{layer}.self_attn.o_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.q_norm.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.q_norm.weight",
                        f"model.layers.{layer}.self_attn.q_norm.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.self_attn.k_norm.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.self_attn.k_norm.weight",
                        f"model.layers.{layer}.self_attn.k_norm.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.mlp.gate_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.mlp.gate_proj.weight",
                        f"model.layers.{layer}.mlp.gate_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.mlp.up_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.mlp.up_proj.weight",
                        f"model.layers.{layer}.mlp.up_proj.weight",
                    ],
                ),
                WeightTarget(
                    to_pattern=f"layers.{layer}.mlp.down_proj.weight",
                    from_pattern=[
                        f"model.language_model.layers.{layer}.mlp.down_proj.weight",
                        f"model.layers.{layer}.mlp.down_proj.weight",
                    ],
                ),
            ])
        return targets

