from pathlib import Path
from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux3.weights.flux3_weight_mapping import Flux3WeightMapping


class Flux3KleinWeightDefinition:
    @staticmethod
    def _transformer_variant_selector(root: Path) -> ComponentDefinition:
        if (root / "model.safetensors").exists():
            return ComponentDefinition(
                name="transformer",
                hf_subdir="",
                weight_files=["model.safetensors"],
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_transformer_mapping,
            )
        return ComponentDefinition(
            name="transformer",
            hf_subdir="transformer",
            precision=ModelConfig.precision,
            mapping_getter=Flux3WeightMapping.get_transformer_mapping,
        )

    @staticmethod
    def _vae_variant_selector(root: Path) -> ComponentDefinition:
        if (root / "ae.safetensors").exists():
            return ComponentDefinition(
                name="vae",
                hf_subdir="",
                weight_files=["ae.safetensors"],
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_vae_mapping,
            )
        return ComponentDefinition(
            name="vae",
            hf_subdir="vae",
            precision=ModelConfig.precision,
            mapping_getter=Flux3WeightMapping.get_vae_mapping,
        )

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_transformer_mapping,
                variant_selector=Flux3KleinWeightDefinition._transformer_variant_selector,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_vae_mapping,
                variant_selector=Flux3KleinWeightDefinition._vae_variant_selector,
                skip_quantization=True,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="text_encoder",
                tokenizer_class="Qwen2TokenizerFast",
                encoder_class=LanguageTokenizer,
                max_length=4096,
                use_chat_template=True,
                chat_template_kwargs={"enable_thinking": False},
                download_patterns=["text_encoder/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "model.safetensors",
            "ae.safetensors",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "text_encoder/*.txt",
            "config.json",
            "manifest.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        from mlx import nn
        return isinstance(module, nn.Linear) and not isinstance(module, nn.Conv2d) and getattr(module, "weight", None) is not None and module.weight.ndim == 2

