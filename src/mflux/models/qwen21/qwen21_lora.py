import copy
import json
import math
import warnings
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.qwen21.model.qwen21_scheduler import ViggleTurboScheduler
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


class Qwen21LoRA:
    PRESET = "viggle-turbo"
    PRESET_VIGGLE = "viggle-turbo"
    PRESET_TURBO8 = "turbo8"
    PRESETS = ("viggle-turbo", "turbo8")
    DEFAULT_STEPS = 6
    PRESET_DEFAULT_STEPS = {"viggle-turbo": 6, "turbo8": 8}
    REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
    REVISION = "009a44a895ef85f7e643c80fdca9543795248867"
    FILENAME = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
    SUPPORTED_FILENAMES = (
        "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors",
        "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
    )
    TURBO8_REPO = "chriswritescode/Turbo8-LoRA-Qwen-Image-2.1"
    TURBO8_FILENAME = "turbo8_lora_step2500.safetensors"

    @staticmethod
    def configure(model_config: ModelConfig, preset: str | None) -> ModelConfig:
        if preset not in (None, *Qwen21LoRA.PRESETS):
            raise ValueError(f"Unknown Qwen-Image-2.1 LoRA preset: {preset}")
        if preset is None:
            return model_config
        config = copy.copy(model_config)
        config.sigma_shift_terminal = None
        return config

    @staticmethod
    def validate_generation(preset: str | None, steps: int, guidance: float, negative_prompt: str | None) -> None:
        if preset == Qwen21LoRA.PRESET_VIGGLE:
            if steps not in ViggleTurboScheduler.SCHEDULES:
                raise ValueError("viggle-turbo v0.3 requires --steps 6 or 8.")
        elif preset == Qwen21LoRA.PRESET_TURBO8:
            if steps != 8:
                warnings.warn("Turbo8 was trained specifically for 8 steps.", UserWarning, stacklevel=2)
        if preset and guidance > 1:
            warnings.warn(
                f"{preset} was distilled for guidance=1 without a negative prompt. "
                "Higher CFG is experimental, adds a second denoising pass per step, and can reduce edit quality.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def resolve_scheduler(preset: str | None, scheduler: str | None) -> str:
        if preset == Qwen21LoRA.PRESET_VIGGLE:
            if scheduler not in (None, "viggle_turbo", "ViggleTurboScheduler"):
                raise ValueError("viggle-turbo requires --scheduler viggle_turbo (selected automatically).")
            return "viggle_turbo"
        if preset == Qwen21LoRA.PRESET_TURBO8:
            if scheduler not in (None, "linear", "LinearScheduler"):
                raise ValueError("turbo8 requires --scheduler linear (selected automatically).")
            return "linear"

        if scheduler in ("qwen_turbo", "QwenTurboScheduler"):
            return "qwen_turbo"
        return scheduler or "linear"

    @staticmethod
    def apply(model, paths, scales, preset: str | None, bake_lora: bool, *, reference: bool) -> None:
        if preset and bake_lora:
            raise ValueError(f"{preset} requires runtime adapters; use --no-bake-lora to avoid bf16 merge loss.")
        if preset and paths == []:
            raise ValueError(f"{preset} requires its adapter; remove the preset to disable LoRA.")
        if preset == Qwen21LoRA.PRESET_VIGGLE and paths is None:
            paths = [hf_hub_download(Qwen21LoRA.REPO, Qwen21LoRA.FILENAME, revision=Qwen21LoRA.REVISION)]
        elif preset == Qwen21LoRA.PRESET_TURBO8 and paths is None:
            paths = [hf_hub_download(Qwen21LoRA.TURBO8_REPO, Qwen21LoRA.TURBO8_FILENAME)]
        elif paths:
            resolved = []
            for path in paths:
                if preset == Qwen21LoRA.PRESET_VIGGLE and path in [
                    f"{Qwen21LoRA.REPO}:{name}" for name in Qwen21LoRA.SUPPORTED_FILENAMES
                ]:
                    resolved.append(
                        hf_hub_download(Qwen21LoRA.REPO, path.split(":", 1)[1], revision=Qwen21LoRA.REVISION)
                    )
                elif preset == Qwen21LoRA.PRESET_TURBO8 and path in [
                    f"{Qwen21LoRA.TURBO8_REPO}:{Qwen21LoRA.TURBO8_FILENAME}",
                    Qwen21LoRA.TURBO8_FILENAME,
                ]:
                    resolved.append(hf_hub_download(Qwen21LoRA.TURBO8_REPO, Qwen21LoRA.TURBO8_FILENAME))
                elif ":" in path and not path.startswith("http") and not Path(path).exists():
                    r_repo, r_file = path.split(":", 1)
                    resolved.append(hf_hub_download(r_repo, r_file))
                else:
                    resolved.append(path)
            paths = resolved
        paths = LoraResolution.resolve_paths(paths)
        accelerator_names = (*Qwen21LoRA.SUPPORTED_FILENAMES, Qwen21LoRA.TURBO8_FILENAME)
        accelerators = [Path(path).name for path in paths if Path(path).name in accelerator_names]
        if len(accelerators) > 1:
            raise ValueError("Use exactly one acceleration LoRA; do not stack distilled students.")
        if accelerators:
            expected = (
                Qwen21LoRA.PRESET_TURBO8 if accelerators[0] == Qwen21LoRA.TURBO8_FILENAME else Qwen21LoRA.PRESET_VIGGLE
            )
            if preset != expected:
                raise ValueError(
                    f"Load this adapter with --lora-preset {expected} so its required scheduler is enabled."
                )
        scales = LoraResolution.resolve_scales(scales, len(paths))
        if any(not math.isfinite(scale) for scale in scales):
            raise ValueError("LoRA scales must be finite.")
        if preset == Qwen21LoRA.PRESET_VIGGLE and not any(
            Path(p).name in Qwen21LoRA.SUPPORTED_FILENAMES for p in paths
        ):
            allowed = ", ".join(Qwen21LoRA.SUPPORTED_FILENAMES)
            raise ValueError(f"viggle-turbo requires at least one of the official v0.3 adapters: {allowed}")
        if preset == Qwen21LoRA.PRESET_TURBO8 and not any(Path(p).name == Qwen21LoRA.TURBO8_FILENAME for p in paths):
            raise ValueError(f"turbo8 requires the {Qwen21LoRA.TURBO8_FILENAME} adapter.")
        mapping = Qwen21LoRAMapping.get_mapping(reference=reference)
        for path in paths:
            Qwen21LoRA._validate(model.transformer, path, mapping)
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            mapping, model.transformer, paths, scales, bake_lora=bake_lora
        )
        model.lora_preset = preset
        model.bake_lora = bake_lora

    @staticmethod
    def _validate(transformer: nn.Module, path: str, mapping) -> None:
        weights, metadata = mx.load(path, return_metadata=True)
        adapter_metadata = json.loads(metadata.get("lora_adapter_metadata", "{}"))
        if adapter_metadata.get("transformer.use_rslora") or adapter_metadata.get("transformer.use_dora"):
            raise ValueError("Qwen-Image-2.1 currently supports standard LoRA, not rsLoRA or DoRA.")
        if adapter_metadata.get("transformer.alpha_pattern") or adapter_metadata.get("transformer.rank_pattern"):
            raise ValueError("Per-layer PEFT alpha/rank metadata patterns are not supported; export explicit tensors.")
        alpha, rank = adapter_metadata.get("transformer.lora_alpha"), adapter_metadata.get("transformer.r")
        if alpha is not None and rank is not None and alpha != rank:
            raise ValueError("Adapter metadata has alpha != rank; export per-layer alpha tensors before loading.")
        patterns = LoRALoader._build_pattern_mappings(mapping)
        grouped = {}
        for key, value in weights.items():
            matches = [(p, LoRALoader._match_pattern(key, p.source_pattern)) for p in patterns]
            matches = [(p, index) for p, index in matches if index is not None]
            if not matches:
                raise ValueError(
                    f"Unsupported Qwen-Image-2.1 LoRA tensor: {key}. Qwen Image 1.x adapters are incompatible."
                )
            for pattern, index in matches:
                target = pattern.target_path.format(block=index)
                group = grouped.setdefault(target, {})
                if pattern.matrix_name in group:
                    raise ValueError(f"Duplicate LoRA matrix for {target}: {pattern.matrix_name}")
                group[pattern.matrix_name] = pattern.transform(value) if pattern.transform else value
        if not grouped:
            raise ValueError("The LoRA file contains no adapter tensors.")
        for target, tensors in grouped.items():
            try:
                layer = LoRALoader._get_target_module(transformer, target)
            except (AttributeError, KeyError, IndexError) as exc:
                raise ValueError(f"LoRA target is absent from Qwen-Image-2.1: {target}") from exc
            if not isinstance(layer, (nn.Linear, nn.QuantizedLinear, LoRALinear, FusedLoRALinear)):
                raise ValueError(f"Unsupported LoRA target type at {target}")
            a, b = tensors.get("lora_A"), tensors.get("lora_B")
            if "alpha" in tensors and (tensors["alpha"].size != 1 or not math.isfinite(float(tensors["alpha"]))):
                raise ValueError(f"LoRA alpha must be a finite scalar for {target}")
            base_module = getattr(layer, 'base_linear', getattr(layer, 'linear', layer))
            out_dim, in_dim = base_module.weight.shape
            if isinstance(base_module, nn.QuantizedLinear):
                in_dim *= 32 // base_module.bits
            if (
                a is None
                or b is None
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[0] == 0
                or a.shape != (b.shape[1], in_dim)
                or b.shape[0] != out_dim
            ):
                raise ValueError(
                    f"Invalid LoRA matrix shapes for {target}; expected A=(rank,{in_dim}), B=({out_dim},rank)."
                )
