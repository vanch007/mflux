import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.cli import qwen21_edit_generate, qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.qwen21_lora import Qwen21LoRA
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping

pytestmark = pytest.mark.fast


class TestQwen21LoRA:
    @staticmethod
    def transformer(reference):
        config = dict(
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            axes_dims_rope=(4, 6, 6),
            context_in_dim=32,
            in_channels=4,
            out_channels=4,
            mlp_ratio=3,
            eps=1e-6,
        )
        return QwenImage21Transformer(config) if reference else Qwen21Transformer(**config)

    @pytest.mark.parametrize("reference", [True, False], ids=["edit", "txt2img"])
    @pytest.mark.parametrize("convention", ["diffusers", "peft", "peft-default", "comfy", "kohya", "bare"])
    def test_all_targets_match_runtime_delta(self, tmp_path, reference, convention):
        transformer = self.transformer(reference)
        weights, expected = {}, {}
        for target in Qwen21LoRAMapping.get_mapping(reference=reference):
            path = target.model_path.format(block=1)
            source = path.replace("modulation.layers.1", "modulation.1")
            layer = LoRALoader._get_target_module(transformer, path)
            out_dim, in_dim = layer.weight.shape
            a = mx.random.normal((2, in_dim)) * 0.05
            b = mx.random.normal((out_dim, 2)) * 0.05
            x = mx.random.normal((3, in_dim))
            expected[path] = (x, layer(x) + 0.75 * (x @ a.T @ b.T) * 1.5)
            prefix = {
                "diffusers": "transformer.",
                "peft": "base_model.model.",
                "peft-default": "base_model.model.",
                "comfy": "diffusion_model.",
                "bare": "",
            }.get(convention)
            if convention == "kohya":
                root = "lora_unet_" + source.replace(".", "_")
                weights[f"{root}.lora_down.weight"] = a
                weights[f"{root}.lora_up.weight"] = b
            else:
                root = prefix + source
                suffix = {"comfy": "", "peft-default": ".default.weight"}.get(convention, ".weight")
                weights[f"{root}.lora_A{suffix}"] = a
                weights[f"{root}.lora_B{suffix}"] = b
            weights[f"{root}.alpha"] = mx.array(3.0)
        path = tmp_path / "adapter.safetensors"
        mx.save_safetensors(str(path), weights)
        model = SimpleNamespace(transformer=transformer)
        Qwen21LoRA.apply(model, [str(path)], [0.75], None, False, reference=reference)
        for target, (x, value) in expected.items():
            layer = LoRALoader._get_target_module(transformer, target)
            assert isinstance(layer, LoRALinear)
            np.testing.assert_allclose(np.array(layer(x)), np.array(value), atol=1e-6, rtol=1e-5)
        assert model.lora_paths == [str(path)] and model.lora_scales == [0.75]

    @pytest.mark.parametrize("failure", ["shape", "missing", "foreign", "mixed", "metadata"])
    def test_bad_adapters_fail_before_modifying_layers(self, tmp_path, failure):
        transformer = self.transformer(True)
        root = "transformer.transformer_blocks.0.attn.to_q"
        weights = {f"{root}.lora_A.weight": mx.ones((2, 32)), f"{root}.lora_B.weight": mx.ones((32, 2))}
        metadata = {}
        if failure == "shape":
            weights[f"{root}.lora_A.weight"] = mx.ones((2, 16))
        elif failure == "missing":
            del weights[f"{root}.lora_B.weight"]
        elif failure in ("foreign", "mixed"):
            foreign = {"transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": mx.ones((2, 32))}
            weights = foreign if failure == "foreign" else {**weights, **foreign}
        else:
            metadata = {"lora_adapter_metadata": json.dumps({"transformer.r": 2, "transformer.lora_alpha": 4})}
        path = tmp_path / "invalid.safetensors"
        mx.save_safetensors(str(path), weights, metadata)
        with pytest.raises(ValueError):
            Qwen21LoRA.apply(SimpleNamespace(transformer=transformer), [str(path)], None, None, False, reference=True)
        assert isinstance(transformer.transformer_blocks[0].attn.to_q, nn.Linear)

    @pytest.mark.parametrize("width,height", [(512, 736), (1024, 1024), (1536, 1536), (2048, 2048)])
    @pytest.mark.parametrize("steps", [6, 8])
    def test_turbo_sigmas_match_published_scheduler_without_mutating_base(self, width, height, steps):
        base = ModelConfig.qwen_image_21()
        config = Qwen21LoRA.configure(base, "viggle-turbo")
        assert config is not base and base.sigma_shift_terminal == 0.02
        scheduler = Config(
            model_config=config, num_inference_steps=steps, width=width, height=height, scheduler="viggle_turbo"
        ).scheduler
        mu = 0.5 + (0.9 - 0.5) * (width * height / 256 - 256) / (8192 - 256)
        sigmas = np.array(
            [1, 0.9375, 0.875, 0.75, 0.5, 0.25] if steps == 6 else [1, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125]
        )
        expected = np.r_[np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1)), 0]
        np.testing.assert_allclose(np.array(scheduler.sigmas), expected, atol=1e-6)
        assert Qwen21LoRA.configure(base, None) is base

    @pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
    def test_cli_metadata_restores_preset_and_runtime_adapters(self, tmp_path, monkeypatch, module):
        sidecar = tmp_path / "sample.json"
        sidecar.write_text(
            json.dumps(
                dict(prompt="test", steps=8, lora_preset="viggle-turbo", bake_lora=False, scheduler="viggle_turbo")
            )
        )
        monkeypatch.setattr(sys, "argv", ["qwen21", "--config-from-metadata", str(sidecar)])
        parser = module.build_parser()
        args = parser.parse_args()
        assert parser.supports_lora and args.lora_preset == "viggle-turbo"
        assert args.steps == 8 and args.bake_lora is False and args.scheduler == "viggle_turbo"
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test"])
        args = module.build_parser().parse_args()
        assert args.steps == 40 and args.lora_preset is None and args.bake_lora is False

    @pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
    @pytest.mark.parametrize(
        "options",
        [
            ["--steps", "40"],
            ["--steps", "4"],
            ["--steps", "7"],
            ["--steps", "6", "--bake-lora"],
            ["--steps", "6", "--scheduler", "linear"],
        ],
    )
    def test_invalid_turbo_cli_rejected_before_loading(self, monkeypatch, module, options):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test", "--lora-preset", "viggle-turbo", *options])
        name = "QwenImage21Edit" if module is qwen21_edit_generate else "QwenImage21"
        monkeypatch.setattr(module, name, lambda **kwargs: pytest.fail("Should reject before loading weights"))
        with pytest.raises(SystemExit) as exc:
            module.main()
        assert exc.value.code == 2

    @pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
    def test_cli_passes_lora_settings_to_model(self, monkeypatch, module):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test", "--lora-preset", "viggle-turbo"])
        received = {}

        def construct(**kwargs):
            received.update(kwargs)

            def generate(**kwargs):
                received["generation"] = kwargs
                return SimpleNamespace(save=lambda *a, **k: None)

            return SimpleNamespace(generate_image=generate)

        name = "QwenImage21Edit" if module is qwen21_edit_generate else "QwenImage21"
        monkeypatch.setattr(module, name, construct)
        monkeypatch.setattr(module.CallbackManager, "register_callbacks", lambda *args, **kwargs: None)
        module.main()
        assert received["lora_preset"] == "viggle-turbo" and received["bake_lora"] is False
        assert received["generation"]["num_inference_steps"] == 6
        assert received["generation"]["scheduler"] == "viggle_turbo"

    def test_preset_refuses_baking_and_disabled_adapter(self):
        for paths, bake in [(None, True), ([], False)]:
            with pytest.raises(ValueError):
                Qwen21LoRA.apply(SimpleNamespace(), paths, None, "viggle-turbo", bake, reference=True)

    def test_turbo_adapter_cannot_silently_use_base_scheduler(self, tmp_path):
        path = tmp_path / Qwen21LoRA.FILENAME
        path.touch()
        with pytest.raises(ValueError, match="required scheduler"):
            Qwen21LoRA.apply(SimpleNamespace(), [str(path)], None, None, False, reference=True)

    def test_quantized_layer_keeps_runtime_delta(self, tmp_path):
        transformer = self.transformer(True)
        layer = transformer.transformer_blocks[0].attn.to_q.to_quantized(group_size=32, bits=8)
        transformer.transformer_blocks[0].attn.to_q = layer
        a, b = mx.ones((2, 32)) * 0.01, mx.ones((32, 2)) * 0.02
        x = mx.random.normal((3, 32))
        expected = layer(x) + x @ a.T @ b.T
        path = tmp_path / "adapter.safetensors"
        root = "transformer.transformer_blocks.0.attn.to_q"
        mx.save_safetensors(str(path), {f"{root}.lora_A.weight": a, f"{root}.lora_B.weight": b})
        Qwen21LoRA.apply(SimpleNamespace(transformer=transformer), [str(path)], None, None, False, reference=True)
        np.testing.assert_allclose(
            np.array(transformer.transformer_blocks[0].attn.to_q(x)), np.array(expected), atol=1e-6
        )

    @pytest.mark.parametrize("reference", [True, False])
    @pytest.mark.parametrize("prefix", ["transformer.", "diffusion_model."])
    def test_fused_gate_up_preserves_both_runtime_updates(self, tmp_path, reference, prefix):
        transformer = self.transformer(reference)
        mlp = transformer.transformer_blocks[0].img_mlp
        out_dim, in_dim = mlp.gate_layer.weight.shape
        a, b = mx.random.normal((2, in_dim)), mx.random.normal((2 * out_dim, 2))
        x = mx.random.normal((3, in_dim))
        gate = mlp.gate_layer(x) + 0.7 * (x @ a.T @ b[:out_dim].T)
        proj = mlp.proj(x) + 0.7 * (x @ a.T @ b[out_dim:].T)
        expected = mlp.out(nn.silu(gate) * proj)
        mx.eval(expected)
        path = tmp_path / "fused.safetensors"
        root = f"{prefix}transformer_blocks.0.img_mlp.gate_up"
        mx.save_safetensors(str(path), {f"{root}.lora_A.weight": a, f"{root}.lora_B.weight": b})
        Qwen21LoRA.apply(SimpleNamespace(transformer=transformer), [str(path)], [0.7], None, False, reference=reference)
        np.testing.assert_allclose(np.array(mlp(x)), np.array(expected), rtol=1e-5, atol=1e-5)

    def test_turbo_allows_custom_scale(self, tmp_path, monkeypatch):
        path = tmp_path / Qwen21LoRA.FILENAME
        path.touch()
        monkeypatch.setattr(Qwen21LoRA, "_validate", lambda *_: None)
        model = SimpleNamespace(transformer=SimpleNamespace())
        monkeypatch.setattr("mflux.models.common.lora.mapping.lora_loader.LoRALoader.load_and_apply_lora", lambda *args, **kwargs: ([str(path)], [0.85]))
        Qwen21LoRA.apply(model, [str(path)], [0.85], "viggle-turbo", False, reference=True)
        assert model.lora_scales == [0.85]

    @pytest.mark.parametrize("preset,scheduler", [("viggle-turbo", "linear"), (None, "viggle_turbo")])
    def test_scheduler_cannot_silently_switch_model_recipe(self, preset, scheduler):
        with pytest.raises(ValueError):
            Qwen21LoRA.resolve_scheduler(preset, scheduler)
