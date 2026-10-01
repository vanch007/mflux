import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.cli.qwen21_edit_generate import build_parser
from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

pytestmark = pytest.mark.fast


class TestQwenImage21EditNoise:
    def test_seed_depends_on_pixels_order_and_requested_seed(self):
        red = Image.new("RGBA", (32, 32), "red")
        blue = Image.new("RGBA", (32, 32), "blue")
        resolve = QwenImage21LatentCreator.edit_noise_seed
        assert resolve(42, []) == 42
        assert resolve(42, [red]) == resolve(42, [red.copy()])
        assert resolve(42, [red]) == resolve(42, [red.convert("RGB")])
        transparent = red.copy()
        transparent.putalpha(128)
        seeds = [
            42,
            resolve(42, [red]),
            resolve(43, [red]),
            resolve(42, [blue]),
            resolve(42, [transparent]),
            resolve(42, [red, blue]),
            resolve(42, [blue, red]),
        ]
        assert len(set(seeds)) == len(seeds)

    def test_generation_separates_edit_noise_and_records_replay_policy(self, tmp_path):
        source = tmp_path / "source.png"
        Image.new("RGBA", (32, 32), "red").save(source)
        captured = []

        def transformer(hidden, *_, **kwargs):
            captured.append(np.array(hidden[:, -4:]))
            return mx.zeros((1, 4, 64), dtype=hidden.dtype)

        transformer.axes = (4, 6, 6)
        model = SimpleNamespace(
            lora_preset=None,
            model_config=ModelConfig.qwen_image_21(),
            processor=SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 4096})),
            _encode_prompt=lambda _, images: (mx.zeros((1, 2, 16)), mx.array([False, bool(images)])),
            transformer=transformer,
            vae=SimpleNamespace(
                encode=lambda _: mx.zeros((1, 64, 1, 2, 2)),
                decode=lambda _: mx.zeros((1, 4, 32, 32)),
            ),
            callbacks=CallbackRegistry(),
            tiling_config=None,
            bits=None,
            lora_paths=[],
            lora_scales=[],
            bake_lora=False,
        )
        noises, metadata = [], []
        for paths, independent in [([], True), ([source], True), ([source], True), ([source], False)]:
            captured.clear()
            image = QwenImage21Edit.generate_image(
                model,
                seed=42,
                prompt="red",
                image_paths=paths,
                output_resolution=32,
                reference_resolution=32,
                num_inference_steps=2,
                independent_edit_noise=independent,
            )
            noises.append(captured[0])
            metadata.append(image._get_metadata())
        np.testing.assert_array_equal(noises[0], noises[3])
        np.testing.assert_array_equal(noises[1], noises[2])
        assert not np.array_equal(noises[0], noises[1])
        assert all(item["seed"] == 42 for item in metadata)
        assert metadata[1]["noise_seed"] != 42
        assert metadata[1]["independent_edit_noise"] is True
        assert metadata[3]["noise_seed"] == 42
        assert metadata[3]["independent_edit_noise"] is False

    @pytest.mark.parametrize("flag,expected", [(None, True), ("--no-independent-edit-noise", False)])
    def test_cli_defaults_and_opt_out(self, monkeypatch, flag, expected):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test"] + ([flag] if flag else []))
        assert build_parser().parse_args().independent_edit_noise is expected

    @pytest.mark.parametrize("policy", [None, False, True])
    def test_metadata_replay_preserves_old_and_new_noise(self, tmp_path, monkeypatch, policy):
        data = {"model": "Qwen/Qwen-Image-2.1", "prompt": "test", "seed": 42}
        if policy is not None:
            data["independent_edit_noise"] = policy
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(data))
        argv = ["qwen21", "--config-from-metadata", str(path)]
        monkeypatch.setattr(sys, "argv", argv)
        assert build_parser().parse_args().independent_edit_noise is bool(policy)
        monkeypatch.setattr(sys, "argv", argv + ["--independent-edit-noise"])
        assert build_parser().parse_args().independent_edit_noise is True
