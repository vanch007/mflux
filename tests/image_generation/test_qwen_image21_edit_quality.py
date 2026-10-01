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
from mflux.models.qwen21.qwen21_lora import Qwen21LoRA
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.prompt_encoder import QwenImage21PromptEncoder
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

pytestmark = pytest.mark.fast


class TestQwenImage21EditQuality:
    @pytest.mark.parametrize("output_resolution", [64, 128])
    @pytest.mark.parametrize("reference_resolution", [None, 32])
    def test_output_budget_does_not_change_reference_encoding(self, tmp_path, output_resolution, reference_resolution):
        source = tmp_path / "source.png"
        Image.new("RGB", (96, 96), "red").save(source)
        seen = {}

        def encode_prompt(_, images):
            seen["vision"] = images[0].size
            slots = (images[0].width // 32) * (images[0].height // 32)
            return mx.zeros((1, slots + 1, 16)), mx.array([False] + [True] * slots)

        def encode_vae(pixels):
            seen["vae"] = pixels.shape[-2:]
            return mx.zeros((1, 64, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16))

        def transformer(hidden, *_, **kwargs):
            return mx.zeros((1, (output_resolution // 16) ** 2, 64), dtype=hidden.dtype)

        transformer.axes = (4, 6, 6)
        model = SimpleNamespace(
            lora_preset=None,
            model_config=ModelConfig.qwen_image_21(),
            processor=SimpleNamespace(
                image_processor=SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 4096**2})
            ),
            _encode_prompt=encode_prompt,
            transformer=transformer,
            vae=SimpleNamespace(
                encode=encode_vae, decode=lambda _: mx.zeros((1, 4, output_resolution, output_resolution))
            ),
            callbacks=CallbackRegistry(),
            tiling_config=None,
            bits=None,
            lora_paths=[],
            lora_scales=[],
            bake_lora=False,
        )
        kwargs = {} if reference_resolution is None else {"reference_resolution": reference_resolution}
        image = QwenImage21Edit.generate_image(
            model,
            seed=42,
            prompt="red",
            image_paths=[source],
            num_inference_steps=2,
            output_resolution=output_resolution,
            **kwargs,
        )
        budget = reference_resolution or 1024
        assert seen == {"vision": (budget, budget), "vae": (budget, budget)}
        assert image.image.size == (output_resolution, output_resolution)
        assert image._get_metadata()["reference_resolution"] == budget

    @pytest.mark.parametrize("saved_reference", [None, 1024])
    def test_reference_budget_metadata_and_explicit_override(self, tmp_path, monkeypatch, saved_reference):
        data = {"model": "Qwen/Qwen-Image-2.1", "prompt": "red", "output_resolution": 1536}
        if saved_reference is not None:
            data["reference_resolution"] = saved_reference
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(data))
        args = ["qwen21", "--config-from-metadata", str(path)]
        monkeypatch.setattr(sys, "argv", args)
        restored = build_parser().parse_args()
        assert restored.output_resolution == 1536
        assert restored.reference_resolution == (saved_reference or 1536)
        monkeypatch.setattr(sys, "argv", args + ["--reference-resolution", "512"])
        assert build_parser().parse_args().reference_resolution == 512

    def test_new_cli_has_independent_reference_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test", "--output-resolution", "1536"])
        args = build_parser().parse_args()
        assert (args.output_resolution, args.reference_resolution) == (1536, 1024)

    def test_experimental_cfg_warns_but_remains_available(self):
        with pytest.warns(UserWarning, match="can reduce edit quality"):
            Qwen21LoRA.validate_generation("viggle-turbo", 6, 2, "pink dress")

    @pytest.mark.parametrize(
        "mode,pixel,expected", [("RGB", (255, 0, 0), (255, 0, 0)), ("RGBA", (255, 0, 0, 128), (255, 127, 127))]
    )
    def test_vision_input_rgb_and_alpha_compositing(self, mode, pixel, expected):
        class Processor:
            def __call__(self, **kwargs):
                self.image = kwargs["images"][0]
                return {
                    "input_ids": np.array([[1, 99, 2]]),
                    "pixel_values": np.zeros((1, 1)),
                    "image_grid_thw": np.ones((1, 3)),
                }

            def apply_chat_template(self, *args, **kwargs):
                return [1]

        class Encoder:
            image_token_id = 99

            def __call__(self, *args, **kwargs):
                return mx.zeros((1, 3, 4))

        processor = Processor()
        source = Image.new(mode, (32, 32), pixel)
        hidden, slots = QwenImage21PromptEncoder.encode("red", [source], processor, Encoder())
        assert processor.image.mode == "RGB"
        assert processor.image.getpixel((0, 0)) == expected
        assert source.getpixel((0, 0)) == pixel
        assert hidden.shape == (1, 2, 4)
        assert slots.tolist() == [True, False]
