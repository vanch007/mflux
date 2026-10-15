from pathlib import Path
import time
import numpy as np
from PIL import Image

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux3.flux3_initializer import Flux3Initializer
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.model.flux3_scheduler import Flux3Scheduler
from mflux.models.flux3.model.flux3_transformer.transformer import IN_CHANNELS, MODES, Packed
from mflux.utils.generated_image import GeneratedImage


def pack_samples(samples: list[list[tuple[mx.array, mx.array] | None]]) -> Packed:
    """samples[s] = [text, image_cond or None, image], each a (tokens, ids) pair."""
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = {m: 0 for m in MODES}

    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))

    out_data = {}
    out_ids = {}
    out_timesteps = {}
    for m in MODES:
        if data[m]:
            out_data[m] = mx.concatenate(data[m], axis=0)
            out_ids[m] = mx.concatenate(ids[m], axis=0).astype(mx.float32)
        else:
            out_data[m] = mx.zeros((0, IN_CHANNELS[m]), dtype=mx.bfloat16)
            out_ids[m] = mx.zeros((0, 4), dtype=mx.float32)
        out_timesteps[m] = mx.zeros((out_data[m].shape[0],), dtype=out_data[m].dtype)

    return Packed(
        data=out_data,
        ids=out_ids,
        timesteps=out_timesteps,
        mode_indices=mode_indices,
    )


class Flux3Klein(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        model_path: str | None = None,
        compute_precision: mx.Dtype | None = None,
    ):
        super().__init__()
        model_config = model_config or ModelConfig.from_name("flux-3-image-klein")
        Flux3Initializer.init(
            model=self,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            compute_precision=compute_precision,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 16,
        alpha: float = 6.93,
        stepwise_output_dir: Path | str | None = None,
    ) -> GeneratedImage:
        start_time = time.time()

        # 1. Encode text prompt
        text_tokens, text_ids = self.prompt_encoder.encode(prompt, self.text_encoder)

        # 2. Sample noise
        noise, raw_ids, scaled_ids = Flux3LatentCreator.create_noise(seed, height, width)

        # 3. Pack into multimodal structure
        sample = [(text_tokens, text_ids), None, (noise, scaled_ids)]
        packed = pack_samples([sample])

        # 4. Schedule & Denoise
        scheduler = Flux3Scheduler(steps=steps, alpha=alpha)
        for i in range(steps):
            scheduler.denoise_step(self.transformer, packed, i)

        # 5. Decode tokens via VAE
        image_array = self.vae.decode_tokens(packed.data["image"], raw_ids, tiling_config=self.tiling_config)
        # Convert [-1, 1] -> [0, 255] uint8 Image
        image_array = mx.clip((image_array + 1.0) * 127.5, 0, 255).astype(mx.uint8)
        image_array = mx.transpose(image_array, (1, 2, 0))
        pil_image = Image.fromarray(np.array(image_array))
        generation_time = time.time() - start_time

        return GeneratedImage(
            image=pil_image,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=steps,
            guidance=None,
            precision=ModelConfig.precision,
            quantization=self.bits,
            generation_time=generation_time,
            height=height,
            width=width,
        )


    def save_model(self, base_path: str) -> None:
        from mflux.models.common.weights.saving.model_saver import ModelSaver
        from mflux.models.flux3.weights.flux3_weight_definition import Flux3KleinWeightDefinition
        ModelSaver.save_model(self, self.bits, base_path, Flux3KleinWeightDefinition)
