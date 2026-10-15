import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux3.flux3_initializer import Flux3Initializer
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.model.bbox import to_canvas
from mflux.models.flux3.model.flux3_scheduler import Flux3Scheduler
from mflux.models.flux3.variants.edit.reference_prep import (
    MATCH_MAX_PIXELS,
    prepare_multi_reference,
    prepare_reference,
)
from mflux.models.flux3.variants.txt2img.flux3_klein import pack_samples
from mflux.utils.generated_image import GeneratedImage


class Flux3KleinEdit(nn.Module):
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

    def encode_references(self, references: list[Image.Image]) -> tuple[mx.array, mx.array] | None:
        if not references:
            return None
        tokens_list = []
        ids_list = []
        for index, ref in enumerate(references):
            arr = np.array(ref.convert("RGB")).astype(np.float32) / 127.5 - 1.0
            arr = np.transpose(arr, (2, 0, 1))
            img_tensor = mx.array(arr)
            t, ids = self.vae.encode_to_tokens(img_tensor)
            scaled = Flux3LatentCreator.scaled_ids(ids)
            # l-axis marks the reference index as negative offset: -(1 + index)
            l_channel = mx.full((scaled.shape[0], 1), -(1 + index), dtype=scaled.dtype)
            scaled = mx.concatenate([scaled[:, :3], l_channel], axis=-1)
            tokens_list.append(t)
            ids_list.append(scaled)
        return mx.concatenate(tokens_list, axis=0), mx.concatenate(ids_list, axis=0)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str | Path] | None = None,
        images: list[Image.Image] | None = None,
        boxes: list[dict] | None = None,
        width: int | None = None,
        height: int | None = None,
        reference_resolution: int = 1024,
        steps: int = 16,
        alpha: float = 6.93,
    ) -> GeneratedImage:
        start_time = time.time()

        loaded_images = []
        if images is not None:
            loaded_images = list(images)
        elif image_paths:
            loaded_images = [Image.open(p) for p in image_paths]

        ref_budget = (reference_resolution or 1024) ** 2
        if len(loaded_images) > 1:
            references = [prepare_multi_reference(img, max_pixels=ref_budget) for img in loaded_images]
            crop = None
        elif len(loaded_images) == 1:
            prepared, crop = prepare_reference(loaded_images[0], max_pixels=max(MATCH_MAX_PIXELS, ref_budget))
            references = [prepared]
        else:
            references = []
            crop = None

        if references and (width is None or height is None):
            width = width or references[0].width
            height = height or references[0].height
        width = width or 1024
        height = height or 1024

        if boxes:
            caption = f"{prompt.rstrip()} {json.dumps(to_canvas(boxes), ensure_ascii=False)}"
        else:
            caption = prompt

        # 1. Encode prompt
        text_tokens, text_ids = self.prompt_encoder.encode(caption, self.text_encoder)

        # 2. Encode conditioning references
        cond = self.encode_references(references)

        # 3. Sample latent noise
        noise, raw_ids, scaled_ids = Flux3LatentCreator.create_noise(seed, height, width)

        # 4. Pack into multimodal structure
        sample = [(text_tokens, text_ids), cond, (noise, scaled_ids)]
        packed = pack_samples([sample])

        # 5. Denoise
        scheduler = Flux3Scheduler(steps=steps, alpha=alpha)
        for i in range(steps):
            scheduler.denoise_step(self.transformer, packed, i)

        # 6. Decode
        image_array = self.vae.decode_tokens(packed.data["image"], raw_ids, tiling_config=self.tiling_config)
        image_array = mx.clip((image_array + 1.0) * 127.5, 0, 255).astype(mx.uint8)
        image_array = mx.transpose(image_array, (1, 2, 0))
        pil_image = Image.fromarray(np.array(image_array))

        if crop is not None:
            pil_image = pil_image.crop(crop)

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
