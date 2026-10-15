import math
import mlx.core as mx
from transformers import AutoTokenizer

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


class Flux3PromptEncoder:
    def __init__(self, tokenizer_path: str):
        from pathlib import Path
        p = Path(tokenizer_path)
        if (p / "text_encoder").exists():
            p = p / "text_encoder"
        self.tokenizer = AutoTokenizer.from_pretrained(str(p))
        self.tokenizer.padding_side = "right"

    def encode(
        self,
        prompt: str,
        text_encoder: Qwen3TextEncoder,
        max_length: int = MAX_LENGTH,
        pad_multiple: int = PAD_MULTIPLE,
        output_layers: tuple[int, ...] = OUTPUT_LAYERS,
    ) -> tuple[mx.array, mx.array]:
        """Encodes prompt -> (tokens [L, 20480], ids [L, 4])"""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_tokens = self.tokenizer(text, return_tensors="np", truncation=True, max_length=max_length)
        real_len = raw_tokens["input_ids"].shape[1]
        length = min(math.ceil(real_len / pad_multiple) * pad_multiple, max_length)

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=length,
        )

        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # (1, length, 20480)
        embeds = text_encoder.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=output_layers,
        )
        tokens = embeds[0]

        positions = mx.arange(length, dtype=mx.float32)
        zeros = mx.zeros_like(positions)
        ids = mx.stack([zeros, zeros, zeros, positions], axis=-1)

        return tokens, ids

