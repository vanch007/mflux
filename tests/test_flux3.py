import pytest
import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.model.bbox import to_canvas, detect
from mflux.models.flux3.model.flux3_scheduler import Flux3Scheduler
from mflux.models.flux3.model.flux3_transformer.pos_embed import (
    compute_rope_freqs,
    rmsnorm_rope,
    timestep_embedding,
)
from mflux.models.flux3.model.flux3_transformer.transformer import Flux3Transformer, Packed
from mflux.models.flux3.variants.txt2img.flux3_klein import pack_samples
from mflux.models.flux3.weights.flux3_weight_definition import Flux3KleinWeightDefinition
from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS


@pytest.mark.fast
def test_flux3_model_config():
    config = ModelConfig.from_name("flux-3-image-klein")
    assert config is ModelConfig.flux_3_image_klein()
    assert config.model_name == "black-forest-labs/flux-3-image-klein"
    assert "flux-3" in config.aliases
    assert "flux3" in config.aliases
    assert config.supports_guidance is False
    assert MODEL_INFERENCE_STEPS["flux-3-image-klein"] == 16


@pytest.mark.fast
def test_flux3_scheduler():
    sched = Flux3Scheduler(steps=16, alpha=6.93)
    s = sched.schedule
    assert len(s) == 17
    assert float(s[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(s[-1]) == pytest.approx(0.0, abs=1e-5)
    for i in range(len(s) - 1):
        assert float(s[i]) > float(s[i + 1])


@pytest.mark.fast
def test_flux3_latent_creator():
    h, w = Flux3LatentCreator.latent_shape(512, 512)
    assert (h, w) == (32, 32)

    with pytest.raises(ValueError):
        Flux3LatentCreator.latent_shape(500, 500)

    noise, raw_ids, scaled_ids = Flux3LatentCreator.create_noise(seed=42, height=256, width=256)
    assert noise.shape == (256, 128)
    assert raw_ids.shape == (256, 4)
    assert scaled_ids.shape == (256, 4)

    assert float(scaled_ids[1, 2]) == pytest.approx(float(raw_ids[1, 2]) * 0.5)


@pytest.mark.fast
def test_flux3_pos_embed():
    t = mx.array([0.5, 0.8])
    emb = timestep_embedding(t, dim=256)
    assert emb.shape == (2, 256)

    ids = mx.zeros((1, 16, 4), dtype=mx.float32)
    cos_all, sin_all = compute_rope_freqs(ids, axes_dim=(32, 32, 32, 32))
    assert cos_all.shape == (1, 1, 16, 64)
    assert sin_all.shape == (1, 1, 16, 64)

    x = mx.ones((1, 24, 16, 128), dtype=mx.float32)
    scale = mx.ones((128,), dtype=mx.float32)
    rot = rmsnorm_rope(x, scale, cos_all, sin_all)
    assert rot.shape == (1, 24, 16, 128)


@pytest.mark.fast
def test_flux3_bbox_parsing():
    boxes = [{"id": "box1", "bbox": [0.1, 0.2, 0.5, 0.6], "desc": "a car"}]
    canvas_boxes = to_canvas(boxes)
    assert canvas_boxes[0]["bbox"] == [100, 200, 500, 600]
    assert canvas_boxes[0]["desc"] == "a car"


@pytest.mark.fast
def test_flux3_transformer_tiny_forward():
    dit = Flux3Transformer(
        hidden=256,
        heads=2,
        depth=1,
        depth_single=1,
        mlp_ratio=2.0,
        flow_map=True,
    )
    dit.emb_in.image = nn.Linear(128, 256, bias=False)
    dit.emb_in.image_cond = nn.Linear(128, 256, bias=False)
    dit.txt_in = nn.Linear(128, 256, bias=False)
    dit.final_layer.image.linear = nn.Linear(256, 128, bias=False)

    text_tokens = mx.zeros((8, 128), dtype=mx.bfloat16)
    text_ids = mx.zeros((8, 4), dtype=mx.float32)
    noise = mx.zeros((16, 128), dtype=mx.bfloat16)
    image_ids = mx.zeros((16, 4), dtype=mx.float32)

    sample = [(text_tokens, text_ids), None, (noise, image_ids)]
    packed = pack_samples([sample])
    out = dit(packed, timesteps_ref=mx.array([0.9]))
    assert out.shape == (16, 128)


@pytest.mark.fast
def test_flux3_weight_definition():
    comps = Flux3KleinWeightDefinition.get_components()
    names = {c.name for c in comps}
    assert "transformer" in names
    assert "vae" in names
    assert "text_encoder" in names

    assert Flux3KleinWeightDefinition.quantization_predicate("", nn.Linear(64, 64)) is True
    assert Flux3KleinWeightDefinition.quantization_predicate("", nn.Conv2d(3, 3, 3)) is False

