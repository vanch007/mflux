import hashlib
import importlib.util
import json
import math
import re
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import requests

from mflux.models.common.weights.loading.weight_loader import WeightLoader

_spec = importlib.util.spec_from_file_location(
    "restore_qwen21_heretic_head", Path(__file__).parents[2] / "scripts/restore_qwen21_heretic_head.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Restorer = _module.HereticGenerationHeadRestorer

pytestmark = pytest.mark.fast


class TinyCheckpoint:
    @staticmethod
    def setup(tmp_path, monkeypatch, *, wrong_shape=False, truncated=False, quantized=False):
        root = tmp_path / "model" / "text_encoder"
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 3, "hidden_size": 2}}))
        metadata = {"quantization_level": "8" if quantized else "None", "mflux_version": "0.20.0"}
        mx.save_safetensors(str(root / "0.safetensors"), {"embed.weight": mx.ones((3, 2))}, metadata)
        index = {"metadata": metadata, "weight_map": {"embed.weight": "0.safetensors"}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
        source = {}
        for filename, key, shape in (
            ("head.safetensors", "lm_head.weight", (4 if wrong_shape else 3, 2)),
            ("norm.safetensors", "model.language_model.norm.weight", (2,)),
        ):
            path = tmp_path / filename
            mx.save_safetensors(str(path), {key: mx.arange(math.prod(shape)).reshape(shape).astype(mx.bfloat16)})
            source[filename] = path.read_bytes()
        source["model.safetensors.index.json"] = json.dumps(
            {
                "weight_map": {
                    "lm_head.weight": "head.safetensors",
                    "model.language_model.norm.weight": "norm.safetensors",
                }
            }
        ).encode()

        def get(_session, url, **kwargs):
            data = source[url.rsplit("/", 1)[1]]
            byte_range = kwargs.get("headers", {}).get("Range")
            headers, status = {}, 200
            if byte_range:
                start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range).groups())
                headers, status = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, 206
                data = data[start : end + 1]
                if truncated and start > 8:
                    data = data[:-1]
            return TinyResponse(data, headers, status)

        monkeypatch.setattr(requests.Session, "get", get)
        return root


class TinyResponse:
    def __init__(self, data, headers, status):
        self.content, self.headers, self.status_code = data, headers, status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def json(self):
        return json.loads(self.content)

    def iter_content(self, chunk_size):
        yield self.content


def test_restoration_preserves_weights_index_and_exact_source_bytes(tmp_path, monkeypatch):
    root = TinyCheckpoint.setup(tmp_path, monkeypatch)
    index_before = (root / "model.safetensors.index.json").read_bytes()
    weights_before = (root / "0.safetensors").read_bytes()
    result = Restorer.restore(root.parent)
    assert Path(result["index_backup"]).read_bytes() == index_before
    assert (root / "0.safetensors").read_bytes() == weights_before
    weights, bits, _version = WeightLoader._try_load_mflux_format(root)
    assert bits is None
    assert weights["lm_head"]["weight"].dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(weights["lm_head"]["weight"].astype(mx.float32)), np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(
        np.array(weights["language_model"]["norm"]["weight"].astype(mx.float32)), np.arange(2)
    )
    for key, filename in (("lm_head.weight", "head.safetensors"), ("language_model.norm.weight", "norm.safetensors")):
        data = (tmp_path / filename).read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + length])
        start, end = header[Restorer.KEYS[key]]["data_offsets"]
        tensor_bytes = data[8 + length + start : 8 + length + end]
        assert result["tensors"][key]["sha256"] == hashlib.sha256(tensor_bytes).hexdigest()
    assert Restorer.restore(root.parent)["status"] == "already_present"


@pytest.mark.parametrize("option", ["wrong_shape", "truncated", "quantized"])
def test_invalid_source_or_quantization_leaves_original_checkpoint_intact(tmp_path, monkeypatch, option):
    root = TinyCheckpoint.setup(tmp_path, monkeypatch, **{option: True})
    index_before = (root / "model.safetensors.index.json").read_bytes()
    with pytest.raises(ValueError):
        Restorer.restore(root.parent)
    assert (root / "model.safetensors.index.json").read_bytes() == index_before
    assert not (root / "generation_head.safetensors").exists()
    assert len(list(root.glob("*.safetensors"))) == 1


def test_range_download_refuses_an_ignored_range():
    session = type("Session", (), {"get": lambda *_a, **_k: TinyResponse(b"x", {}, 200)})()
    with pytest.raises(ValueError, match="exact byte range"):
        Restorer._range_bytes(session, "https://example.test/weights", 0, 7)
