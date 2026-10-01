import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url


class HereticGenerationHeadRestorer:
    REPO_ID = "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic"
    REVISION = "047e54342fc4bfcd2addd54049db9c90bb74731e"
    KEYS = {"lm_head.weight": "lm_head.weight", "language_model.norm.weight": "model.language_model.norm.weight"}

    @classmethod
    def restore(cls, model_path: Path) -> dict:
        root = model_path / "text_encoder"
        index_path = root / "model.safetensors.index.json"
        original_index = index_path.read_bytes()
        index = json.loads(original_index)
        missing = [key for key in cls.KEYS if key not in index["weight_map"]]
        if not missing:
            return {"status": "already_present", "model_path": str(model_path)}
        if index.get("metadata", {}).get("quantization_level") not in (None, "None", "null", ""):
            raise ValueError("Restore into the BF16 export, then quantize/save it; do not mix checkpoint precisions.")
        config = json.loads((root / "config.json").read_text())["text_config"]
        shapes = {
            "lm_head.weight": [config["vocab_size"], config["hidden_size"]],
            "language_model.norm.weight": [config["hidden_size"]],
        }
        target = root / "generation_head.safetensors"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite unindexed weights: {target}")
        headers, tensors, offset = {}, {}, 0
        with requests.Session() as session:
            url = hf_hub_url(cls.REPO_ID, "model.safetensors.index.json", revision=cls.REVISION)
            with session.get(url, timeout=(10, 60)) as response:
                response.raise_for_status()
                source_index = response.json()["weight_map"]
            for key in missing:
                source_key = cls.KEYS[key]
                filename = source_index[source_key]
                url = hf_hub_url(cls.REPO_ID, filename, revision=cls.REVISION)
                if filename not in headers:
                    prefix = cls._range_bytes(session, url, 0, 7)
                    length = struct.unpack("<Q", prefix)[0]
                    if not 0 < length <= 16 * 1024 * 1024:
                        raise ValueError("Invalid safetensors header length.")
                    headers[filename] = (length, json.loads(cls._range_bytes(session, url, 8, 7 + length)))
                length, header = headers[filename]
                tensor = header[source_key]
                begin, end = tensor["data_offsets"]
                if tensor["dtype"] != "BF16" or tensor["shape"] != shapes[key]:
                    raise ValueError(f"Source tensor {source_key} does not match the BF16 target config.")
                if end - begin != math.prod(shapes[key]) * 2:
                    raise ValueError(f"Invalid tensor byte length: {source_key}")
                tensors[key] = {"dtype": "BF16", "shape": shapes[key], "data_offsets": [offset, offset + end - begin]}
                offset += end - begin
            output_header = {"__metadata__": {k: str(v) for k, v in index.get("metadata", {}).items()}, **tensors}
            encoded = json.dumps(output_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            temporary = None
            provenance = {}
            try:
                with tempfile.NamedTemporaryFile(dir=root, suffix=".safetensors", delete=False) as output:
                    temporary = Path(output.name)
                    output.write(struct.pack("<Q", len(encoded)))
                    output.write(encoded)
                    for key in missing:
                        source_key = cls.KEYS[key]
                        filename = source_index[source_key]
                        length, header = headers[filename]
                        begin, end = header[source_key]["data_offsets"]
                        url = hf_hub_url(cls.REPO_ID, filename, revision=cls.REVISION)
                        digest, downloaded = hashlib.sha256(), 0
                        with cls._range_response(session, url, 8 + length + begin, 7 + length + end) as response:
                            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                                output.write(chunk)
                                digest.update(chunk)
                                downloaded += len(chunk)
                                if downloaded % (128 * 1024 * 1024) < len(chunk):
                                    print(f"Downloading {key}: {downloaded / 2**20:.0f} MiB", flush=True)
                        if downloaded != end - begin:
                            raise ValueError(f"Incomplete tensor download: {source_key}")
                        provenance[key] = {
                            "source_shard": filename,
                            "source_key": source_key,
                            "sha256": digest.hexdigest(),
                            "bytes": downloaded,
                            "shape": shapes[key],
                            "dtype": "BF16",
                        }
                        print(f"Restored {key}: {downloaded:,} bytes", flush=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                backup = index_path.with_name(f"model.safetensors.index.pre-head-{stamp}.json")
                if index_path.read_bytes() != original_index:
                    raise RuntimeError("The checkpoint index changed during download; no weights were installed.")
                shutil.copy2(index_path, backup)
                temporary.replace(target)
                index["weight_map"].update(dict.fromkeys(missing, target.name))
                if "total_size" in index.get("metadata", {}):
                    index["metadata"]["total_size"] += offset
                pending_index = index_path.with_suffix(".json.tmp")
                pending_index.write_text(json.dumps(index, indent=2) + "\n")
                pending_index.replace(index_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        result = {
            "status": "restored",
            "model_path": str(model_path),
            "source_repo": cls.REPO_ID,
            "source_revision": cls.REVISION,
            "index_backup": str(backup),
            "tensors": provenance,
        }
        target.with_suffix(".provenance.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    @staticmethod
    def _range_response(session: requests.Session, url: str, start: int, end: int):
        response = session.get(
            url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(10, 60),
        )
        expected = f"bytes {start}-{end}/"
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            response.close()
            raise ValueError("Source did not honor the exact byte range; refusing a full-shard download.")
        return response

    @classmethod
    def _range_bytes(cls, session: requests.Session, url: str, start: int, end: int) -> bytes:
        with cls._range_response(session, url, start, end) as response:
            data = response.content
        if len(data) != end - start + 1:
            raise ValueError("Incomplete safetensors header download.")
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the original Heretic generation head into a BF16 MLX export.")
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(HereticGenerationHeadRestorer.restore(args.model_path.expanduser().resolve()), indent=2))


if __name__ == "__main__":
    main()
