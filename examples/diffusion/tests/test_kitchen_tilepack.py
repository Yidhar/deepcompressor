from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepcompressor.backend.kitchen.tilepack import (  # noqa: E402
    KITCHEN_TILEPACK_LAYOUT_NAME,
    encode_comfy_quant,
    pack_int4_pairs,
    repack_safetensors,
    to_kitchen_tile_packed_params,
)


def _decode_comfy_quant(tensor: torch.Tensor) -> dict[str, object]:
    return json.loads(bytes(tensor.tolist()).decode("utf-8"))


def test_repack_safetensors_writes_kitchen_tilepack(tmp_path: Path) -> None:
    n, k, rank = 128, 128, 16
    dense = (torch.arange(n * k, dtype=torch.int16).remainder(16) - 8).view(n, k).to(torch.int8)
    natural = {
        "weight": pack_int4_pairs(dense),
        "weight_scale": torch.arange((k // 64) * n, dtype=torch.float32).view(k // 64, n).to(torch.float16),
        "smooth_factor": torch.arange(k, dtype=torch.float32).to(torch.float16),
        "proj_down": torch.arange(k * rank, dtype=torch.float32).view(k, rank).to(torch.float16),
        "proj_up": torch.arange(n * rank, dtype=torch.float32).view(n, rank).to(torch.float16),
        "bias": torch.arange(n, dtype=torch.float32).to(torch.float16),
    }
    state_dict = {f"layer.{key}": value for key, value in natural.items()}
    state_dict["layer.comfy_quant"] = encode_comfy_quant({"format": "svdquant_w4a4", "act_unsigned": True})

    src = tmp_path / "natural.safetensors"
    dst = tmp_path / "tilepack.safetensors"
    save_file(state_dict, str(src), metadata={"source": "unit"})

    count = repack_safetensors(src, dst, device="cpu", verbose=False)

    assert count == 1
    with safe_open(dst, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        tensors = {key: reader.get_tensor(key) for key in reader.keys()}

    assert metadata["source"] == "unit"
    assert metadata["svdquant_storage_layout"] == KITCHEN_TILEPACK_LAYOUT_NAME
    assert tuple(tensors["layer.weight"].shape) == (1, 2, 32, 128)
    assert tuple(tensors["layer.weight_scale"].shape) == (1, 2, 128)
    assert tuple(tensors["layer.proj_up"].shape) == (1, rank, 128)
    assert tuple(tensors["layer.bias"].shape) == (n,)

    comfy_quant = _decode_comfy_quant(tensors["layer.comfy_quant"])
    assert comfy_quant["format"] == "svdquant_w4a4"
    assert comfy_quant["layout"] == KITCHEN_TILEPACK_LAYOUT_NAME
    assert comfy_quant["act_unsigned"] is True

    expected = to_kitchen_tile_packed_params(natural)
    for key in ("weight", "weight_scale", "smooth_factor", "proj_down", "proj_up", "bias"):
        assert torch.equal(tensors[f"layer.{key}"], expected[key])
