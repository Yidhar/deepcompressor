"""Kitchen SVDQuant W4A4 tile-packed checkpoint utilities.

The tile-packed layout is consumed by comfy-kitchen's CUDA SVDQuant W4A4
kernel. This module supports both direct conversion from a natural-layout
parameter set and repacking an existing kitchen-native safetensors checkpoint.
It is a layout-only transform: the logical signed int4 values and fp16/bf16
side tensors are preserved.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KITCHEN_BLOCK_N = 128
KITCHEN_GROUP_SIZE = 64
KITCHEN_INTERLEAVE = 4
KITCHEN_TILEPACK_LAYOUT_NAME = "kitchen_tile_packed_w4a4"

_SVDQUANT_FORMAT = "svdquant_w4a4"

ProgressCallback = Callable[[int, int, str], None]


def decode_comfy_quant(tensor: torch.Tensor | None) -> dict[str, object] | None:
    """Decode a `*.comfy_quant` JSON tensor."""
    if tensor is None:
        return None
    return json.loads(bytes(tensor.cpu().tolist()).decode("utf-8"))


def encode_comfy_quant(config: Mapping[str, object]) -> torch.Tensor:
    """Encode a `*.comfy_quant` JSON tensor."""
    return torch.tensor(list(json.dumps(dict(config)).encode("utf-8")), dtype=torch.uint8)


def is_svdquant_config(config: Mapping[str, object] | None) -> bool:
    return config is not None and config.get("format") == _SVDQUANT_FORMAT


def pack_int4_pairs(values: torch.Tensor) -> torch.Tensor:
    """Pack adjacent signed int4 values from the last dimension into int8 bytes."""
    lo = values[..., 0::2].to(torch.int32).bitwise_and(0x0F)
    hi = values[..., 1::2].to(torch.int32).bitwise_and(0x0F).bitwise_left_shift(4)
    return (lo | hi).to(torch.int8).contiguous()


def unpack_int4_pairs(packed: torch.Tensor) -> torch.Tensor:
    """Unpack row-major int4-pair bytes into signed int8 values in [-8, 7]."""
    x32 = packed.to(torch.int32)
    lo = x32.bitwise_and(0x0F)
    hi = x32.bitwise_right_shift(4).bitwise_and(0x0F)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).view(packed.shape[0], packed.shape[1] * 2).to(torch.int8)


def _validate_weight_tile_shape(weight: torch.Tensor) -> None:
    expected_tail = (KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE, KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE // 2)
    if weight.dim() != 4 or tuple(weight.shape[2:]) != expected_tail:
        raise ValueError(
            "expected tile-packed SVDQuant weight shape "
            f"(N/{KITCHEN_BLOCK_N}, K/{KITCHEN_GROUP_SIZE}, {expected_tail[0]}, {expected_tail[1]}), "
            f"got {tuple(weight.shape)}"
        )


def pack_weight_tile(weight: torch.Tensor) -> torch.Tensor:
    """Pack a natural `(N, K/2)` int8 W4A4 weight into kitchen tile storage.

    The output shape is `(N/128, K/64, 32, 128)`. If `weight` is already a
    4-D tile-packed tensor, only contiguity is normalized.
    """
    if weight.dim() == 4:
        _validate_weight_tile_shape(weight)
        return weight.contiguous()
    if weight.dim() != 2:
        raise ValueError(f"expected 2D natural or 4D tile-packed SVDQuant weight, got {tuple(weight.shape)}")

    n, k_half = weight.shape
    k = k_half * 2
    if n % KITCHEN_BLOCK_N != 0:
        raise ValueError(f"N={n} is not divisible by KITCHEN_BLOCK_N={KITCHEN_BLOCK_N}")
    if k % KITCHEN_GROUP_SIZE != 0:
        raise ValueError(f"K={k} is not divisible by KITCHEN_GROUP_SIZE={KITCHEN_GROUP_SIZE}")

    dense = unpack_int4_pairs(weight)
    tiled = dense.view(
        n // KITCHEN_BLOCK_N,
        KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE,
        KITCHEN_INTERLEAVE,
        k // KITCHEN_GROUP_SIZE,
        KITCHEN_GROUP_SIZE,
    ).permute(0, 3, 1, 2, 4).contiguous()
    return pack_int4_pairs(tiled).view(
        n // KITCHEN_BLOCK_N,
        k // KITCHEN_GROUP_SIZE,
        KITCHEN_BLOCK_N // KITCHEN_INTERLEAVE,
        KITCHEN_INTERLEAVE * KITCHEN_GROUP_SIZE // 2,
    )


def pack_n_axis(tensor: torch.Tensor) -> torch.Tensor:
    """Tile-pack the N axis of a natural `(N, *)` tensor to `(N/128, *, 128)`.

    Existing tile-packed tensors with rank >= 3 are returned as contiguous
    tensors. This is used for `weight_scale` after transpose and `proj_up`.
    """
    if tensor.dim() >= 3:
        return tensor.contiguous()

    n = tensor.shape[0]
    if n % KITCHEN_BLOCK_N != 0:
        raise ValueError(f"N={n} is not divisible by KITCHEN_BLOCK_N={KITCHEN_BLOCK_N}")
    return tensor.view(n // KITCHEN_BLOCK_N, KITCHEN_BLOCK_N, *tensor.shape[1:]).movedim(1, -1).contiguous()


def pack_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Pack `weight_scale` from natural `(K/64, N)` to `(N/128, K/64, 128)`."""
    if weight_scale.dim() == 3:
        return weight_scale.contiguous()
    if weight_scale.dim() != 2:
        raise ValueError(f"expected 2D natural or 3D tile-packed weight_scale, got {tuple(weight_scale.shape)}")
    return pack_n_axis(weight_scale.t().contiguous())


def patch_comfy_quant_layout(comfy_quant: torch.Tensor | None) -> torch.Tensor:
    """Return a `comfy_quant` tensor marked as kitchen tile-packed SVDQuant."""
    config = decode_comfy_quant(comfy_quant)
    if config is None:
        config = {"format": _SVDQUANT_FORMAT}
    config["format"] = _SVDQUANT_FORMAT
    config["layout"] = KITCHEN_TILEPACK_LAYOUT_NAME
    return encode_comfy_quant(config)


def to_kitchen_tile_packed_params(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert one natural-layout SVDQuant param set to kitchen tile-packed layout.

    Expected natural input keys and shapes:
      weight        `(N, K/2)` int8 packed signed int4
      weight_scale  `(K/64, N)` fp16/bf16
      smooth_factor `(K,)` fp16/bf16
      proj_down     `(K, R)` fp16/bf16
      proj_up       `(N, R)` fp16/bf16
      bias          `(N,)` fp16/bf16, optional

    `bias`, `smooth_factor`, and `proj_down` stay in their natural layouts.
    """
    out = {
        "weight": pack_weight_tile(params["weight"]),
        "weight_scale": pack_weight_scale(params["weight_scale"]),
        "smooth_factor": params["smooth_factor"].contiguous(),
        "proj_down": params["proj_down"].contiguous(),
        "proj_up": pack_n_axis(params["proj_up"]),
    }
    if "bias" in params:
        out["bias"] = params["bias"].contiguous()
    return out


def svdquant_prefixes(keys: set[str], tensors: Mapping[str, torch.Tensor]) -> list[str]:
    """Find SVDQuant W4A4 layer prefixes in a kitchen-native state dict."""
    prefixes: list[str] = []
    for key in keys:
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        config = decode_comfy_quant(tensors.get(f"{prefix}.comfy_quant"))
        if is_svdquant_config(config):
            prefixes.append(prefix)
    return sorted(prefixes)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _convert_on_device(
    tensor: torch.Tensor,
    device: torch.device,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    work = tensor.to(device=device) if tensor.device != device else tensor
    out = fn(work)
    return out.cpu() if out.device.type != "cpu" else out


def repack_state_dict(
    tensors: MutableMapping[str, torch.Tensor],
    *,
    device: str | torch.device = "auto",
    progress: ProgressCallback | None = None,
) -> list[str]:
    """Repack all SVDQuant W4A4 layers in `tensors` in place.

    Returns the list of repacked layer prefixes.
    """
    work_device = _resolve_device(device)
    prefixes = svdquant_prefixes(set(tensors.keys()), tensors)

    for index, prefix in enumerate(prefixes, 1):
        tensors[f"{prefix}.weight"] = _convert_on_device(
            tensors[f"{prefix}.weight"], work_device, pack_weight_tile
        )
        tensors[f"{prefix}.weight_scale"] = _convert_on_device(
            tensors[f"{prefix}.weight_scale"], work_device, pack_weight_scale
        )
        tensors[f"{prefix}.proj_up"] = _convert_on_device(
            tensors[f"{prefix}.proj_up"], work_device, pack_n_axis
        )
        tensors[f"{prefix}.comfy_quant"] = patch_comfy_quant_layout(tensors.get(f"{prefix}.comfy_quant"))
        if progress is not None:
            progress(index, len(prefixes), prefix)

    return prefixes


def repack_safetensors(
    input_path: Path,
    output_path: Path,
    *,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> int:
    """Repack a kitchen-native safetensors checkpoint into tile-packed storage."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    work_device = _resolve_device(device)

    if verbose:
        print(f"input : {input_path}")
        print(f"output: {output_path}")
        print(f"work device: {work_device}")

    with safe_open(input_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        keys = list(reader.keys())
        tensors = {key: reader.get_tensor(key) for key in keys}

    if verbose:
        prefixes = svdquant_prefixes(set(keys), tensors)
        print(f"SVDQuant W4A4 layers: {len(prefixes)}")

        def log_progress(index: int, total: int, _prefix: str) -> None:
            if index % 25 == 0 or index == total:
                print(f"  repacked {index}/{total}")

        repacked = repack_state_dict(tensors, device=work_device, progress=log_progress)
    else:
        repacked = repack_state_dict(tensors, device=work_device)

    new_metadata = dict(metadata)
    new_metadata["svdquant_storage_layout"] = KITCHEN_TILEPACK_LAYOUT_NAME

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=new_metadata)

    if verbose:
        print(f"wrote {output_path}")
    return len(repacked)
