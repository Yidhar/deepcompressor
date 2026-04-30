"""Convert a raw nunchaku Qwen-Image-Edit int4 checkpoint into the
comfy-kitchen-native layout consumed by ComfyUI's MixedPrecisionOps.Linear.

This is the production conversion entry point — combines two transformations:

  1. SVDQuant W4A4 (attention + MLP):
     layout-only repack from nunchaku tile-packed (qweight/wscales) into
     kitchen tile-packed (weight/weight_scale), with per-head QKV split
     (attn.to_qkv -> attn.to_q / .to_k / .to_v, similarly for add_qkv_proj).
     No dequantize / requantize — bit-exact int4 values preserved.

  2. AWQ W4A16 (img_mod.1 / txt_mod.1 modulation):
     layout repack from nunchaku TRT-LLM-style (N//4, K//2) int32 + zero-shift
     adjustment into kitchen (N, K//2) int8 + adjusted weight_zero. Stays at
     int4 — the previous v3 converter dequantized these to bf16, inflating
     the checkpoint by ~10 GB on r96 (modulation 13.6 GB → ~3.4 GB after this
     change).

Companion runtime: comfy-kitchen `feat/awq-w4a16-modulation` (provides
TensorCoreAWQW4A16Layout) + ComfyUI `feat/comfykit-awq-w4a16-modulation`
(adds `awq_w4a16` quant_format branch in MixedPrecisionOps.Linear).

Usage:
  python -m examples.diffusion.scripts.convert_kitchen_native \\
    --raw-nunchaku /path/to/nunchaku_quality_r96_int4.safetensors \\
    --base-comfy   /path/to/r96_kitchen_native.safetensors \\
    --output       /path/to/r96_kitchen_native_awq.safetensors

Importing nunchaku helpers:
  This script reuses unpack/pack helpers that live in
  ``nunchaku/tools/kitchen_native/`` (interop.py + awq_modulation.py).
  Either install nunchaku or add /path/to/nunchaku to PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from deepcompressor.backend.kitchen.tilepack import (
    KITCHEN_TILEPACK_LAYOUT_NAME,
    to_kitchen_tile_packed_params,
)

# Auto-discover nunchaku helpers if the user has the repo cloned alongside.
for _candidate in (
    os.environ.get("NUNCHAKU_REPO_DIR"),
    "/workspace/nunchaku",
    str(Path.home() / "nunchaku"),
):
    if _candidate and (Path(_candidate) / "tools" / "kitchen_native").is_dir():
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

from tools.kitchen_native.awq_modulation import (  # type: ignore  # noqa: E402
    convert_modulation_awq,
)
from tools.kitchen_native.interop import (  # type: ignore  # noqa: E402
    convert_nunchaku_svdquant_params,
    split_natural_svdquant_params,
)

_W4A4_RAW_SUFFIXES = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up")
_AWQ_RAW_SUFFIXES = ("qweight", "wscales", "wzeros")
_QKV_SPLIT_TARGETS = {
    "attn.to_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn.add_qkv_proj": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
_AWQ_MODULATION_SUFFIXES = (".img_mod.1", ".txt_mod.1")
_ACT_UNSIGNED_SUFFIXES = (".img_mlp.net.2", ".txt_mlp.net.2", ".ff.net.2")

# --- helpers --------------------------------------------------------------

def _prefixes(keys: list[str], suffixes: tuple[str, ...]) -> set[str]:
    wanted = set(keys)
    out = set()
    for key in keys:
        prefix, _, _ = key.rpartition(".")
        if prefix and all(f"{prefix}.{suffix}" in wanted for suffix in suffixes):
            out.add(prefix)
    return out


def _drop_prefix_tensors(out: dict[str, torch.Tensor], prefix: str) -> None:
    for key in [k for k in out if k.startswith(f"{prefix}.")]:
        out.pop(key)


def _qkv_split_prefixes(prefix: str) -> tuple[str, str, str] | None:
    for suffix, targets in _QKV_SPLIT_TARGETS.items():
        if prefix.endswith(suffix):
            stem = prefix[: -len(suffix)]
            return tuple(f"{stem}{t}" for t in targets)
    return None


def _is_act_unsigned(prefix: str) -> bool:
    return any(prefix.endswith(s) for s in _ACT_UNSIGNED_SUFFIXES)


def _is_awq_modulation(prefix: str) -> bool:
    return any(prefix.endswith(s) for s in _AWQ_MODULATION_SUFFIXES)


def _patch_comfy_quant(existing: torch.Tensor | None, *, fmt: str, **extra) -> torch.Tensor:
    if existing is None:
        conf: dict[str, object] = {"format": fmt}
    else:
        conf = json.loads(bytes(existing.tolist()).decode("utf-8"))
        conf["format"] = fmt
    for k, v in extra.items():
        if v is None or v is False:
            conf.pop(k, None)
        else:
            conf[k] = v
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _equal_qkv_split_sizes(total_out: int) -> tuple[int, int, int]:
    if total_out % 3 != 0:
        raise ValueError(f"fused-QKV out_features {total_out} not divisible by 3")
    chunk = total_out // 3
    return (chunk, chunk, chunk)


def _write_svdquant(
    out: dict[str, torch.Tensor],
    *,
    prefix: str,
    params: dict[str, torch.Tensor],
    comfy_quant: torch.Tensor | None,
) -> None:
    """Write an SVDQuant W4A4 layer to the output state dict.

    `params` is in natural (N, K//2) layout; this re-packs into the kitchen
    tile-packed layout (BLOCK_N=128, kInterleave=4, WARP_K=64) consumed by
    the CUDA kernel in comfy_kitchen/backends/cuda/ops/svdquant_w4a4_native/.
    """
    tp = to_kitchen_tile_packed_params(params)
    out[f"{prefix}.weight"] = tp["weight"].cpu()
    out[f"{prefix}.weight_scale"] = tp["weight_scale"].cpu()
    out[f"{prefix}.smooth_factor"] = tp["smooth_factor"].cpu()
    out[f"{prefix}.proj_down"] = tp["proj_down"].cpu()
    out[f"{prefix}.proj_up"] = tp["proj_up"].cpu()
    if "bias" in tp:
        out[f"{prefix}.bias"] = tp["bias"].cpu()
    out[f"{prefix}.comfy_quant"] = _patch_comfy_quant(
        comfy_quant, fmt="svdquant_w4a4",
        layout=KITCHEN_TILEPACK_LAYOUT_NAME,
        act_unsigned=True if _is_act_unsigned(prefix) else None,
    )


def _write_awq(
    out: dict[str, torch.Tensor],
    *,
    prefix: str,
    params: dict[str, torch.Tensor],
    bias: torch.Tensor | None,
    comfy_quant: torch.Tensor | None,
    group_size: int,
) -> None:
    out[f"{prefix}.weight"] = params["weight"].cpu()
    out[f"{prefix}.weight_scale"] = params["weight_scale"].cpu()
    out[f"{prefix}.weight_zero"] = params["weight_zero"].cpu()
    if bias is not None:
        out[f"{prefix}.bias"] = bias.cpu()
    out[f"{prefix}.comfy_quant"] = _patch_comfy_quant(
        comfy_quant, fmt="awq_w4a16", group_size=group_size,
    )


# --- main --------------------------------------------------------------------

def convert(raw_nunchaku: Path, base_comfy: Path, output: Path,
            *, awq_modulation: bool = True, awq_group_size: int = 64) -> dict:
    print(f"raw  : {raw_nunchaku}")
    print(f"base : {base_comfy}")
    print(f"out  : {output}")
    print(f"awq_modulation passthrough: {awq_modulation}")

    # Step 1: copy the base scaffold into memory.
    with safe_open(base_comfy, framework="pt", device="cpu") as r:
        base_meta = r.metadata() or {}
        base_keys = list(r.keys())
        out: dict[str, torch.Tensor] = {}
        for i, k in enumerate(base_keys, 1):
            out[k] = r.get_tensor(k)
            if i % 500 == 0 or i == len(base_keys):
                print(f"  loaded base tensors: {i}/{len(base_keys)}")

    stats = {"svdquant_patched": 0, "qkv_split_emitted": 0, "awq_patched": 0, "awq_skipped": 0}

    # Step 2: patch from raw nunchaku.
    with safe_open(raw_nunchaku, framework="pt", device="cpu") as r:
        raw_keys = list(r.keys())
        w4a4_prefixes = _prefixes(raw_keys, _W4A4_RAW_SUFFIXES)
        awq_prefixes  = _prefixes(raw_keys, _AWQ_RAW_SUFFIXES)
        # AWQ prefixes that are also W4A4 prefixes: shouldn't happen for Qwen
        # (W4A4 and AWQ layers are disjoint), but disambiguate just in case.
        awq_prefixes = {p for p in awq_prefixes if p not in w4a4_prefixes}

        # 2a. SVDQuant W4A4 layers (with optional QKV split).
        for prefix in sorted(w4a4_prefixes):
            split = _qkv_split_prefixes(prefix)
            converted = convert_nunchaku_svdquant_params(
                qweight=r.get_tensor(f"{prefix}.qweight"),
                wscales=r.get_tensor(f"{prefix}.wscales"),
                smooth_factor=r.get_tensor(f"{prefix}.smooth_factor"),
                proj_down=r.get_tensor(f"{prefix}.proj_down"),
                proj_up=r.get_tensor(f"{prefix}.proj_up"),
                bias=r.get_tensor(f"{prefix}.bias") if f"{prefix}.bias" in raw_keys else None,
            )
            cq = out.get(f"{prefix}.comfy_quant")
            if split is None:
                _write_svdquant(out, prefix=prefix, params=converted, comfy_quant=cq)
            else:
                _drop_prefix_tensors(out, prefix)
                for sp in split:
                    _drop_prefix_tensors(out, sp)
                split_params = split_natural_svdquant_params(
                    converted, _equal_qkv_split_sizes(converted["weight"].shape[0]),
                )
                for sp, p in zip(split, split_params, strict=True):
                    _write_svdquant(out, prefix=sp, params=p, comfy_quant=cq)
                stats["qkv_split_emitted"] += len(split)
            stats["svdquant_patched"] += 1

        # 2b. AWQ W4A16 modulation layers — only if requested.
        if awq_modulation:
            for prefix in sorted(awq_prefixes):
                if not _is_awq_modulation(prefix):
                    stats["awq_skipped"] += 1
                    continue
                qw = r.get_tensor(f"{prefix}.qweight")
                ws = r.get_tensor(f"{prefix}.wscales")
                wz = r.get_tensor(f"{prefix}.wzeros")
                bias = r.get_tensor(f"{prefix}.bias") if f"{prefix}.bias" in raw_keys else None
                # N comes from wscales shape (K//G, N).
                n_orig = ws.shape[1]
                params = convert_modulation_awq(
                    qweight=qw, wscales=ws, wzeros=wz,
                    n_orig=n_orig, group_size=awq_group_size,
                )
                # nunchaku stores modulation rows as [dim, 6] interleaved
                # (shift1, scale1, gate1, shift2, scale2, gate2 packed per
                # channel); ComfyUI's QwenImageTransformerBlock chunks the
                # output as [6, dim], so transpose the leading 2 axes for
                # weight rows and bias. Mirrors `_reorder_modulation` in
                # nunchaku/tools/convert_nunchaku_qwen_to_comfy.py.
                assert n_orig % 6 == 0, f"modulation N={n_orig} not divisible by 6"
                dim = n_orig // 6
                params["weight"] = (
                    params["weight"].view(dim, 6, -1).transpose(0, 1)
                    .reshape(n_orig, -1).contiguous()
                )
                params["weight_scale"] = (
                    params["weight_scale"].view(-1, dim, 6).transpose(1, 2)
                    .reshape(-1, n_orig).contiguous()
                )
                params["weight_zero"] = (
                    params["weight_zero"].view(-1, dim, 6).transpose(1, 2)
                    .reshape(-1, n_orig).contiguous()
                )
                if bias is not None:
                    bias = bias.view(dim, 6).transpose(0, 1).reshape(n_orig).contiguous()
                cq = out.get(f"{prefix}.comfy_quant")
                _drop_prefix_tensors(out, prefix)
                _write_awq(out, prefix=prefix, params=params, bias=bias,
                           comfy_quant=cq, group_size=awq_group_size)
                stats["awq_patched"] += 1

    print(f"  svdquant patched: {stats['svdquant_patched']} "
          f"(QKV splits: {stats['qkv_split_emitted']})")
    print(f"  awq modulation patched: {stats['awq_patched']}, "
          f"skipped non-modulation AWQ: {stats['awq_skipped']}")

    # Step 3: meta + write.
    new_meta = dict(base_meta)
    suffix = ("comfy-kitchen qwen-edit kitchen-native split-qkv v3 "
              "(svdquant_w4a4 + awq_w4a16 modulation)")
    existing = new_meta.get("comfy_converter", "")
    new_meta["comfy_converter"] = f"{existing} + {suffix}" if existing else suffix
    new_meta["source_nunchaku"] = raw_nunchaku.name
    new_meta["source_base_comfy"] = base_comfy.name
    new_meta["svdquant_storage_layout"] = KITCHEN_TILEPACK_LAYOUT_NAME
    new_meta["awq_modulation_layout"] = (
        "kitchen-native-uint4-row-major" if awq_modulation else "dequant-bf16"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {output} ...")
    save_file(out, str(output), metadata=new_meta)
    print(f"  wrote {output.stat().st_size / 1e9:.2f} GB")
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--raw-nunchaku", required=True, type=Path,
                   help="raw nunchaku-format SVDQuant W4A4 + AWQ W4A16 safetensors")
    p.add_argument("--base-comfy", required=True, type=Path,
                   help="kitchen-native scaffold (used for non-quantized tensors + topology)")
    p.add_argument("--output", required=True, type=Path,
                   help="destination kitchen-native safetensors path")
    p.add_argument("--no-awq-modulation", action="store_true",
                   help="dequantize modulation to bf16 (legacy v2 behavior, +10 GB)")
    p.add_argument("--awq-group-size", type=int, default=64,
                   help="K-axis group size for AWQ modulation (default 64)")
    args = p.parse_args(argv)

    if not args.raw_nunchaku.is_file():
        raise FileNotFoundError(args.raw_nunchaku)
    if not args.base_comfy.is_file():
        raise FileNotFoundError(args.base_comfy)

    convert(args.raw_nunchaku, args.base_comfy, args.output,
            awq_modulation=not args.no_awq_modulation,
            awq_group_size=args.awq_group_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
