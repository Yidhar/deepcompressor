#!/usr/bin/env python3
"""Merge a diffusers-style LoRA into a quantized Nunchaku Qwen-Image-Edit-2511 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import safetensors
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepcompressor.backend.nunchaku.convert_lora import reorder_adanorm_lora_up
from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker
from deepcompressor.backend.utils import pad


SVDQ_MERGE_MAP: dict[str, list[list[str]]] = {
    "attn.to_qkv": [["attn.to_q"], ["attn.to_k"], ["attn.to_v"]],
    "attn.add_qkv_proj": [["attn.add_q_proj"], ["attn.add_k_proj"], ["attn.add_v_proj"]],
    "attn.to_out.0": [["attn.to_out.0"]],
    "attn.to_add_out": [["attn.to_add_out"]],
    "img_mlp.net.0.proj": [["img_mlp.net.0.proj"]],
    "img_mlp.net.2": [["img_mlp.net.2", "img_mlp.net.2.linear"]],
    "txt_mlp.net.0.proj": [["txt_mlp.net.0.proj"]],
    "txt_mlp.net.2": [["txt_mlp.net.2", "txt_mlp.net.2.linear"]],
}

MODULATION_MERGE_MAP: dict[str, list[list[str]]] = {
    "img_mod.1": [["img_mod.1", "img_mod.linear", "img_mod.lin"]],
    "txt_mod.1": [["txt_mod.1", "txt_mod.linear", "txt_mod.lin"]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=Path, required=True, help="Path to the quantized Nunchaku safetensors")
    parser.add_argument("--lora-path", type=Path, required=True, help="Path to the diffusers LoRA safetensors")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to the merged Nunchaku safetensors")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Additional LoRA scaling multiplier")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Override dtype for the merged low-rank tensors",
    )
    return parser.parse_args()



def load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    return state_dict, metadata



def infer_dtype(args: argparse.Namespace, quant_state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    for key in quant_state_dict:
        if key.endswith("proj_down"):
            return quant_state_dict[key].dtype
    return torch.bfloat16



def get_lora_pair(
    lora_state_dict: dict[str, torch.Tensor],
    base_name: str,
    target_dtype: torch.dtype,
    lora_scale: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    down_key = None
    up_key = None
    for candidate_down, candidate_up in (
        (f"{base_name}.lora_down.weight", f"{base_name}.lora_up.weight"),
        (f"{base_name}.lora_A.weight", f"{base_name}.lora_B.weight"),
    ):
        if candidate_down in lora_state_dict and candidate_up in lora_state_dict:
            down_key = candidate_down
            up_key = candidate_up
            break
    if down_key is None or up_key is None:
        return None

    down = lora_state_dict[down_key].to(dtype=target_dtype)
    up = lora_state_dict[up_key].to(dtype=target_dtype)
    alpha = lora_state_dict.get(f"{base_name}.alpha", None)
    alpha_value = float(alpha.item()) if alpha is not None else float(down.shape[0])
    scaled_up = up.mul(alpha_value / max(down.shape[0], 1) * lora_scale)
    return down, scaled_up



def resolve_candidate_loras(
    block_name: str,
    candidate_groups: list[list[str]],
    lora_state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    lora_scale: float,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    resolved: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for candidate_group in candidate_groups:
        pair = None
        for candidate in candidate_group:
            pair = get_lora_pair(
                lora_state_dict=lora_state_dict,
                base_name=f"{block_name}.{candidate}",
                target_dtype=target_dtype,
                lora_scale=lora_scale,
            )
            if pair is not None:
                break
        resolved.append(pair)
    return resolved



def merge_candidate_loras(
    lora_list: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    present = [pair for pair in lora_list if pair is not None]
    if not present:
        return None
    if len(lora_list) == 1:
        return present[0]

    filled = list(lora_list)
    first_down, first_up = present[0]
    for index, pair in enumerate(filled):
        if pair is None:
            filled[index] = (first_down.clone(), torch.zeros_like(first_up))

    assert all(pair is not None for pair in filled)
    merged_pairs = [pair for pair in filled if pair is not None]
    if all(pair[0].equal(merged_pairs[0][0]) for pair in merged_pairs):
        merged_down = merged_pairs[0][0]
        merged_up = torch.cat([pair[1] for pair in merged_pairs], dim=0)
        return merged_down, merged_up

    merged_down = torch.cat([pair[0] for pair in merged_pairs], dim=0)
    out_features = sum(pair[1].shape[0] for pair in merged_pairs)
    total_rank = sum(pair[1].shape[1] for pair in merged_pairs)
    merged_up = torch.zeros((out_features, total_rank), dtype=merged_down.dtype)
    out_offset = 0
    rank_offset = 0
    for down, up in merged_pairs:
        next_out = out_offset + up.shape[0]
        next_rank = rank_offset + up.shape[1]
        merged_up[out_offset:next_out, rank_offset:next_rank] = up
        out_offset = next_out
        rank_offset = next_rank
    return merged_down, merged_up



def load_existing_linear_branch(
    quant_state_dict: dict[str, torch.Tensor],
    prefix: str,
    packer: NunchakuWeightPacker,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    down_key = f"{prefix}.proj_down"
    up_key = f"{prefix}.proj_up"
    if down_key not in quant_state_dict or up_key not in quant_state_dict:
        return None
    down = packer.unpack_lowrank_weight(quant_state_dict[down_key], down=True).to(dtype=target_dtype)
    up = packer.unpack_lowrank_weight(quant_state_dict[up_key], down=False).to(dtype=target_dtype)
    return down, up



def store_linear_branch(
    quant_state_dict: dict[str, torch.Tensor],
    prefix: str,
    branch: tuple[torch.Tensor, torch.Tensor],
    packer: NunchakuWeightPacker,
):
    down, up = branch
    quant_state_dict[f"{prefix}.proj_down"] = packer.pack_lowrank_weight(
        packer.pad_lowrank_weight(down, down=True), down=True
    )
    quant_state_dict[f"{prefix}.proj_up"] = packer.pack_lowrank_weight(
        packer.pad_lowrank_weight(up, down=False), down=False
    )



def load_existing_modulation_branch(
    quant_state_dict: dict[str, torch.Tensor],
    prefix: str,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    down = quant_state_dict.get(f"{prefix}.lora_down")
    up = quant_state_dict.get(f"{prefix}.lora_up")
    if down is None or up is None or down.numel() == 0 or up.numel() == 0:
        return None
    return down.to(dtype=target_dtype), up.to(dtype=target_dtype)



def store_modulation_branch(
    quant_state_dict: dict[str, torch.Tensor],
    prefix: str,
    branch: tuple[torch.Tensor, torch.Tensor],
):
    down, up = branch
    quant_state_dict[f"{prefix}.lora_down"] = pad(down, divisor=16, dim=0)
    quant_state_dict[f"{prefix}.lora_up"] = pad(up, divisor=16, dim=1)



def merge_branches(
    existing: tuple[torch.Tensor, torch.Tensor] | None,
    extra: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if existing is None:
        return extra
    if extra is None:
        return existing
    return torch.cat([existing[0], extra[0]], dim=0), torch.cat([existing[1], extra[1]], dim=1)



def block_names_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[str]:
    blocks = {".".join(key.split(".")[:2]) for key in state_dict if key.startswith("transformer_blocks.")}
    return sorted(blocks, key=lambda name: int(name.split(".")[-1]))



def main() -> None:
    args = parse_args()
    quant_state_dict, metadata = load_safetensors(args.quant_path)
    lora_state_dict, _ = load_safetensors(args.lora_path)
    target_dtype = infer_dtype(args, quant_state_dict)
    packer = NunchakuWeightPacker(bits=4)

    max_rank = 0
    linear_merges = 0
    modulation_merges = 0
    blocks = block_names_from_state_dict(quant_state_dict)
    for block_name in blocks:
        for local_name, candidate_groups in SVDQ_MERGE_MAP.items():
            extra = merge_candidate_loras(
                resolve_candidate_loras(block_name, candidate_groups, lora_state_dict, target_dtype, args.lora_scale)
            )
            if extra is None:
                continue
            prefix = f"{block_name}.{local_name}"
            existing = load_existing_linear_branch(quant_state_dict, prefix, packer, target_dtype)
            merged = merge_branches(existing, extra)
            assert merged is not None
            store_linear_branch(quant_state_dict, prefix, merged, packer)
            max_rank = max(max_rank, merged[0].shape[0])
            linear_merges += 1
            print(f"Merged linear LoRA into {prefix} -> rank {merged[0].shape[0]}")

        for local_name, candidate_groups in MODULATION_MERGE_MAP.items():
            extra = merge_candidate_loras(
                resolve_candidate_loras(block_name, candidate_groups, lora_state_dict, target_dtype, args.lora_scale)
            )
            if extra is None:
                continue
            prefix = f"{block_name}.{local_name}"
            extra = (extra[0], reorder_adanorm_lora_up(extra[1], splits=6))
            existing = load_existing_modulation_branch(quant_state_dict, prefix, target_dtype)
            merged = merge_branches(existing, extra)
            assert merged is not None
            store_modulation_branch(quant_state_dict, prefix, merged)
            max_rank = max(max_rank, merged[0].shape[0])
            modulation_merges += 1
            print(f"Merged modulation LoRA into {prefix} -> rank {merged[0].shape[0]}")

    if linear_merges == 0 and modulation_merges == 0:
        raise RuntimeError("No matching LoRA modules were found for the quantized Qwen checkpoint")

    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
    if max_rank > 0:
        quantization_config["rank"] = max(max_rank, int(quantization_config.get("rank", 0) or 0))
    metadata = dict(metadata)
    metadata["quantization_config"] = json.dumps(quantization_config)
    metadata["merged_lora_path"] = str(args.lora_path)
    metadata["merged_lora_scale"] = str(args.lora_scale)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(quant_state_dict, str(args.output_path), metadata=metadata)
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "linear_merges": linear_merges,
                "modulation_merges": modulation_merges,
                "max_rank": max_rank,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
