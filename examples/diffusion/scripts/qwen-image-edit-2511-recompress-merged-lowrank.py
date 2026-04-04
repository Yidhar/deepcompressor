#!/usr/bin/env python3
"""Recompress merged low-rank branches inside a Nunchaku Qwen-Image-Edit-2511 int4 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import safetensors
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker
from deepcompressor.backend.utils import pad


LINEAR_FAMILY_MAP: dict[str, str] = {
    "attn.to_qkv": "qkv",
    "attn.add_qkv_proj": "qkv",
    "attn.to_out.0": "attn_out",
    "attn.to_add_out": "attn_out",
    "img_mlp.net.0.proj": "mlp",
    "img_mlp.net.2": "mlp",
    "txt_mlp.net.0.proj": "mlp",
    "txt_mlp.net.2": "mlp",
}

MODULATION_FAMILY_MAP: dict[str, str] = {
    "img_mod.1": "mod",
    "txt_mod.1": "mod",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="Merged Nunchaku int4 checkpoint")
    parser.add_argument("--output-path", type=Path, required=True, help="Recompressed output checkpoint")
    parser.add_argument("--device", default="cpu", help="Compute device for QR/SVD, e.g. cpu or cuda:0")
    parser.add_argument(
        "--compute-dtype",
        choices=["fp32", "fp64"],
        default="fp32",
        help="Temporary dtype used during QR/SVD",
    )
    parser.add_argument("--qkv-rank", type=int, default=128, help="Target rank for attn.to_qkv/add_qkv_proj")
    parser.add_argument("--attn-out-rank", type=int, default=64, help="Target rank for attn.to_out/to_add_out")
    parser.add_argument("--mlp-rank", type=int, default=64, help="Target rank for img/txt MLP branches")
    parser.add_argument(
        "--mod-rank",
        type=int,
        default=0,
        help="Target rank for modulation LoRA branches; <=0 keeps the current rank",
    )
    parser.add_argument(
        "--min-rank-multiple",
        type=int,
        default=16,
        help="Round target ranks down to a multiple of this value for packer compatibility",
    )
    return parser.parse_args()


def load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    return state_dict, metadata


def block_names_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[str]:
    blocks = {".".join(key.split(".")[:2]) for key in state_dict if key.startswith("transformer_blocks.")}
    return sorted(blocks, key=lambda name: int(name.split(".")[-1]))


def local_name(prefix: str) -> str:
    return prefix.split(".", 2)[2]


def compute_dtype_from_arg(name: str) -> torch.dtype:
    return torch.float64 if name == "fp64" else torch.float32


def normalized_rank(rank: int, max_rank: int, multiple: int) -> int:
    if rank <= 0:
        return max_rank
    rank = min(rank, max_rank)
    rank = max(multiple, (rank // multiple) * multiple)
    return min(rank, max_rank)


def target_rank_for_prefix(prefix: str, args: argparse.Namespace, current_rank: int) -> int:
    lname = local_name(prefix)
    family = LINEAR_FAMILY_MAP.get(lname)
    if family == "qkv":
        return normalized_rank(args.qkv_rank, current_rank, args.min_rank_multiple)
    if family == "attn_out":
        return normalized_rank(args.attn_out_rank, current_rank, args.min_rank_multiple)
    if family == "mlp":
        return normalized_rank(args.mlp_rank, current_rank, args.min_rank_multiple)
    if MODULATION_FAMILY_MAP.get(lname) == "mod":
        if args.mod_rank > 0:
            return normalized_rank(args.mod_rank, current_rank, args.min_rank_multiple)
        return current_rank
    return current_rank


def load_existing_linear_branch(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    packer: NunchakuWeightPacker,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    down_key = f"{prefix}.proj_down"
    up_key = f"{prefix}.proj_up"
    if down_key not in state_dict or up_key not in state_dict:
        return None
    down = packer.unpack_lowrank_weight(state_dict[down_key], down=True)
    up = packer.unpack_lowrank_weight(state_dict[up_key], down=False)
    return down, up


def store_linear_branch(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    down: torch.Tensor,
    up: torch.Tensor,
    packer: NunchakuWeightPacker,
) -> None:
    state_dict[f"{prefix}.proj_down"] = packer.pack_lowrank_weight(
        packer.pad_lowrank_weight(down, down=True), down=True
    )
    state_dict[f"{prefix}.proj_up"] = packer.pack_lowrank_weight(
        packer.pad_lowrank_weight(up, down=False), down=False
    )


def load_existing_modulation_branch(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    down = state_dict.get(f"{prefix}.lora_down")
    up = state_dict.get(f"{prefix}.lora_up")
    if down is None or up is None or down.numel() == 0 or up.numel() == 0:
        return None
    return down, up


def store_modulation_branch(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    down: torch.Tensor,
    up: torch.Tensor,
) -> None:
    state_dict[f"{prefix}.lora_down"] = pad(down, divisor=16, dim=0)
    state_dict[f"{prefix}.lora_up"] = pad(up, divisor=16, dim=1)


def recompress_lowrank_pair(
    down: torch.Tensor,
    up: torch.Tensor,
    target_rank: int,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    current_rank = min(down.shape[0], up.shape[1])
    if target_rank >= current_rank:
        return down, up, {
            "old_rank": current_rank,
            "new_rank": current_rank,
            "discarded_energy_ratio": 0.0,
            "retained_energy_ratio": 1.0,
        }

    up_work = up.to(device=device, dtype=compute_dtype)
    down_work = down.to(device=device, dtype=compute_dtype)

    q_up, r_up = torch.linalg.qr(up_work, mode="reduced")
    q_down, r_down = torch.linalg.qr(down_work.transpose(0, 1), mode="reduced")
    core = r_up @ r_down.transpose(0, 1)
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

    keep = target_rank
    sqrt_s = singular_values[:keep].clamp_min(0).sqrt()
    up_new = (q_up @ u_core[:, :keep]) * sqrt_s.unsqueeze(0)
    down_basis = vh_core[:keep, :] @ q_down.transpose(0, 1)
    down_new = sqrt_s.unsqueeze(1) * down_basis

    total_energy = float(singular_values.square().sum().item())
    kept_energy = float(singular_values[:keep].square().sum().item())
    retained = kept_energy / total_energy if total_energy > 0 else 1.0

    up_new = up_new.to(dtype=up.dtype, device="cpu").contiguous()
    down_new = down_new.to(dtype=down.dtype, device="cpu").contiguous()

    del up_work, down_work, q_up, r_up, q_down, r_down, core, u_core, singular_values, vh_core, sqrt_s, down_basis
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return down_new, up_new, {
        "old_rank": current_rank,
        "new_rank": keep,
        "discarded_energy_ratio": 1.0 - retained,
        "retained_energy_ratio": retained,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    compute_dtype = compute_dtype_from_arg(args.compute_dtype)
    packer = NunchakuWeightPacker(bits=4)

    state_dict, metadata = load_safetensors(args.input_path)
    blocks = block_names_from_state_dict(state_dict)

    by_family = Counter()
    by_family_old_rank = Counter()
    by_family_new_rank = Counter()
    energy_by_family: dict[str, list[float]] = defaultdict(list)
    changed = 0
    max_rank = 0

    for block_name in blocks:
        for lname in sorted(LINEAR_FAMILY_MAP):
            prefix = f"{block_name}.{lname}"
            branch = load_existing_linear_branch(state_dict, prefix, packer)
            if branch is None:
                continue
            down, up = branch
            current_rank = min(down.shape[0], up.shape[1])
            target_rank = target_rank_for_prefix(prefix, args, current_rank)
            new_down, new_up, stats = recompress_lowrank_pair(
                down=down,
                up=up,
                target_rank=target_rank,
                compute_dtype=compute_dtype,
                device=device,
            )
            family = LINEAR_FAMILY_MAP[lname]
            by_family[family] += 1
            by_family_old_rank[family] += int(stats["old_rank"])
            by_family_new_rank[family] += int(stats["new_rank"])
            energy_by_family[family].append(float(stats["discarded_energy_ratio"]))
            max_rank = max(max_rank, int(stats["new_rank"]))
            if int(stats["new_rank"]) != int(stats["old_rank"]):
                changed += 1
            store_linear_branch(state_dict, prefix, new_down, new_up, packer)
            print(
                f"{prefix}: rank {int(stats['old_rank'])} -> {int(stats['new_rank'])}, "
                f"discarded_energy={float(stats['discarded_energy_ratio']):.6f}"
            )

        for lname in sorted(MODULATION_FAMILY_MAP):
            prefix = f"{block_name}.{lname}"
            branch = load_existing_modulation_branch(state_dict, prefix)
            if branch is None:
                continue
            down, up = branch
            current_rank = min(down.shape[0], up.shape[1])
            target_rank = target_rank_for_prefix(prefix, args, current_rank)
            new_down, new_up, stats = recompress_lowrank_pair(
                down=down,
                up=up,
                target_rank=target_rank,
                compute_dtype=compute_dtype,
                device=device,
            )
            family = MODULATION_FAMILY_MAP[lname]
            by_family[family] += 1
            by_family_old_rank[family] += int(stats["old_rank"])
            by_family_new_rank[family] += int(stats["new_rank"])
            energy_by_family[family].append(float(stats["discarded_energy_ratio"]))
            max_rank = max(max_rank, int(stats["new_rank"]))
            if int(stats["new_rank"]) != int(stats["old_rank"]):
                changed += 1
            store_modulation_branch(state_dict, prefix, new_down, new_up)
            print(
                f"{prefix}: rank {int(stats['old_rank'])} -> {int(stats['new_rank'])}, "
                f"discarded_energy={float(stats['discarded_energy_ratio']):.6f}"
            )

    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
    quantization_config["rank"] = max_rank

    budget = {
        "qkv_rank": args.qkv_rank,
        "attn_out_rank": args.attn_out_rank,
        "mlp_rank": args.mlp_rank,
        "mod_rank": args.mod_rank,
        "min_rank_multiple": args.min_rank_multiple,
    }
    family_summary = {}
    for family, count in sorted(by_family.items()):
        avg_old_rank = by_family_old_rank[family] / max(count, 1)
        avg_new_rank = by_family_new_rank[family] / max(count, 1)
        family_summary[family] = {
            "count": count,
            "avg_old_rank": avg_old_rank,
            "avg_new_rank": avg_new_rank,
            "avg_discarded_energy_ratio": sum(energy_by_family[family]) / max(len(energy_by_family[family]), 1),
            "max_discarded_energy_ratio": max(energy_by_family[family], default=0.0),
        }

    metadata = dict(metadata)
    metadata["quantization_config"] = json.dumps(quantization_config)
    metadata["recompressed_from"] = str(args.input_path)
    metadata["recompress_method"] = "exact-small-core-svd"
    metadata["recompress_budget"] = json.dumps(budget, sort_keys=True)
    metadata["recompress_summary"] = json.dumps(family_summary, sort_keys=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state_dict, str(args.output_path), metadata=metadata)

    summary = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "changed_branches": changed,
        "max_rank": max_rank,
        "budget": budget,
        "family_summary": family_summary,
        "input_size_gib": args.input_path.stat().st_size / 1024**3,
        "output_size_gib": args.output_path.stat().st_size / 1024**3,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
