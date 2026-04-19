"""Verify convert_to_nunchaku_qwenimage_transformer_block_state_dict emits
per-head to_q/to_k/to_v (and add_q_proj/add_k_proj/add_v_proj) instead of the
legacy fused to_qkv / add_qkv_proj.

The caller path has not changed: calibration still produces separate linears
per Q/K/V in state_dict; only the export dispatch table was updated to stop
concatenating them.

The test sidesteps AWQ modulation (which requires calibrated int4-quantized
inputs that are hard to mock correctly) by instead calling the lower-level
`convert_to_nunchaku_transformer_block_state_dict` with just the attention
W4A4 linear entries — exactly the part this patch modifies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_transformer_block_state_dict,
)


def _make_linear(block: str, name: str, in_features: int, out_features: int,
                 rank: int, group_size: int, dtype):
    """Build the four per-linear dicts the converter consumes.

    DeepCompressor's post-calibration `state_dict` holds pre-quantized weights:
    each value is `scale * q` where q is already a signed int4 value. We mock
    that by constructing `weight = 0` (which trivially quantizes to 0 ∈ [-8, 7])
    with random but bounded scale/bias/smooth/branch tensors.
    """
    full = f"{block}.{name}"
    state = {
        # Zero weight — after `round(weight / scale)` quantization lands on 0,
        # guaranteed within int4 range. We only care about output keys + shapes.
        f"{full}.weight": torch.zeros(out_features, in_features, dtype=dtype),
        f"{full}.bias":   torch.randn(out_features, dtype=dtype),
    }
    ngroups = in_features // group_size
    scale = {f"{full}.weight.scale.0": (torch.randn(out_features, 1, ngroups, 1, dtype=dtype).abs() + 1e-3)}
    smooth = {full: (torch.randn(in_features, dtype=dtype).abs() + 1e-2)}
    branch = {full: {
        "a.weight": torch.randn(rank, in_features, dtype=dtype),
        "b.weight": torch.randn(out_features, rank, dtype=dtype),
    }}
    return state, scale, smooth, branch


def _merge(dicts_list):
    out: dict = {}
    for d in dicts_list:
        out.update(d)
    return out


def test_qwen_block_emits_split_qkv_keys():
    """Smoke-test the split-QKV dispatch table.

    Calls the dispatch-table consumer directly with the SAME local_name_map /
    smooth_name_map / branch_name_map / convert_map the patched
    convert_to_nunchaku_qwenimage_transformer_block_state_dict builds for
    attention, then asserts each output projection is emitted as its own
    per-head W4A4 entry.
    """
    block = "transformer_blocks.0"
    hidden, rank, group_size = 512, 16, 64
    dtype = torch.bfloat16

    sps, scs, sms, brs = [], [], [], []
    for name in ("attn.to_q", "attn.to_k", "attn.to_v",
                 "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                 "attn.to_out.0", "attn.to_add_out"):
        s, sc, sm, br = _make_linear(block, name, hidden, hidden, rank, group_size, dtype)
        sps.append(s); scs.append(sc); sms.append(sm); brs.append(br)

    state_dict = _merge(sps)
    scale_dict = _merge(scs)
    smooth_dict = _merge(sms)
    branch_dict = _merge(brs)

    # Post-patch convert table for attention only (modulation + MLP omitted for
    # test focus).
    local_name_map = {
        "attn.to_q":        "attn.to_q",
        "attn.to_k":        "attn.to_k",
        "attn.to_v":        "attn.to_v",
        "attn.add_q_proj":  "attn.add_q_proj",
        "attn.add_k_proj":  "attn.add_k_proj",
        "attn.add_v_proj":  "attn.add_v_proj",
        "attn.to_out.0":    "attn.to_out.0",
        "attn.to_add_out":  "attn.to_add_out",
    }
    same_map = dict(local_name_map)  # smooth/branch collapse to linear name
    convert_map = {k: "linear" for k in local_name_map}

    converted = convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block,
        local_name_map=local_name_map,
        smooth_name_map=same_map,
        branch_name_map=same_map,
        convert_map=convert_map,
    )

    # Legacy fused keys must NOT appear
    for fused_prefix in ("attn.to_qkv.", "attn.add_qkv_proj."):
        fused_keys = [k for k in converted if k.startswith(fused_prefix)]
        assert not fused_keys, f"unexpected fused keys: {fused_keys}"

    # Each split linear must carry the SVDQuant tensor family
    svd_suffixes = ("qweight", "wscales", "proj_down", "proj_up", "smooth_factor", "bias")
    for name in local_name_map:
        for suffix in svd_suffixes:
            key = f"{name}.{suffix}"
            assert key in converted, f"missing {key} in converted block — keys: {sorted(converted.keys())[:10]}..."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
