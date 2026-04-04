#!/usr/bin/env python3
"""Offline merge a LoRA into a Qwen-Image-Edit-2511 diffusers pipeline."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from deepcompressor.app.diffusion.pipeline.config import DiffusionPipelineConfig, LoRAConfig
from deepcompressor.data.utils.dtype import eval_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA into a Qwen-Image-Edit-2511 diffusers pipeline and save the merged model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Base diffusers model path or repo id.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="LoRA repo id or local path passed to diffusers pipeline.lora_state_dict().",
    )
    parser.add_argument(
        "--weight-name",
        type=str,
        required=True,
        help="LoRA safetensors filename inside --lora-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the merged diffusers pipeline.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LoRA strength baked into the merged weights.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype used while loading and saving the merged pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used while merging the LoRA.",
    )
    parser.add_argument(
        "--unsafe-serialization",
        action="store_true",
        help="Save using PyTorch .bin files instead of safetensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = DiffusionPipelineConfig(
        name="qwen-image-edit-2511",
        path=args.model_path,
        dtype=eval_dtype(args.dtype, with_quant_dtype=False, with_none=False),
        device=args.device,
        lora=LoRAConfig(
            path=args.lora_path,
            weight_name=args.weight_name,
            alpha=args.alpha,
            merge=True,
        ),
    )

    pipeline = pipeline_config.build()
    pipeline_config.load_lora(pipeline)

    pipeline = pipeline.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pipeline.save_pretrained(output_dir, safe_serialization=not args.unsafe_serialization)
    print(f"Saved merged pipeline to {output_dir}")


if __name__ == "__main__":
    main()
