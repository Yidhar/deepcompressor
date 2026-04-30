#!/usr/bin/env python3
"""Repack kitchen-native SVDQuant W4A4 checkpoints into tile-packed storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from deepcompressor.backend.kitchen.tilepack import (
        KITCHEN_TILEPACK_LAYOUT_NAME,
        repack_safetensors,
    )
except ModuleNotFoundError:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from deepcompressor.backend.kitchen.tilepack import (
        KITCHEN_TILEPACK_LAYOUT_NAME,
        repack_safetensors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="natural-layout kitchen safetensors")
    parser.add_argument("--output", required=True, type=Path, help="tile-packed output safetensors")
    parser.add_argument(
        "--device",
        default="auto",
        help="temporary repack device: auto, cpu, cuda, or cuda:N. Default uses CUDA when available.",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    count = repack_safetensors(args.input, args.output, device=args.device, verbose=True)
    print(f"repacked {count} SVDQuant W4A4 layers to {KITCHEN_TILEPACK_LAYOUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
