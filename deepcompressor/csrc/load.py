# -*- coding: utf-8 -*-
"""Deepcompressor Extension — lazy loaded to avoid slow JIT compilation at import time."""

import os

__all__ = ["_C"]

_C = None


def get_extension():
    """Load the CUDA extension on first use."""
    global _C
    if _C is None:
        from torch.utils.cpp_extension import load
        dirpath = os.path.dirname(__file__)
        _C = load(
            name="deepcompressor_C",
            sources=[f"{dirpath}/pybind.cpp", f"{dirpath}/quantize/quantize.cu"],
            extra_cflags=["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3", "-std=c++20",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_HALF2_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "--use_fast_math", "--ptxas-options=--allow-expensive-optimizations=true",
                "--threads=8",
            ],
        )
    return _C
