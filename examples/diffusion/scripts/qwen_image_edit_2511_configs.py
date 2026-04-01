#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
GENERATED_ROOT = REPO_ROOT / ".tmp" / "generated-configs" / "qwen-image-edit-2511"

MODEL_CFG = "examples/diffusion/configs/model/qwen-image-edit-2511.yaml"
INT4_CFG = "examples/diffusion/configs/svdquant/int4.yaml"

PROMPT_DEMO = "examples/diffusion/prompts/qwen-image-edit-demo.yaml"
PROMPT_SEARCH = "examples/diffusion/prompts/qwen-image-edit-search-holdout.yaml"


def _default_search_calib_path() -> str:
    relative = Path("datasets/torch.bfloat16/qwen-image-edit-2511/fmeuler50-g4.0/qdiff/s128")
    candidates = [
        REPO_ROOT / relative,
        WORKSPACE_ROOT / "deploy" / relative,
        WORKSPACE_ROOT / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])


MODEL_PATH = os.environ.get("QWEN_IMAGE_EDIT_2511_MODEL_PATH", "Qwen/Qwen-Image-Edit-2511")
REF_ROOT = os.environ.get(
    "QWEN_IMAGE_EDIT_2511_REF_ROOT",
    str((REPO_ROOT / "references/torch.bfloat16/qwen-image-edit-2511/fmeuler50-cfg4.0").resolve()),
)
SEARCH_CALIB_PATH = os.environ.get("QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH", _default_search_calib_path())


BASELINE_DEMO_OVERLAY = {
    "quant": {
        "calib": {
            "data": "qwen-image-edit-demo",
            "num_samples": 2,
            "num_workers": 0,
        },
        "develop_dtype": "torch.bfloat16",
        "smooth": {
            "proj": {
                "strategy": "Manual",
                "alpha": 0.5,
                "beta": -1,
                "sample_batch_size": 8,
                "sample_size": 2,
                "outputs_device": "cuda",
                "skips": [
                    "embed",
                    "resblock",
                    "transformer_proj_in",
                    "transformer_proj_out",
                    "transformer_norm",
                    "transformer_add_norm",
                    "transformer_mod",
                    "down_sample",
                    "up_sample",
                ],
            }
        },
        "wgts": {
            "calib_range": {
                "num_grids": 10,
                "sample_batch_size": 8,
                "sample_size": 2,
            },
            "low_rank": {
                "num_iters": 2,
                "sample_batch_size": 8,
                "sample_size": 2,
                "outputs_device": "cuda",
                "skips": [
                    "embed",
                    "resblock",
                    "transformer_proj_in",
                    "transformer_proj_out",
                    "transformer_norm",
                    "transformer_add_norm",
                    "transformer_mod",
                    "down_sample",
                    "up_sample",
                ],
            },
        },
    }
}

GPTQ_BF16_OVERLAY = {
    "quant": {
        "develop_dtype": "torch.bfloat16",
        "wgts": {
            "enable_kernel_gptq": True,
            "kernel_gptq": {
                "damp_percentage": 0.01,
                "block_size": 128,
                "num_inv_tries": 250,
                "hessian_block_size": 512,
            },
        },
    },
    "output": {"dirname": "qwen-image-edit-demo-gptq-bf16"},
}

SEARCH_BASE_OVERLAY = {
    "pipeline": {"path": MODEL_PATH},
    "output": {"dirname": "qwen-image-edit-2511-search"},
    "eval": {
        "num_gpus": 1,
        "batch_size": 1,
        "batch_size_per_gpu": 1,
        "num_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "max_sequence_length": 512,
        "protocol": "fmeuler{num_steps}-cfg{true_cfg_scale}",
        "num_samples": 6,
        "benchmarks": [PROMPT_SEARCH],
        "gt_metrics": ["clip_iqa", "clip_score", "image_reward"],
        "ref_metrics": ["psnr", "lpips", "ssim", "fid"],
        "ref_root": REF_ROOT,
    },
    "quant": {
        "calib": {
            "data": "qdiff-s128",
            "path": SEARCH_CALIB_PATH,
            "num_samples": 128,
            "num_workers": 0,
            "batch_size": 8,
        },
        "develop_dtype": "torch.bfloat16",
        "smooth": {
            "proj": {
                "objective": "OutputsError",
                "strategy": "GridSearch",
                "granularity": "Layer",
                "spans": [["AbsMax", "AbsMax"]],
                "alpha": 0.5,
                "beta": -1,
                "num_grids": 10,
                "allow_low_rank": True,
                "fuse_when_possible": False,
                "element_batch_size": -1,
                "sample_batch_size": 8,
                "element_size": -1,
                "sample_size": -1,
                "outputs_device": "cpu",
                "skips": [
                    "embed",
                    "resblock",
                    "transformer_proj_in",
                    "transformer_proj_out",
                    "transformer_norm",
                    "transformer_add_norm",
                    "transformer_mod",
                    "down_sample",
                    "up_sample",
                ],
            }
        },
        "wgts": {
            "calib_range": {
                "objective": "OutputsError",
                "strategy": "Manual",
                "granularity": "Layer",
                "element_batch_size": 64,
                "sample_batch_size": 8,
                "element_size": 512,
                "sample_size": -1,
                "ratio": 1.0,
                "max_shrink": 0.2,
                "max_expand": 1.0,
                "num_grids": 40,
            },
            "low_rank": {
                "rank": 32,
                "objective": "OutputsError",
                "sample_batch_size": 8,
                "sample_size": -1,
                "num_iters": 50,
                "outputs_device": "cpu",
                "early_stop": True,
                "skips": [
                    "embed",
                    "resblock",
                    "transformer_proj_in",
                    "transformer_proj_out",
                    "transformer_norm",
                    "transformer_add_norm",
                    "transformer_mod",
                    "down_sample",
                    "up_sample",
                ],
            },
        },
    },
}

SEARCH_CANDIDATES = {
    "fast-r32": {
        "output": {"dirname": "qwen-image-edit-2511-search-fast-r32"},
        "quant": {
            "calib": {"num_samples": 64},
            "smooth": {
                "proj": {
                    "strategy": "Manual",
                    "beta": 0.5,
                    "num_grids": 10,
                    "sample_batch_size": 8,
                    "sample_size": 64,
                }
            },
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 32, "num_iters": 20, "sample_size": 64},
            },
        },
    },
    "fast-r64": {
        "output": {"dirname": "qwen-image-edit-2511-search-fast-r64"},
        "quant": {
            "calib": {"num_samples": 64},
            "smooth": {
                "proj": {
                    "strategy": "Manual",
                    "beta": 0.5,
                    "num_grids": 10,
                    "sample_batch_size": 8,
                    "sample_size": 64,
                }
            },
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 64, "num_iters": 20, "sample_size": 64},
            },
        },
    },
    "fast-r128": {
        "output": {"dirname": "qwen-image-edit-2511-search-fast-r128"},
        "quant": {
            "calib": {"num_samples": 64},
            "smooth": {
                "proj": {
                    "strategy": "Manual",
                    "beta": 0.5,
                    "num_grids": 10,
                    "sample_batch_size": 8,
                    "sample_size": 64,
                }
            },
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 128, "num_iters": 20, "sample_size": 64},
            },
        },
    },
    "balanced-r32": {
        "output": {"dirname": "qwen-image-edit-2511-search-balanced-r32"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1, "num_grids": 10}},
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 32, "num_iters": 50},
            },
        },
    },
    "balanced-r32-i64": {
        "output": {"dirname": "qwen-image-edit-2511-search-balanced-r32-i64"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1, "num_grids": 10}},
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 32, "num_iters": 64},
            },
        },
    },
    "balanced-r128": {
        "output": {"dirname": "qwen-image-edit-2511-search-balanced-r128"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1, "num_grids": 10}},
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 128, "num_iters": 50},
            },
        },
    },
    "balanced-r32-b125": {
        "output": {"dirname": "qwen-image-edit-2511-search-balanced-r32-b125"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.25, "num_grids": 12}},
            "wgts": {
                "calib_range": {"num_grids": 40},
                "low_rank": {"rank": 32, "num_iters": 50},
            },
        },
    },
    "mid-r32": {
        "output": {"dirname": "qwen-image-edit-2511-search-mid-r32"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.25, "num_grids": 12}},
            "wgts": {
                "calib_range": {"num_grids": 60},
                "low_rank": {"rank": 32, "num_iters": 64},
            },
        },
    },
    "mid-r128": {
        "output": {"dirname": "qwen-image-edit-2511-search-mid-r128"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.25, "num_grids": 12}},
            "wgts": {
                "calib_range": {"num_grids": 60},
                "low_rank": {"rank": 128, "num_iters": 64},
            },
        },
    },
    "quality-r32": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r32"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -2, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 32, "num_iters": 100},
            },
        },
    },
    "quality-r64": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r64"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -2, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 64, "num_iters": 100},
            },
        },
    },
    "quality-r64-b175": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r64-b175"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.75, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 64, "num_iters": 100},
            },
        },
    },
    "quality-r96": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r96"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -2, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 96, "num_iters": 100},
            },
        },
    },
    "quality-r96-b175": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r96-b175"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.75, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 96, "num_iters": 100},
            },
        },
    },
    "quality-r96-i128": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r96-i128"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -2, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 96, "num_iters": 128},
            },
        },
    },
    "quality-r128": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r128"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -2, "num_grids": 20}},
            "wgts": {
                "calib_range": {"num_grids": 80},
                "low_rank": {"rank": 128, "num_iters": 100},
            },
        },
    },
    "quality-r128-b15": {
        "output": {"dirname": "qwen-image-edit-2511-search-quality-r128-b15"},
        "quant": {
            "calib": {"num_samples": 128},
            "smooth": {"proj": {"strategy": "GridSearch", "beta": -1.5, "num_grids": 16}},
            "wgts": {
                "calib_range": {"num_grids": 60},
                "low_rank": {"rank": 128, "num_iters": 80},
            },
        },
    },
}

SEARCH_REF_OVERLAY = {
    "pipeline": {"path": MODEL_PATH},
    "output": {"dirname": "reference"},
    "eval": {
        "num_gpus": 4,
        "batch_size": 4,
        "batch_size_per_gpu": 1,
        "num_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "max_sequence_length": 512,
        "protocol": "fmeuler{num_steps}-cfg{true_cfg_scale}",
        "num_samples": 6,
        "benchmarks": [PROMPT_SEARCH],
        "gt_metrics": [],
        "ref_metrics": [],
        "ref_root": REF_ROOT,
    },
}

SEARCH_EVAL_OVERLAY = {
    "eval": {"gt_metrics": []},
    "quant": {"smooth": None, "ipts": {"dtype": None}},
}


def _write_yaml(filename: str, data: dict) -> str:
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    path = GENERATED_ROOT / filename
    rendered = yaml.safe_dump(data, sort_keys=False)
    if not path.exists() or path.read_text(encoding="utf-8") != rendered:
        path.write_text(rendered, encoding="utf-8")
    return str(path)


def _sample_overlay(dirname: str, *, num_steps: int | None = None, copy_on_save: bool | None = None) -> dict:
    overlay = {
        "skip_gen": False,
        "skip_eval": True,
        "output": {"dirname": dirname},
        "eval": {
            "num_gpus": 1,
            "batch_size": 1,
            "batch_size_per_gpu": 1,
            "num_samples": 2,
            "benchmarks": [PROMPT_DEMO],
        },
    }
    if num_steps is not None:
        overlay["eval"]["num_steps"] = num_steps
    if copy_on_save is not None:
        overlay["copy_on_save"] = copy_on_save
    return overlay


def materialize_baseline_stage(stage: str) -> list[str]:
    overlays: list[tuple[str, dict]] = []
    if stage == "collect":
        overlays = [(
            "baseline.collect.yaml",
            {
                "collect": {
                    "root": "datasets",
                    "dataset_name": "qwen-image-edit-demo",
                    "data_path": PROMPT_DEMO,
                    "num_samples": 2,
                }
            },
        )]
    elif stage == "quantize":
        overlays = [
            ("baseline.demo.yaml", BASELINE_DEMO_OVERLAY),
            ("baseline.gptq-bf16.yaml", GPTQ_BF16_OVERLAY),
        ]
    elif stage == "export":
        overlays = [
            ("baseline.demo.yaml", BASELINE_DEMO_OVERLAY),
            ("baseline.gptq-bf16.yaml", GPTQ_BF16_OVERLAY),
            (
                "baseline.export.yaml",
                {
                    "save_model": "default",
                    "skip_gen": True,
                    "skip_eval": True,
                    "copy_on_save": False,
                    "output": {"dirname": "qwen-image-edit-demo-gptq-bf16-export"},
                },
            ),
        ]
    elif stage == "sample-bf16":
        overlays = [("baseline.sample-bf16.yaml", _sample_overlay("qwen-image-edit-demo-bf16-samples"))]
    elif stage == "sample-bf16-50":
        overlays = [("baseline.sample-bf16-50.yaml", _sample_overlay("qwen-image-edit-demo-bf16-samples-50steps", num_steps=50))]
    elif stage == "sample-gptq":
        overlays = [
            ("baseline.demo.yaml", BASELINE_DEMO_OVERLAY),
            ("baseline.gptq-bf16.yaml", GPTQ_BF16_OVERLAY),
            (
                "baseline.sample-gptq.yaml",
                _sample_overlay("qwen-image-edit-demo-gptq-samples", copy_on_save=False),
            ),
        ]
    elif stage == "sample-gptq-50":
        overlays = [
            ("baseline.demo.yaml", BASELINE_DEMO_OVERLAY),
            ("baseline.gptq-bf16.yaml", GPTQ_BF16_OVERLAY),
            (
                "baseline.sample-gptq-50.yaml",
                _sample_overlay("qwen-image-edit-demo-gptq-samples-50steps", num_steps=50, copy_on_save=False),
            ),
        ]
    else:
        raise ValueError(f"Unsupported baseline stage: {stage}")
    return [_write_yaml(filename, deepcopy(data)) for filename, data in overlays]


def materialize_search_stage(stage: str, candidate: str | None = None) -> list[str]:
    overlays: list[tuple[str, dict]] = []
    if stage == "refs":
        overlays = [("search.refs.yaml", SEARCH_REF_OVERLAY)]
    elif stage in ("launch", "eval"):
        if candidate not in SEARCH_CANDIDATES:
            raise ValueError(f"Unknown search candidate: {candidate}")
        overlays = [
            ("search.base.yaml", SEARCH_BASE_OVERLAY),
            ("search.gptq-bf16.yaml", GPTQ_BF16_OVERLAY),
            (f"search.{candidate}.yaml", SEARCH_CANDIDATES[candidate]),
        ]
        if stage == "eval":
            overlays.append(("search.eval.yaml", SEARCH_EVAL_OVERLAY))
    else:
        raise ValueError(f"Unsupported search stage: {stage}")
    return [_write_yaml(filename, deepcopy(data)) for filename, data in overlays]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize transient Qwen-Image-Edit-2511 YAML overlays.")
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument(
        "stage",
        choices=(
            "collect",
            "quantize",
            "export",
            "sample-bf16",
            "sample-bf16-50",
            "sample-gptq",
            "sample-gptq-50",
        ),
    )

    search = subparsers.add_parser("search")
    search.add_argument("stage", choices=("refs", "launch", "eval"))
    search.add_argument("--candidate", choices=tuple(SEARCH_CANDIDATES), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workflow == "baseline":
        paths = materialize_baseline_stage(args.stage)
    else:
        paths = materialize_search_stage(args.stage, args.candidate)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
