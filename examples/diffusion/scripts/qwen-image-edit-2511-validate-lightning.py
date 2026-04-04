#!/usr/bin/env python3
"""Validate a Lightning LoRA against a merged Nunchaku int4 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image
from PIL import Image
from torch.utils import data
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[3]
NUNCHAKU_ROOT = ROOT.parent / "nunchaku"
for candidate in (ROOT, NUNCHAKU_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from nunchaku import NunchakuQwenImageTransformer2DModel


SMOKE_CASES = [
    {
        "prompt": "change the text to read '双截棍 Qwen Image Edit is here'",
        "filename": "neon_sign",
        "url": "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/neon_sign.png",
    },
    {
        "prompt": "Remove all UI text elements from the image. Keep the feeling that the characters and scene are in water. Also, remove the green UI elements at the bottom.",
        "filename": "comfy_poster",
        "url": "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/comfy_poster.png",
    },
]


class MultiImageDataset(data.Dataset):
    def __init__(self, gen_dirpath: Path, ref_dirpath: Path):
        super().__init__()
        self.gen_names = sorted([name for name in os.listdir(gen_dirpath) if name.endswith(".png")])
        self.ref_names = sorted([name for name in os.listdir(ref_dirpath) if name.endswith(".png")])
        assert len(self.gen_names) == len(self.ref_names)
        self.gen_dirpath = gen_dirpath
        self.ref_dirpath = ref_dirpath
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        ref_image = Image.open(self.ref_dirpath / self.ref_names[idx]).convert("RGB")
        gen_image = Image.open(self.gen_dirpath / self.gen_names[idx]).convert("RGB")
        if ref_image.size != gen_image.size:
            ref_image = ref_image.resize(gen_image.size, Image.Resampling.BICUBIC)
        return self.transform(gen_image), self.transform(ref_image)



def hash_str_to_int(text: str) -> int:
    modulus = 10**9 + 7
    value = 0
    for ch in text:
        value = (value * 31 + ord(ch)) % modulus
    return value



def mib_from_bytes(value: int | float) -> int:
    return int(round(float(value) / (1024 * 1024)))



def gib_from_mib(value: int | float) -> float:
    return float(value) / 1024.0



def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]



def is_remote_path(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}



def resolve_image_path(prompt_yaml: Path, image_path: str) -> str:
    if is_remote_path(image_path):
        return image_path
    path = Path(image_path)
    if not path.is_absolute():
        path = (prompt_yaml.parent / path).resolve()
    return str(path)



def split_sample_id(sample_id: str, known_prompt_keys: set[str]) -> tuple[str, str]:
    if sample_id in known_prompt_keys:
        return sample_id, sample_id
    base, sep, suffix = sample_id.rpartition("-")
    if sep and suffix.isdigit() and base in known_prompt_keys:
        return base, sample_id
    raise KeyError(f"Cannot map sample id '{sample_id}' to a prompt key.")



def build_prompt_dataset(prompt_yaml: Path, ref_samples_dir: Path | None, names: list[str], num_variants: int) -> list[dict]:
    with open(prompt_yaml, "r", encoding="utf-8") as handle:
        data_map = yaml.safe_load(handle)
    prompt_map: dict[str, dict[str, str]] = {}
    for key, row in data_map.items():
        if names and key not in names:
            continue
        prompt_map[key] = {
            "prompt": row["prompt"],
            "image": resolve_image_path(prompt_yaml, row["image"]),
        }
    if not prompt_map:
        raise ValueError("No prompts selected for validation.")

    dataset: list[dict] = []
    prompt_keys = set(prompt_map)
    if ref_samples_dir is not None:
        if not ref_samples_dir.exists():
            raise FileNotFoundError(f"Missing reference sample directory: {ref_samples_dir}")
        for image_path in sorted(ref_samples_dir.glob("*.png")):
            prompt_key, sample_id = split_sample_id(image_path.stem, prompt_keys)
            row = prompt_map[prompt_key]
            dataset.append(
                {
                    "prompt": row["prompt"],
                    "filename": sample_id,
                    "image": load_image(row["image"]).convert("RGB"),
                }
            )
    else:
        variant_count = max(num_variants, 1)
        for prompt_key, row in prompt_map.items():
            for variant_idx in range(variant_count):
                sample_id = prompt_key if variant_count == 1 else f"{prompt_key}-{variant_idx}"
                dataset.append(
                    {
                        "prompt": row["prompt"],
                        "filename": sample_id,
                        "image": load_image(row["image"]).convert("RGB"),
                    }
                )
    return dataset



def query_process_gpu_memory_mib(pid: int) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0
    peak = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            app_pid = int(parts[0])
            used_mib = int(parts[1])
        except ValueError:
            continue
        if app_pid == pid:
            peak = max(peak, used_mib)
    return peak


class ProcessMemorySampler:
    def __init__(self, pid: int, interval_s: float) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, query_process_gpu_memory_mib(self.pid))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ProcessMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.peak_mib = max(self.peak_mib, query_process_gpu_memory_mib(self.pid))



def maybe_sync(device_name: str) -> None:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()



def compute_lpips(ref_dir: Path, gen_dir: Path, device: torch.device) -> float:
    metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    dataset = MultiImageDataset(gen_dir, ref_dir)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for gen_tensor, ref_tensor in loader:
            metric.update(gen_tensor.to(device), ref_tensor.to(device))
    return float(metric.compute().item())



def scheduler_for_lightning() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_config(
        {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
    )



def ensure_smoke_inputs(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = []
    for case in SMOKE_CASES:
        image_path = cache_dir / f"{case['filename']}.png"
        if not image_path.exists():
            image = load_image(case["url"]).convert("RGB")
            image.save(image_path)
        dataset.append(
            {
                "prompt": case["prompt"],
                "filename": case["filename"],
                "image": load_image(str(image_path)).convert("RGB"),
            }
        )
    return dataset



def build_dataset(args: argparse.Namespace) -> list[dict]:
    if args.prompt_yaml is not None:
        return build_prompt_dataset(
            prompt_yaml=args.prompt_yaml,
            ref_samples_dir=args.ref_samples_dir,
            names=parse_csv_list(args.names),
            num_variants=args.num_variants,
        )
    return ensure_smoke_inputs(args.output_dir / "inputs")



def build_bf16_pipeline(base_model_path: Path, lora_path: Path, scheduler, torch_dtype: torch.dtype, device_name: str):
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(base_model_path), scheduler=scheduler, torch_dtype=torch_dtype)
    pipe = pipe.to(device_name)
    pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    return pipe



def build_int4_pipeline(base_model_path: Path, int4_path: Path, scheduler, torch_dtype: torch.dtype, device_name: str):
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        str(int4_path),
        torch_dtype=torch_dtype,
        device=device_name,
        precision="int4",
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(base_model_path),
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device_name)
    return pipe, transformer



def run_pipeline(
    pipeline: QwenImageEditPlusPipeline,
    dataset: list[dict],
    save_dir: Path,
    num_inference_steps: int,
    true_cfg_scale: float,
    device_name: str,
    negative_prompt: str,
    poll_interval: float,
) -> dict[str, float | int | str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    pipeline.set_progress_bar_config(disable=True)
    pid = os.getpid()
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        maybe_sync(device_name)
    with ProcessMemorySampler(pid, poll_interval) as sampler:
        start_time = time.perf_counter()
        for item in dataset:
            generator = torch.Generator(device=device_name).manual_seed(hash_str_to_int(item["filename"]))
            kwargs = {
                "prompt": item["prompt"],
                "image": item["image"],
                "generator": generator,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
            }
            if true_cfg_scale > 1.0:
                kwargs["negative_prompt"] = negative_prompt
            with torch.inference_mode():
                image = pipeline(**kwargs).images[0]
            image.save(save_dir / f"{item['filename']}.png")
        maybe_sync(device_name)
        elapsed = time.perf_counter() - start_time
    peak_process_gpu_mib = sampler.peak_mib
    peak_torch_reserved_mib = 0
    peak_torch_allocated_mib = 0
    if device_name.startswith("cuda") and torch.cuda.is_available():
        peak_torch_reserved_mib = mib_from_bytes(torch.cuda.max_memory_reserved())
        peak_torch_allocated_mib = mib_from_bytes(torch.cuda.max_memory_allocated())
    return {
        "backend": pipeline.__class__.__name__,
        "elapsed_sec": elapsed,
        "avg_image_sec": elapsed / max(len(dataset), 1),
        "num_images": len(dataset),
        "peak_process_gpu_mib": peak_process_gpu_mib,
        "peak_process_gpu_gib": gib_from_mib(peak_process_gpu_mib),
        "peak_torch_reserved_mib": peak_torch_reserved_mib,
        "peak_torch_reserved_gib": gib_from_mib(peak_torch_reserved_mib),
        "peak_torch_allocated_mib": peak_torch_allocated_mib,
        "peak_torch_allocated_gib": gib_from_mib(peak_torch_allocated_mib),
        "peak_memory_gib": gib_from_mib(peak_process_gpu_mib),
    }



def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--int4-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=[4, 8], required=True)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--prompt-yaml", type=Path)
    parser.add_argument("--ref-samples-dir", type=Path)
    parser.add_argument("--names", default="")
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--worker-mode", choices=["bf16", "int4"], default="")
    return parser.parse_args()



def worker_metrics_path(output_dir: Path, worker_mode: str) -> Path:
    return output_dir / f"{worker_mode}_metrics.json"



def run_worker(args: argparse.Namespace) -> None:
    device_name = "cuda"
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dataset = build_dataset(args)
    scheduler = scheduler_for_lightning()

    if args.worker_mode == "bf16":
        pipe = build_bf16_pipeline(args.base_model_path, args.lora_path, scheduler, torch_dtype, device_name)
        metrics = run_pipeline(
            pipe,
            dataset=dataset,
            save_dir=args.output_dir / "bf16",
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            device_name=device_name,
            negative_prompt=args.negative_prompt,
            poll_interval=args.poll_interval,
        )
        cleanup(pipe)
    elif args.worker_mode == "int4":
        pipe, transformer = build_int4_pipeline(args.base_model_path, args.int4_path, scheduler, torch_dtype, device_name)
        metrics = run_pipeline(
            pipe,
            dataset=dataset,
            save_dir=args.output_dir / "int4",
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            device_name=device_name,
            negative_prompt=args.negative_prompt,
            poll_interval=args.poll_interval,
        )
        cleanup(pipe, transformer)
    else:
        raise ValueError(f"Unsupported worker mode: {args.worker_mode}")

    metrics_path = worker_metrics_path(args.output_dir, args.worker_mode)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))



def spawn_worker(args: argparse.Namespace, worker_mode: str) -> dict[str, float | int | str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--base-model-path",
        str(args.base_model_path),
        "--lora-path",
        str(args.lora_path),
        "--int4-path",
        str(args.int4_path),
        "--output-dir",
        str(args.output_dir),
        "--steps",
        str(args.steps),
        "--true-cfg-scale",
        str(args.true_cfg_scale),
        "--dtype",
        args.dtype,
        "--negative-prompt",
        args.negative_prompt,
        "--poll-interval",
        str(args.poll_interval),
        "--num-variants",
        str(args.num_variants),
        "--worker-mode",
        worker_mode,
    ]
    if args.prompt_yaml is not None:
        cmd.extend(["--prompt-yaml", str(args.prompt_yaml)])
    if args.ref_samples_dir is not None:
        cmd.extend(["--ref-samples-dir", str(args.ref_samples_dir)])
    if args.names:
        cmd.extend(["--names", args.names])
    subprocess.run(cmd, check=True)
    metrics_path = worker_metrics_path(args.output_dir, worker_mode)
    return json.loads(metrics_path.read_text())



def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_worker(args)
        return

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(args)

    bf16_metrics = spawn_worker(args, "bf16")
    int4_metrics = spawn_worker(args, "int4")

    lpips = compute_lpips(args.output_dir / "bf16", args.output_dir / "int4", device)
    summary = {
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "dtype": args.dtype,
        "negative_prompt": args.negative_prompt,
        "poll_interval": args.poll_interval,
        "measurement": "isolated_worker_processes",
        "prompt_yaml": str(args.prompt_yaml) if args.prompt_yaml is not None else "",
        "ref_samples_dir": str(args.ref_samples_dir) if args.ref_samples_dir is not None else "",
        "names": args.names,
        "num_variants": args.num_variants,
        "bf16": bf16_metrics,
        "int4": int4_metrics,
        "lpips_vs_bf16": lpips,
        "bf16_dir": str(args.output_dir / "bf16"),
        "int4_dir": str(args.output_dir / "int4"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
