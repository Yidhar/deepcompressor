#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
import yaml
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from deepcompressor.utils.common import hash_str_to_int


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_MODEL_PATH = os.environ.get("QWEN_IMAGE_EDIT_2511_MODEL_PATH", "Qwen/Qwen-Image-Edit-2511")
DEFAULT_PROMPT_YAML = REPO_ROOT / "examples/diffusion/prompts/qwen-image-edit-demo.yaml"
DEFAULT_NUNCHAKU_ROOT = os.environ.get("NUNCHAKU_ROOT", str((WORKSPACE_ROOT / "nunchaku").resolve()))


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    prompt_key: str
    prompt: str
    image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare Qwen-Image-Edit-2511 BF16 and Nunchaku int4 inference.')
    parser.add_argument('--mode', choices=('bf16', 'nunchaku'), required=True)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        '--prompt-yaml',
        default=str(DEFAULT_PROMPT_YAML),
    )
    parser.add_argument(
        '--ref-samples-dir',
        default='',
        help='Optional reference sample directory. When set, filenames and seeds follow this directory exactly.',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--true-cfg-scale', type=float, default=4.0)
    parser.add_argument('--negative-prompt', default=' ')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num-variants', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--names', default='', help='Comma-separated prompt keys to keep.')
    parser.add_argument('--torch-dtype', default='bfloat16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--nunchaku-ckpt', default='')
    parser.add_argument('--nunchaku-root', default=DEFAULT_NUNCHAKU_ROOT)
    parser.add_argument('--precision', default='int4')
    parser.add_argument('--offload', choices=('none', 'model', 'sequential'), default='none')
    parser.add_argument('--num-blocks-on-gpu', type=int, default=1)
    parser.add_argument('--poll-interval', type=float, default=0.1)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f'Unsupported dtype: {name}') from exc


def is_remote_path(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {'http', 'https'}


def resolve_image_path(prompt_yaml: Path, image_path: str) -> str:
    if is_remote_path(image_path):
        return image_path
    path = Path(image_path)
    if not path.is_absolute():
        path = (prompt_yaml.parent / path).resolve()
    return str(path)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def split_sample_id(sample_id: str, known_prompt_keys: set[str]) -> tuple[str, str]:
    if sample_id in known_prompt_keys:
        return sample_id, sample_id
    base, sep, suffix = sample_id.rpartition('-')
    if sep and suffix.isdigit() and base in known_prompt_keys:
        return base, sample_id
    raise KeyError(f"Cannot map sample id '{sample_id}' to a prompt key.")


def build_sample_plan(
    prompt_yaml: Path,
    ref_samples_dir: Path | None,
    names: list[str],
    num_variants: int,
    limit: int,
) -> tuple[list[PromptSample], str]:
    with open(prompt_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    prompt_map: dict[str, dict[str, str]] = {}
    for key, row in data.items():
        if names and key not in names:
            continue
        prompt_map[key] = {
            'prompt': row['prompt'],
            'image': resolve_image_path(prompt_yaml, row['image']),
        }
    if not prompt_map:
        raise ValueError('No prompts selected for benchmarking.')

    prompt_keys = set(prompt_map)
    samples: list[PromptSample] = []
    if ref_samples_dir is not None:
        if not ref_samples_dir.exists():
            raise FileNotFoundError(f'Missing reference sample directory: {ref_samples_dir}')
        for image_path in sorted(ref_samples_dir.glob('*.png')):
            prompt_key, sample_id = split_sample_id(image_path.stem, prompt_keys)
            entry = prompt_map[prompt_key]
            samples.append(
                PromptSample(
                    sample_id=sample_id,
                    prompt_key=prompt_key,
                    prompt=entry['prompt'],
                    image=entry['image'],
                )
            )
        sample_dir_name = ref_samples_dir.name
    else:
        for prompt_key, entry in prompt_map.items():
            variant_count = max(num_variants, 1)
            for variant_idx in range(variant_count):
                sample_id = prompt_key if variant_count == 1 else f'{prompt_key}-{variant_idx}'
                samples.append(
                    PromptSample(
                        sample_id=sample_id,
                        prompt_key=prompt_key,
                        prompt=entry['prompt'],
                        image=entry['image'],
                    )
                )
        sample_dir_name = f'{prompt_yaml.stem}-{len(samples)}'

    if limit > 0:
        samples = samples[:limit]
        sample_dir_name = f'{sample_dir_name}-limit{limit}'
    return samples, sample_dir_name


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def round_value(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def mib_from_bytes(value: int | float) -> int:
    return int(round(float(value) / (1024 * 1024)))


def query_process_gpu_memory_mib(pid: int) -> int:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0
    peak = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
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

    def __enter__(self) -> 'ProcessMemorySampler':
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.peak_mib = max(self.peak_mib, query_process_gpu_memory_mib(self.pid))


def maybe_sync(device: str) -> None:
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_pipeline(args: argparse.Namespace, torch_dtype: torch.dtype):
    load_start = time.perf_counter()
    transformer = None
    if args.mode == 'bf16':
        pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    else:
        if not args.nunchaku_ckpt:
            raise ValueError('--nunchaku-ckpt is required for --mode nunchaku')
        if args.nunchaku_root:
            sys.path.insert(0, args.nunchaku_root)
        from nunchaku import NunchakuQwenImageTransformer2DModel

        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            args.nunchaku_ckpt,
            device=args.device,
            precision=args.precision,
            torch_dtype=torch_dtype,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            args.model_path,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    if args.offload == 'none':
        pipe = pipe.to(args.device)
    elif args.offload == 'model':
        pipe.enable_model_cpu_offload()
    elif args.offload == 'sequential':
        if transformer is None:
            pipe.enable_sequential_cpu_offload()
        else:
            transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=args.num_blocks_on_gpu)
            pipe._exclude_from_cpu_offload.append('transformer')
            pipe.enable_sequential_cpu_offload()
    load_s = time.perf_counter() - load_start
    return pipe, load_s


def main() -> None:
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_yaml = Path(args.prompt_yaml).resolve()
    ref_samples_dir = Path(args.ref_samples_dir).resolve() if args.ref_samples_dir else None
    selected_names = parse_csv_list(args.names)
    samples, sample_dir_name = build_sample_plan(
        prompt_yaml=prompt_yaml,
        ref_samples_dir=ref_samples_dir,
        names=selected_names,
        num_variants=args.num_variants,
        limit=args.limit,
    )
    sample_output_dir = output_dir / 'samples' / 'YAML' / sample_dir_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    pipe, load_s = build_pipeline(args, torch_dtype)
    pipe.set_progress_bar_config(disable=False)

    results: list[dict[str, object]] = []
    pid = os.getpid()

    for index, sample in enumerate(samples):
        input_image = load_image(sample.image).convert('RGB')
        seed = hash_str_to_int(sample.sample_id) if ref_samples_dir is not None else args.seed + index
        generator = torch.Generator(args.device).manual_seed(seed)
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            maybe_sync(args.device)
        with ProcessMemorySampler(pid, args.poll_interval) as sampler:
            maybe_sync(args.device)
            start = time.perf_counter()
            with torch.inference_mode():
                output = pipe(
                    image=input_image,
                    prompt=sample.prompt,
                    negative_prompt=args.negative_prompt,
                    true_cfg_scale=args.true_cfg_scale,
                    num_inference_steps=args.num_steps,
                    generator=generator,
                )
            maybe_sync(args.device)
            inference_s = time.perf_counter() - start
        image = output.images[0]
        image_path = sample_output_dir / f'{sample.sample_id}.png'
        image.save(image_path)
        peak_torch_reserved_mib = 0
        peak_torch_allocated_mib = 0
        if args.device.startswith('cuda') and torch.cuda.is_available():
            peak_torch_reserved_mib = mib_from_bytes(torch.cuda.max_memory_reserved())
            peak_torch_allocated_mib = mib_from_bytes(torch.cuda.max_memory_allocated())
        results.append(
            {
                'name': sample.sample_id,
                'prompt_key': sample.prompt_key,
                'prompt': sample.prompt,
                'image': sample.image,
                'output': str(image_path),
                'seed': seed,
                'inference_s': round_value(inference_s),
                'peak_process_gpu_mib': sampler.peak_mib,
                'peak_torch_reserved_mib': peak_torch_reserved_mib,
                'peak_torch_allocated_mib': peak_torch_allocated_mib,
            }
        )
        print(
            f'[{args.mode}] {sample.sample_id}: {inference_s:.3f}s, '
            f'peak_process_gpu_mib={sampler.peak_mib}, '
            f'peak_torch_reserved_mib={peak_torch_reserved_mib}, output={image_path}'
        )

    inference_values = [float(item['inference_s']) for item in results]
    process_vram_values = [float(item['peak_process_gpu_mib']) for item in results]
    torch_reserved_values = [float(item['peak_torch_reserved_mib']) for item in results]
    torch_allocated_values = [float(item['peak_torch_allocated_mib']) for item in results]
    summary = {
        'mode': args.mode,
        'model_path': args.model_path,
        'prompt_yaml': str(prompt_yaml),
        'ref_samples_dir': str(ref_samples_dir) if ref_samples_dir else '',
        'sample_dir': str(sample_output_dir),
        'sample_dir_name': sample_dir_name,
        'num_samples': len(results),
        'nunchaku_ckpt': args.nunchaku_ckpt,
        'num_steps': args.num_steps,
        'true_cfg_scale': args.true_cfg_scale,
        'offload': args.offload,
        'load_s': round_value(load_s),
        'avg_inference_s': round_value(sum(inference_values) / len(inference_values)),
        'p50_inference_s': round_value(percentile(inference_values, 0.50)),
        'p95_inference_s': round_value(percentile(inference_values, 0.95)),
        'max_inference_s': round_value(max(inference_values)),
        'avg_peak_process_gpu_mib': round_value(sum(process_vram_values) / len(process_vram_values)),
        'p95_peak_process_gpu_mib': round_value(percentile(process_vram_values, 0.95)),
        'max_peak_process_gpu_mib': int(max(process_vram_values)),
        'avg_peak_torch_reserved_mib': round_value(sum(torch_reserved_values) / len(torch_reserved_values)),
        'max_peak_torch_reserved_mib': int(max(torch_reserved_values)),
        'avg_peak_torch_allocated_mib': round_value(sum(torch_allocated_values) / len(torch_allocated_values)),
        'max_peak_torch_allocated_mib': int(max(torch_allocated_values)),
        'results': results,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
