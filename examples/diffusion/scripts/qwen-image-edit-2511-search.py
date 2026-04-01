#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from qwen_image_edit_2511_configs import INT4_CFG, MODEL_CFG, SEARCH_CANDIDATES, materialize_search_stage


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "runs"
LOG_ROOT = REPO_ROOT / "runs" / "search-logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Qwen-Image-Edit-2511 W4A4+GPTQ search jobs.")
    parser.add_argument("stage", choices=("refs", "launch", "eval", "summary"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--candidates", default=",".join(SEARCH_CANDIDATES))
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--micromamba-env", default=os.environ.get("MICROMAMBA_ENV", ""))
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def env_prefix(gpu: str, python_bin: str, micromamba_env: str) -> str:
    env = {
        "PYTHONPATH": str(REPO_ROOT),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": gpu,
    }
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    python_cmd = shlex.quote(python_bin)
    if micromamba_env:
        python_cmd = f"micromamba run -p {shlex.quote(micromamba_env)} {python_cmd}"
    return f"{assignments} {python_cmd}"


def shell_join(items: list[str] | tuple[str, ...]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def build_ptq_command(
    gpu: str,
    python_bin: str,
    micromamba_env: str,
    config_paths: list[str],
    extra_args: list[str] | tuple[str, ...] = (),
) -> str:
    cmd = (
        f"{env_prefix(gpu, python_bin, micromamba_env)} "
        f"-m deepcompressor.app.diffusion.ptq {shell_join(config_paths)}"
    )
    if extra_args:
        cmd += f" {shell_join(list(extra_args))}"
    return cmd


def run_shell(command: str, log_path: Path | None = None, wait: bool = False) -> subprocess.Popen | subprocess.CompletedProcess:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        wrapped = f"{command} > {log_path} 2>&1"
    else:
        wrapped = command
    if wait:
        return subprocess.run(wrapped, cwd=REPO_ROOT, shell=True, check=True, text=True)
    return subprocess.Popen(wrapped, cwd=REPO_ROOT, shell=True)


def find_latest_named_run(job_name: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in RUNS_ROOT.rglob("run-*"):
        parent_name = path.parent.name
        if parent_name not in (job_name, f"{job_name}.RUNNING"):
            continue
        if path.name.endswith(".RUNNING") or path.name.endswith(".ERROR"):
            continue
        candidates.append((path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No run found for {job_name}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_latest_model_run(job_name: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in RUNS_ROOT.rglob("run-*"):
        parent_name = path.parent.name
        if parent_name not in (job_name, f"{job_name}.RUNNING"):
            continue
        if path.name.endswith(".RUNNING") or path.name.endswith(".ERROR"):
            continue
        model_path = path / "model" / "model.pt"
        if not model_path.exists():
            continue
        candidates.append((model_path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No model run found for {job_name}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def stage_refs(args: argparse.Namespace) -> None:
    gpu_list = parse_csv_list(args.gpus)
    ref_cfgs = materialize_search_stage("refs")
    base_configs = [MODEL_CFG, *ref_cfgs]
    procs = []
    chunk_step = len(gpu_list)
    for chunk_start, gpu in enumerate(gpu_list):
        cmd = build_ptq_command(
            gpu,
            args.python_bin,
            args.micromamba_env,
            base_configs,
            extra_args=(
                "--eval-num-gpus",
                "1",
                "--eval-chunk-start",
                str(chunk_start),
                "--eval-chunk-step",
                str(chunk_step),
                "--skip-eval",
                "true",
            ),
        )
        log_path = LOG_ROOT / f"bf16-refs-c{chunk_start}-of-{chunk_step}.log"
        print(f"[GPU {gpu}] refs chunk {chunk_start}/{chunk_step}")
        print(cmd)
        procs.append((gpu, chunk_start, run_shell(cmd, log_path=log_path, wait=False), log_path))
        time.sleep(2)
    for gpu, chunk_start, proc, log_path in procs:
        code = proc.wait()
        print(f"refs chunk {chunk_start} gpu={gpu}: exit_code={code} log={log_path}")
        if code != 0:
            raise subprocess.CalledProcessError(code, f"refs chunk {chunk_start} on gpu {gpu}")


def stage_launch(args: argparse.Namespace) -> None:
    gpu_list = parse_csv_list(args.gpus)
    candidates = parse_csv_list(args.candidates)
    if len(gpu_list) < len(candidates):
        raise ValueError("Need at least as many GPUs as candidates for launch stage.")
    procs = []
    for gpu, candidate in zip(gpu_list, candidates, strict=True):
        config_paths = [MODEL_CFG, INT4_CFG, *materialize_search_stage("launch", candidate)]
        cmd = build_ptq_command(
            gpu,
            args.python_bin,
            args.micromamba_env,
            config_paths,
            extra_args=("--save-model", "default", "--skip-gen", "true", "--skip-eval", "true"),
        )
        log_path = LOG_ROOT / f"launch-{candidate}.log"
        print(f"[GPU {gpu}] {candidate}")
        print(cmd)
        procs.append((candidate, run_shell(cmd, log_path=log_path, wait=False), log_path))
        time.sleep(2)
    if args.wait:
        for candidate, proc, log_path in procs:
            code = proc.wait()
            print(f"{candidate}: exit_code={code} log={log_path}")


def stage_eval(args: argparse.Namespace) -> None:
    gpu_list = parse_csv_list(args.gpus)
    candidates = parse_csv_list(args.candidates)
    if len(gpu_list) < len(candidates):
        raise ValueError("Need at least as many GPUs as candidates for eval stage.")
    procs = []
    for gpu, candidate in zip(gpu_list, candidates, strict=True):
        run_dir = find_latest_model_run(f"qwen-image-edit-2511-search-{candidate}")
        model_dir = run_dir / "model"
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing model dir for {candidate}: {model_dir}")
        config_paths = [MODEL_CFG, INT4_CFG, *materialize_search_stage("eval", candidate)]
        cmd = build_ptq_command(
            gpu,
            args.python_bin,
            args.micromamba_env,
            config_paths,
            extra_args=("--load-from", str(model_dir), "--skip-gen", "false", "--skip-eval", "false"),
        )
        log_path = LOG_ROOT / f"eval-{candidate}.log"
        print(f"[GPU {gpu}] {candidate}")
        print(cmd)
        procs.append((candidate, run_shell(cmd, log_path=log_path, wait=False), log_path))
        time.sleep(2)
    if args.wait:
        for candidate, proc, log_path in procs:
            code = proc.wait()
            print(f"{candidate}: exit_code={code} log={log_path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_summary(args: argparse.Namespace) -> None:
    rows = []
    for candidate in parse_csv_list(args.candidates):
        try:
            run_dir = find_latest_named_run(f"qwen-image-edit-2511-search-{candidate}")
            results = load_json(run_dir / "results.json")
            key = next(iter(results))
            metrics = results[key]["with_orig"]
            rows.append(
                {
                    "candidate": candidate,
                    "run_dir": str(run_dir),
                    "psnr": metrics.get("psnr"),
                    "lpips": metrics.get("lpips"),
                    "ssim": metrics.get("ssim"),
                    "fid": metrics.get("fid"),
                }
            )
        except Exception as exc:
            rows.append({"candidate": candidate, "error": str(exc)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.stage == "refs":
        stage_refs(args)
    elif args.stage == "launch":
        stage_launch(args)
    elif args.stage == "eval":
        stage_eval(args)
    else:
        stage_summary(args)


if __name__ == "__main__":
    main()
