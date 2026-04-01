#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_YAML = REPO_ROOT / "examples/diffusion/prompts/qwen-image-edit-search-holdout.yaml"
DEFAULT_REF_ROOT = REPO_ROOT / "references/torch.bfloat16/qwen-image-edit-2511/fmeuler50-cfg4.0"
METRIC_KEYS = ("psnr", "ssim", "lpips", "fid")
RUNTIME_KEYS = (
    ("load_s", "Load(s)"),
    ("avg_inference_s", "Avg(s)"),
    ("p50_inference_s", "P50(s)"),
    ("p95_inference_s", "P95(s)"),
    ("max_peak_process_gpu_mib", "Peak VRAM(MiB)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a side-by-side visual report for Qwen Image Edit search runs.")
    parser.add_argument("--dataset-yaml", type=Path, default=DEFAULT_DATASET_YAML)
    parser.add_argument("--ref-root", type=Path, default=DEFAULT_REF_ROOT)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Qwen-Image-Edit-2511 Search Report")
    return parser.parse_args()


def is_remote_path(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def load_prompt_rows(dataset_yaml: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for key, row in data.items():
        image_value = str(row["image"])
        if is_remote_path(image_value):
            input_image = image_value
        else:
            input_image = str((dataset_yaml.parent / image_value).resolve())
        rows.append(
            {
                "key": key,
                "prompt": row["prompt"],
                "input_image": input_image,
            }
        )
    return rows


def find_sample_dir(root: Path, dataset_name: str) -> Path:
    base = root / "samples" / "YAML"
    if not base.exists():
        raise FileNotFoundError(f"Missing sample root: {base}")
    candidates: list[tuple[int, str, Path]] = []
    for path in base.iterdir():
        if not path.is_dir() or not path.name.startswith(f"{dataset_name}-"):
            continue
        png_count = len(list(path.glob("*.png")))
        candidates.append((png_count, path.name, path))
    if not candidates:
        raise FileNotFoundError(f"No sample directory under {base} for dataset {dataset_name}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_quality_metrics(run_dir: Path) -> dict[str, float]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        return {}
    first_key = next(iter(data))
    return data.get(first_key, {}).get("with_orig", {}) or {}


def load_runtime_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {key: data[key] for key, _ in RUNTIME_KEYS if key in data}


def derive_label(run_dir: Path) -> str:
    parent = run_dir.parent.name.replace(".RUNNING", "")
    match = re.search(r"search-(.+)$", parent)
    if match:
        return match.group(1)
    return run_dir.name.replace("run-", "")


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def collect_variants(sample_dir: Path, sample_key: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(sample_key)}(?:-(\d+))?\.png$")
    variants: list[tuple[int, str]] = []
    for path in sample_dir.glob(f"{sample_key}*.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        idx = int(match.group(1) or 0)
        variants.append((idx, path.stem))
    variants.sort(key=lambda item: item[0])
    return [stem for _, stem in variants]


def first_existing_variants(sample_key: str, sample_dirs: list[Path]) -> list[str]:
    for sample_dir in sample_dirs:
        variants = collect_variants(sample_dir, sample_key)
        if variants:
            return variants
    return [sample_key]


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_runtime(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


def ensure_link(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = target_path.resolve()
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.is_symlink() and link_path.resolve() == target_path:
                return
        except FileNotFoundError:
            pass
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target_path, link_path)


def prepare_assets(output_dir: Path, ref_dir: Path, run_infos: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    input_root = output_dir / "assets" / "inputs"
    ref_root = output_dir / "assets" / "bf16"
    ensure_link(ref_root, ref_dir)
    for info in run_infos:
        asset_dir = output_dir / "assets" / sanitize_name(str(info["label"]))
        ensure_link(asset_dir, Path(info["sample_dir"]))
        info["asset_dir"] = asset_dir.relative_to(output_dir).as_posix()
    input_links: dict[Path, Path] = {}
    for row in rows:
        input_value = str(row["input_image"])
        if is_remote_path(input_value):
            row["input_relpath"] = input_value
        else:
            input_path = Path(input_value).resolve()
            if input_path.exists():
                if input_path not in input_links:
                    input_link = input_root / input_path.name
                    ensure_link(input_link, input_path)
                    input_links[input_path] = input_link
                row["input_relpath"] = input_links[input_path].relative_to(output_dir).as_posix()
            else:
                row["input_relpath"] = ""
        row["reference_relpath"] = f"assets/bf16/{row['sample_id']}.png"
        for item, info in zip(row["outputs"], run_infos, strict=True):
            item["relpath"] = f"{info['asset_dir']}/{row['sample_id']}.png"


def render_html(title: str, run_infos: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    runtime_headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in RUNTIME_KEYS)
    metric_headers = "".join(f"<th>{html.escape(key.upper())}</th>" for key in METRIC_KEYS)
    metric_rows: list[str] = []
    for info in run_infos:
        runtime = info["runtime"]
        quality = info["quality"]
        runtime_cells = "".join(
            f"<td>{html.escape(format_runtime(runtime.get(key)))}</td>" for key, _ in RUNTIME_KEYS
        )
        metric_cells = "".join(f"<td>{html.escape(format_metric(quality.get(key)))}</td>" for key in METRIC_KEYS)
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(str(info['label']))}</td>"
            f"{runtime_cells}"
            f"{metric_cells}"
            f"<td><code>{html.escape(str(info['run_dir']))}</code></td>"
            "</tr>"
        )
    gallery_rows: list[str] = []
    for row in rows:
        image_cards: list[str] = []
        all_images = [
            ("Input", row["input_relpath"], bool(row["input_relpath"])),
            ("BF16 Ref", row["reference_relpath"], bool(row["reference_image"])),
            *[(item["label"], item["relpath"], bool(item["image"])) for item in row["outputs"]],
        ]
        for label, relpath, exists in all_images:
            if exists:
                body = f'<img src="{html.escape(relpath)}" alt="{html.escape(str(row["sample_id"]))} {html.escape(str(label))}">'
            else:
                body = '<div class="missing">missing</div>'
            image_cards.append(
                '<div class="image-card">'
                f'<div class="image-label">{html.escape(str(label))}</div>'
                f'{body}'
                '</div>'
            )
        gallery_rows.append(
            '<section class="sample">'
            f'<h2>{html.escape(str(row["sample_id"]))}</h2>'
            f'<div class="meta"><strong>Prompt:</strong> {html.escape(str(row["prompt"]))}</div>'
            '<div class="image-grid">'
            + ''.join(image_cards)
            + '</div>'
            '</section>'
        )
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111; background: #faf8f4; }}
    .page {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    p.note {{ margin-top: 0; color: #444; line-height: 1.6; }}
    .table-wrap {{ overflow-x: auto; margin: 16px 0 28px; border: 1px solid #ddd; border-radius: 12px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1100px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 10px 12px; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #f0ece3; position: sticky; top: 0; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    code {{ font-size: 12px; word-break: break-all; }}
    .sample {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 0 0 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    .sample h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .meta {{ margin-bottom: 14px; color: #333; line-height: 1.5; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: 14px; align-items: start; }}
    .image-card {{ display: flex; flex-direction: column; gap: 8px; min-width: 0; }}
    .image-label {{ font-size: 13px; font-weight: 600; color: #555; }}
    img {{ width: 100%; border-radius: 10px; border: 1px solid #ddd; background: #f4f4f4; }}
    .missing {{ display: grid; place-items: center; min-height: 220px; border: 1px dashed #bbb; border-radius: 10px; color: #777; background: #fcfcfc; }}
    @media (max-width: 900px) {{
      .page {{ padding: 16px; }}
      .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p class="note">Runtime columns come from <code>summary.json</code>. Quality columns come from optional <code>results.json</code> <code>with_orig</code> scores against the BF16 reference set. Higher is better for PSNR/SSIM. Lower is better for LPIPS/FID.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Run</th>{runtime_headers}{metric_headers}<th>Run Dir</th></tr>
        </thead>
        <tbody>
          {''.join(metric_rows)}
        </tbody>
      </table>
    </div>
    {''.join(gallery_rows)}
  </div>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    labels = args.labels or []
    if labels and len(labels) != len(args.run_dirs):
        raise ValueError("--labels must match --run-dirs in length")
    prompt_rows = load_prompt_rows(args.dataset_yaml.resolve())
    dataset_name = args.dataset_yaml.stem
    ref_dir = find_sample_dir(args.ref_root.resolve(), dataset_name)
    run_infos: list[dict[str, object]] = []
    for idx, run_dir in enumerate(args.run_dirs):
        run_dir = run_dir.resolve()
        label = labels[idx] if idx < len(labels) else derive_label(run_dir)
        run_infos.append(
            {
                "label": label,
                "run_dir": run_dir,
                "sample_dir": find_sample_dir(run_dir, dataset_name),
                "runtime": load_runtime_metrics(run_dir),
                "quality": load_quality_metrics(run_dir),
            }
        )
    sample_dirs = [ref_dir, *[info["sample_dir"] for info in run_infos]]
    rows: list[dict[str, object]] = []
    for prompt_row in prompt_rows:
        variants = first_existing_variants(prompt_row["key"], sample_dirs)
        for variant in variants:
            row = {
                "sample_id": variant,
                "key": prompt_row["key"],
                "prompt": prompt_row["prompt"],
                "input_image": prompt_row["input_image"],
                "reference_image": str(ref_dir / f"{variant}.png"),
                "outputs": [],
            }
            for info in run_infos:
                output_path = Path(info["sample_dir"]) / f"{variant}.png"
                row["outputs"].append(
                    {
                        "label": info["label"],
                        "image": str(output_path) if output_path.exists() else "",
                    }
                )
            rows.append(row)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_assets(output_dir, ref_dir, run_infos, rows)
    (output_dir / "index.html").write_text(render_html(args.title, run_infos, rows), encoding="utf-8")
    manifest = {
        "title": args.title,
        "dataset_yaml": str(args.dataset_yaml.resolve()),
        "ref_dir": str(ref_dir),
        "runs": [
            {
                "label": info["label"],
                "run_dir": str(info["run_dir"]),
                "sample_dir": str(info["sample_dir"]),
                "runtime": info["runtime"],
                "quality": info["quality"],
                "asset_dir": str(info["asset_dir"]),
            }
            for info in run_infos
        ],
        "num_rows": len(rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
