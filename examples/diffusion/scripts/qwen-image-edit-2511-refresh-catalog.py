#!/usr/bin/env python3
import html
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
MODEL_ROOT = Path(
    os.environ.get(
        'QWEN_IMAGE_EDIT_2511_INT4_MODEL_ROOT',
        str((WORKSPACE_ROOT / 'models' / 'Qwen-iImage-edit-2511-int4').resolve()),
    )
)
BENCH_ROOT = REPO_ROOT / 'runs' / 'benchmarks' / 'qwen-image-edit-2511'
RUN_REPORT_ROOT = REPO_ROOT / 'runs' / 'reports' / 'qwen-image-edit-2511'
EXPORT_ROOT = REPO_ROOT / 'exports' / 'nunchaku' / 'search'
REPORT_SCRIPT = REPO_ROOT / 'examples' / 'diffusion' / 'scripts' / 'qwen-image-edit-2511-report.py'
REF_ROOT = BENCH_ROOT / 'bf16-fullgpu-50steps-qwen-image-edit-search-holdout-24-all24-gpu0-rerun'
ALL_INHOUSE_REPORT_DIR = RUN_REPORT_ROOT / 'all-inhouse-comparable-ref-bf16-rerun'
TOP3_REPORT_DIR = RUN_REPORT_ROOT / 'top3-comparable-ref-bf16-rerun'
QUALITY_VS_FAST_R128_REPORT_DIR = RUN_REPORT_ROOT / 'quality-r64-vs-fast-r128-ref-bf16-rerun'
BALANCED_R32_B125_REPORT_DIR = RUN_REPORT_ROOT / 'bf16-rerun-vs-balanced-r32-b125'
MID_R32_REPORT_DIR = RUN_REPORT_ROOT / 'bf16-rerun-vs-mid-r32-rerun'

REPORT_TARGET_DIRS = {
    'all_inhouse_comparable': ALL_INHOUSE_REPORT_DIR,
    'top3_comparable': TOP3_REPORT_DIR,
    'quality_vs_fast_r128': QUALITY_VS_FAST_R128_REPORT_DIR,
    'balanced_r32_b125': BALANCED_R32_B125_REPORT_DIR,
    'mid_r32': MID_R32_REPORT_DIR,
}

REPORT_LINK_DIRS = {
    'all_inhouse_comparable': MODEL_ROOT / 'reports' / 'all-inhouse-comparable',
    'top3_comparable': MODEL_ROOT / 'reports' / 'top3-comparable',
    'quality_vs_fast_r128': MODEL_ROOT / 'reports' / 'quality-vs-fast-r128',
    'balanced_r32_b125': MODEL_ROOT / 'reports' / 'balanced-r32-b125',
    'mid_r32': MODEL_ROOT / 'reports' / 'mid-r32',
}

REPORTS = {key: str((target_dir / 'index.html').resolve()) for key, target_dir in REPORT_TARGET_DIRS.items()}

VARIANT_REPORTS = {
    'balanced-r32-b125': REPORTS['balanced_r32_b125'],
    'mid-r32': REPORTS['mid_r32'],
}

FOCUS_VARIANTS = ['balanced-r32', 'quality-r64', 'quality-r96', 'quality-r96-i128', 'quality-r128-b15']
NEW_VARIANTS = ['quality-r96-i128', 'quality-r128-b15']
DEFAULT_QUALITY_VARIANT = 'quality-r64'
TOP3_HYBRID_WEIGHTS = {
    'fid': 0.45,
    'lpips': 0.20,
    'psnr': 0.20,
    'ssim': 0.15,
}
TOP3_METRIC_DIRECTIONS = {
    'fid': 'min',
    'lpips': 'min',
    'psnr': 'max',
    'ssim': 'max',
}

VARIANT_SPECS = [
    {
        'variant': 'balanced-r32',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_balanced_r32_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-balanced-r32-gptq-int4.safetensors',
        'benchmark': 'balanced-r32-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu1',
    },
    {
        'variant': 'balanced-r32-i64',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_balanced_r32_i64_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-balanced-r32-i64-gptq-int4.safetensors',
        'benchmark': 'balanced-r32-i64-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu0',
    },
    {
        'variant': 'balanced-r32-b125',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_balanced_r32_b125_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-balanced-r32-b125-gptq-int4.safetensors',
        'benchmark': 'balanced-r32-b125-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu1',
    },
    {
        'variant': 'fast-r32',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_fast_r32_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-fast-r32-gptq-int4.safetensors',
        'benchmark': 'fast-r32-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu1-rerun',
    },
    {
        'variant': 'fast-r64',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_fast_r64_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-fast-r64-gptq-int4.safetensors',
        'benchmark': 'fast-r64-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu0',
    },
    {
        'variant': 'fast-r128',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_fast_r128_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-fast-r128-gptq-int4.safetensors',
        'benchmark': 'fast-r128-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu3',
    },
    {
        'variant': 'mid-r32',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_mid_r32_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-mid-r32-gptq-int4.safetensors',
        'benchmark': 'mid-r32-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu2-rerun',
    },
    {
        'variant': 'mid-r128',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_mid_r128_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-mid-r128-gptq-int4.safetensors',
        'benchmark': 'mid-r128-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu0',
    },
    {
        'variant': 'quality-r32',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_quality_r32_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-quality-r32-gptq-int4.safetensors',
        'benchmark': 'quality-r32-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu2',
    },
    {
        'variant': 'quality-r64',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_quality_r64_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-quality-r64-gptq-int4.safetensors',
        'benchmark': 'quality-r64-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu3',
    },
    {
        'variant': 'quality-r96',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_quality_r96_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-quality-r96-gptq-int4.safetensors',
        'benchmark': 'quality-r96-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu2',
    },
    {
        'variant': 'quality-r96-i128',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_quality_r96_i128_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-quality-r96-i128-gptq-int4.safetensors',
        'benchmark': 'quality-r96-i128-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu2',
    },
    {
        'variant': 'quality-r128-b15',
        'checkpoint': 'nunchaku_qwen_image_edit_2511_quality_r128_b15_int4.safetensors',
        'export': 'qwen-image-edit-2511-search-quality-r128-b15-gptq-int4.safetensors',
        'benchmark': 'quality-r128-b15-nunchaku-50steps-qwen-image-edit-search-holdout-24-all24-gpu3',
    },
]


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
            raise IsADirectoryError(f'refusing to replace real directory: {link_path}')
        link_path.unlink()
    os.symlink(target_path, link_path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_quality_metrics(benchmark_dir: Path) -> dict:
    results_path = benchmark_dir / 'results.json'
    if not results_path.exists():
        return {}
    data = load_json(results_path)
    if not data:
        return {}
    first_key = next(iter(data))
    return data.get(first_key, {}).get('with_orig', {}) or {}


def load_runtime_metrics(benchmark_dir: Path) -> dict:
    summary_path = benchmark_dir / 'summary.json'
    if not summary_path.exists():
        return {}
    data = load_json(summary_path)
    keys = ('load_s', 'avg_inference_s', 'p50_inference_s', 'p95_inference_s', 'max_peak_process_gpu_mib')
    return {key: data[key] for key in keys if key in data}


def format_float(value, digits=4) -> str:
    if value is None:
        return '-'
    return f'{float(value):.{digits}f}'


def size_gib(size_bytes) -> str:
    if size_bytes is None:
        return '-'
    return f'{float(size_bytes) / (1024 ** 3):.2f}'


def best_value(values, direction):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return min(valid) if direction == 'min' else max(valid)


def sync_report_links() -> None:
    for key, target_dir in REPORT_TARGET_DIRS.items():
        ensure_link(REPORT_LINK_DIRS[key], target_dir)


def hydrate_variants():
    variants = []
    for spec in VARIANT_SPECS:
        source_path = EXPORT_ROOT / spec['export']
        benchmark_dir = (BENCH_ROOT / spec['benchmark']).resolve()
        if not source_path.exists():
            print(f'[refresh-catalog] skipping {spec["variant"]}: missing export {source_path}', file=sys.stderr)
            continue
        if not benchmark_dir.exists():
            print(f'[refresh-catalog] skipping {spec["variant"]}: missing benchmark {benchmark_dir}', file=sys.stderr)
            continue
        link_path = MODEL_ROOT / spec['checkpoint']
        ensure_link(link_path, source_path)
        variants.append(
            {
                'variant': spec['variant'],
                'checkpoint': str(link_path),
                'link_path': str(link_path),
                'source_path': str(source_path.resolve()),
                'size_bytes': source_path.stat().st_size,
                'benchmark_dir': str(benchmark_dir),
                'metrics_status': 'comparable_bf16_rerun',
                'metrics_ref_root': str(REF_ROOT),
                'quality_metrics': load_quality_metrics(benchmark_dir),
                'runtime_metrics': load_runtime_metrics(benchmark_dir),
            }
        )
    return variants


def render_focus_table(entries, include_size=True, wrapper_class='summary-strip') -> str:
    by_name = {entry['variant']: entry for entry in entries}
    focus_entries = [by_name[name] for name in FOCUS_VARIANTS if name in by_name]
    specs = []
    if include_size:
        specs.append(('size_bytes', 'Size(GiB)', 'meta', 'min'))
    specs.extend(
        [
            ('load_s', 'Load(s)', 'runtime', 'min'),
            ('avg_inference_s', 'Avg(s)', 'runtime', 'min'),
            ('max_peak_process_gpu_mib', 'Peak VRAM(MiB)', 'runtime', 'min'),
            ('psnr', 'PSNR', 'quality', 'max'),
            ('ssim', 'SSIM', 'quality', 'max'),
            ('lpips', 'LPIPS', 'quality', 'min'),
            ('fid', 'FID', 'quality', 'min'),
        ]
    )
    head = ''.join(f'<th>{html.escape(str(entry["variant"]))}</th>' for entry in focus_entries)
    rows = []
    for key, label, bucket, direction in specs:
        values = []
        for entry in focus_entries:
            if bucket == 'meta':
                raw = entry.get('size_bytes')
                values.append((raw / (1024 ** 3)) if raw is not None else None)
            else:
                raw = entry[f'{bucket}_metrics'].get(key)
                values.append(float(raw) if raw is not None else None)
        best = best_value(values, direction)
        cells = []
        for value in values:
            cls = ''
            if best is not None and value is not None and abs(value - best) < 1e-9:
                cls = ' class="best"'
            digits = 2 if key == 'size_bytes' else 3 if bucket == 'runtime' else 4
            cells.append(f'<td{cls}>{html.escape(format_float(value, digits=digits))}</td>')
        rows.append(f'<tr><th class="metric-name">{html.escape(label)}</th>{"".join(cells)}</tr>')
    return (
        f'<section class="{wrapper_class}">'
        '<h2>Focused Horizontal Comparison</h2>'
        '<p class="note">Current shortlist: legacy anchors plus the two newly finished exports. '
        'Lower is better for size, runtime, VRAM, LPIPS, and FID. Higher is better for PSNR and SSIM.</p>'
        '<div class="table-wrap focus-wrap"><table class="focus-table">'
        f'<thead><tr><th>Metric</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table></div>'
        '</section>'
    )


def inject_focus_table(entries) -> None:
    html_path = ALL_INHOUSE_REPORT_DIR / 'index.html'
    text = html_path.read_text(encoding='utf-8')
    style_block = """
    .summary-strip { background: white; border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 0 0 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .summary-strip h2 { margin: 0 0 8px; font-size: 20px; }
    .focus-wrap { margin-bottom: 0; }
    .focus-table { min-width: 920px; }
    .focus-table th.metric-name { min-width: 140px; position: sticky; left: 0; z-index: 2; background: #f6f2ea; }
    .focus-table td.best { background: #e8f3ea; color: #1d5d38; font-weight: 700; }
"""
    if '.summary-strip' not in text:
        text = text.replace('  </style>', style_block + '  </style>')
    section = render_focus_table(entries, include_size=True, wrapper_class='summary-strip')
    marker_start = '<!-- focus-comparison:start -->'
    marker_end = '<!-- focus-comparison:end -->'
    wrapped = f'{marker_start}\n{section}\n{marker_end}'
    if marker_start in text and marker_end in text:
        before, _, tail = text.partition(marker_start)
        _, _, after = tail.partition(marker_end)
        text = before + wrapped + after
    else:
        text = text.replace('<div class="table-wrap">', wrapped + '\n    <div class="table-wrap">', 1)
    html_path.write_text(text, encoding='utf-8')


def generate_report(entries, output_dir: Path, title: str) -> None:
    run_dirs = [entry['benchmark_dir'] for entry in entries]
    labels = [entry['variant'] for entry in entries]
    cmd = [
        sys.executable,
        str(REPORT_SCRIPT),
        '--ref-root',
        str(REF_ROOT),
        '--run-dirs',
        *run_dirs,
        '--labels',
        *labels,
        '--output',
        str(output_dir),
        '--title',
        title,
    ]
    subprocess.run(cmd, check=True)


def generate_all_inhouse_report(entries) -> None:
    generate_report(
        entries,
        ALL_INHOUSE_REPORT_DIR,
        'Qwen-Image-Edit-2511 All In-House Comparable Report (BF16 Rerun Ref)',
    )
    inject_focus_table(entries)


def rank_variants(entries, metric: str, direction: str) -> dict[str, int]:
    valid_entries = [entry for entry in entries if entry['quality_metrics'].get(metric) is not None]
    sorted_entries = sorted(
        valid_entries,
        key=lambda entry: float(entry['quality_metrics'][metric]),
        reverse=(direction == 'max'),
    )
    return {entry['variant']: idx + 1 for idx, entry in enumerate(sorted_entries)}


def select_top3_variants(entries) -> list[str]:
    by_name = {entry['variant']: entry for entry in entries}
    available = [
        entry
        for entry in entries
        if all(entry['quality_metrics'].get(metric) is not None for metric in TOP3_METRIC_DIRECTIONS)
    ]
    if not available:
        return [entry['variant'] for entry in entries[:3]]

    selected: list[str] = []
    if DEFAULT_QUALITY_VARIANT in by_name:
        selected.append(DEFAULT_QUALITY_VARIANT)

    best_fid_entry = min(available, key=lambda entry: float(entry['quality_metrics']['fid']))
    if best_fid_entry['variant'] not in selected:
        selected.append(best_fid_entry['variant'])

    ranks = {
        metric: rank_variants(available, metric, direction)
        for metric, direction in TOP3_METRIC_DIRECTIONS.items()
    }
    hybrid_pool = [entry for entry in available if entry['variant'] not in selected]
    if hybrid_pool:
        hybrid_entry = min(
            hybrid_pool,
            key=lambda entry: sum(
                ranks[metric].get(entry['variant'], len(available) + 1) * weight
                for metric, weight in TOP3_HYBRID_WEIGHTS.items()
            ),
        )
        selected.insert(1 if selected else 0, hybrid_entry['variant'])

    for entry in available:
        if entry['variant'] not in selected:
            selected.append(entry['variant'])
        if len(selected) == 3:
            break
    return selected[:3]


def generate_top3_report(entries) -> list[str]:
    top3_variants = select_top3_variants(entries)
    by_name = {entry['variant']: entry for entry in entries}
    top3_entries = [by_name[name] for name in top3_variants if name in by_name]
    title = 'Qwen-Image-Edit-2511 Top-3 Comparable Report ' + f'({", ".join(entry["variant"] for entry in top3_entries)})'
    generate_report(top3_entries, TOP3_REPORT_DIR, title)
    return [entry['variant'] for entry in top3_entries]


def build_metrics_payload(entries, generated_at: str, top3_variants: list[str]) -> dict:
    return {
        'model_family': 'Qwen-Image-Edit-2511',
        'format': 'Nunchaku int4 safetensors',
        'root': str(MODEL_ROOT),
        'generated_at_utc': generated_at,
        'default_checkpoint': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_current_best_quality_int4.safetensors'),
        'default_run_tag': 'current-best-quality',
        'recommended': {
            'default_quality': 'quality-r64',
            'best_fid': 'fast-r128',
        },
        'newly_added_variants': NEW_VARIANTS,
        'focus_variants': FOCUS_VARIANTS,
        'top3_variants': top3_variants,
        'aliases': [
            {
                'alias': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_current_best_quality_int4.safetensors'),
                'target': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_quality_r64_int4.safetensors'),
            },
            {
                'alias': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_current_best_fid_int4.safetensors'),
                'target': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_fast_r128_int4.safetensors'),
            },
        ],
        'metrics_ref_root': str(REF_ROOT),
        'catalog_html': str((MODEL_ROOT / 'index.html').resolve()),
        'full_report': REPORTS['all_inhouse_comparable'],
        'top3_report': REPORTS['top3_comparable'],
        'focused_report': REPORTS['quality_vs_fast_r128'],
        'variant_reports': VARIANT_REPORTS,
        'reports': REPORTS,
        'variants': [
            {
                'variant': entry['variant'],
                'checkpoint': entry['checkpoint'],
                'size_bytes': entry['size_bytes'],
                'benchmark_dir': entry['benchmark_dir'],
                'metrics_status': entry['metrics_status'],
                'metrics_ref_root': entry['metrics_ref_root'],
                'quality_metrics': entry['quality_metrics'],
                'runtime_metrics': entry['runtime_metrics'],
            }
            for entry in entries
        ],
    }


def build_manifest(entries, generated_at: str) -> dict:
    return {
        'model_family': 'Qwen-Image-Edit-2511',
        'format': 'Nunchaku int4 safetensors',
        'layout': 'symlinked_from_deepcompressor_exports',
        'root': str(MODEL_ROOT),
        'generated_at_utc': generated_at,
        'catalog_html': str((MODEL_ROOT / 'index.html').resolve()),
        'reports': REPORTS,
        'variants': [
            {
                'variant': entry['variant'],
                'link_path': entry['link_path'],
                'source_path': entry['source_path'],
                'size_bytes': entry['size_bytes'],
            }
            for entry in entries
        ],
        'aliases': [
            {
                'alias': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_current_best_quality_int4.safetensors'),
                'target': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_quality_r64_int4.safetensors'),
            },
            {
                'alias': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_current_best_fid_int4.safetensors'),
                'target': str(MODEL_ROOT / 'nunchaku_qwen_image_edit_2511_fast_r128_int4.safetensors'),
            },
        ],
        'metrics_json': str((MODEL_ROOT / 'metrics.json').resolve()),
        'metrics_js': str((MODEL_ROOT / 'metrics.js').resolve()),
    }


def render_catalog(entries, payload: dict) -> str:
    added_variants_markup = ', '.join(f'<strong>{html.escape(name)}</strong>' for name in NEW_VARIANTS)
    top3_text = ', '.join(payload.get('top3_variants', []))
    hero_badges = [
        f'Default quality: {payload["recommended"]["default_quality"]}',
        f'Best FID: {payload["recommended"]["best_fid"]}',
        f'Top-3: {top3_text}',
        f'Variants: {len(entries)}',
        f'Added now: {", ".join(NEW_VARIANTS)}',
        f'Generated: {payload["generated_at_utc"]}',
    ]
    by_name = {entry['variant']: entry for entry in entries}
    quality_name = Path(str(by_name['quality-r64']['checkpoint'])).name
    fid_name = Path(str(by_name['fast-r128']['checkpoint'])).name
    balanced_name = Path(str(by_name['balanced-r32']['checkpoint'])).name
    badges = ''.join(f'<span class="badge">{html.escape(badge)}</span>' for badge in hero_badges)
    report_links = ''.join(
        f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
        for label, href in [
            ('All in-house comparable report', 'reports/all-inhouse-comparable/index.html'),
            ('Top-3 comparable report', 'reports/top3-comparable/index.html'),
            ('Quality-r64 vs fast-r128 focus report', 'reports/quality-vs-fast-r128/index.html'),
            ('BF16 vs balanced-r32-b125', 'reports/balanced-r32-b125/index.html'),
            ('BF16 vs mid-r32', 'reports/mid-r32/index.html'),
            ('metrics.json', 'metrics.json'),
            ('metrics.js', 'metrics.js'),
            ('manifest.json', 'manifest.json'),
        ]
    )
    full_rows = []
    for entry in entries:
        quality = entry['quality_metrics']
        runtime = entry['runtime_metrics']
        checkpoint_name = Path(str(entry['checkpoint'])).name
        full_rows.append(
            '<tr>'
            f'<td><strong>{html.escape(str(entry["variant"]))}</strong></td>'
            f'<td>{size_gib(entry["size_bytes"])}</td>'
            f'<td>{format_float(quality.get("psnr"), digits=4)}</td>'
            f'<td>{format_float(quality.get("ssim"), digits=4)}</td>'
            f'<td>{format_float(quality.get("lpips"), digits=4)}</td>'
            f'<td>{format_float(quality.get("fid"), digits=4)}</td>'
            f'<td>{format_float(runtime.get("load_s"), digits=3)}</td>'
            f'<td>{format_float(runtime.get("avg_inference_s"), digits=3)}</td>'
            f'<td>{format_float(runtime.get("max_peak_process_gpu_mib"), digits=0)}</td>'
            f'<td><code>{html.escape(checkpoint_name)}</code></td>'
            '</tr>'
        )
    focus_table = render_focus_table(entries, include_size=True, wrapper_class='card wide')
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qwen-Image-Edit-2511 In-House Int4 Catalog</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #ffffff;
      --line: #d9d0c2;
      --text: #16130f;
      --muted: #6a5f53;
      --accent: #8a4f1d;
      --accent-soft: #f2e3d1;
      --best-bg: #e8f3ea;
      --best-fg: #1d5d38;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", sans-serif; background: linear-gradient(180deg, #f3eee5 0%, var(--bg) 100%); color: var(--text); }}
    .wrap {{ max-width: 1520px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 0 0 10px; font-size: 24px; }}
    p {{ line-height: 1.55; color: var(--muted); }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 0; }}
    .badge {{ background: var(--accent-soft); color: var(--accent); border: 1px solid #dfc5a9; border-radius: 999px; padding: 8px 12px; font-size: 14px; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 18px; margin-top: 18px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    .card.wide {{ margin-top: 18px; }}
    .links a {{ display: block; margin: 10px 0; color: var(--accent); text-decoration: none; font-weight: 600; }}
    .links a:hover {{ text-decoration: underline; }}
    .table-wrap {{ overflow-x: auto; margin-top: 16px; border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #ece4d8; text-align: left; font-size: 14px; }}
    th {{ background: #efe7da; }}
    tr:hover td {{ background: #fbf8f3; }}
    .focus-table th.metric-name {{ position: sticky; left: 0; z-index: 2; background: #f7f1e7; min-width: 140px; }}
    .focus-table td.best {{ background: var(--best-bg); color: var(--best-fg); font-weight: 700; }}
    code {{ font-size: 12px; word-break: break-all; }}
    .note {{ margin-top: 12px; font-size: 13px; }}
    .callout {{ margin-top: 14px; padding: 14px 16px; border-left: 4px solid var(--accent); background: #fcf5eb; border-radius: 10px; color: #4d4033; }}
    @media (max-width: 1080px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Qwen-Image-Edit-2511 In-House Int4 Catalog</h1>
      <p>Only in-house DeepCompressor SVDQuant W4A4 + GPTQ exports are listed here. All quality metrics below are computed against the corrected BF16 rerun baseline on the same 24-image holdout set with 50 steps. Higher is better for PSNR/SSIM. Lower is better for LPIPS/FID.</p>
      <div class="badges">{badges}</div>
      <div class="callout">This catalog tracks the current search focus around {added_variants_markup}, and the main all-in-house report is rebuilt whenever any finished export has matching benchmark results.</div>
    </section>
    <section class="grid">
      <div class="card links">
        <h2>Reports</h2>
        {report_links}
      </div>
      <div class="card">
        <h2>Recommended checkpoints</h2>
        <p><strong>Default quality</strong><br><code>{html.escape(quality_name)}</code></p>
        <p><strong>Best FID</strong><br><code>{html.escape(fid_name)}</code></p>
        <p><strong>Stable small-footprint alternative</strong><br><code>{html.escape(balanced_name)}</code></p>
        <p class="note">The focused comparison below is the quickest way to judge the new candidates against the current anchors.</p>
      </div>
    </section>
    {focus_table}
    <section class="card wide">
      <h2>All Comparable Variants</h2>
      <div class="table-wrap"><table class="full-table">
        <thead>
          <tr>
            <th>Variant</th><th>Size(GiB)</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>FID</th>
            <th>Load(s)</th><th>Avg(s)</th><th>Peak VRAM(MiB)</th><th>Checkpoint</th>
          </tr>
        </thead>
        <tbody>{''.join(full_rows)}</tbody>
      </table></div>
    </section>
  </div>
</body>
</html>
'''


def write_catalog(entries) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    top3_variants = select_top3_variants(entries)
    metrics_payload = build_metrics_payload(entries, generated_at, top3_variants)
    manifest_payload = build_manifest(entries, generated_at)
    (MODEL_ROOT / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    (MODEL_ROOT / 'metrics.js').write_text(
        'const qwenImageEdit2511Int4Metrics = ' + json.dumps(metrics_payload, indent=2, ensure_ascii=False) + ';\n\n'
        + 'if (typeof module !== "undefined" && module.exports) {\n  module.exports = qwenImageEdit2511Int4Metrics;\n}\n'
        + 'if (typeof globalThis !== "undefined") {\n  globalThis.qwenImageEdit2511Int4Metrics = qwenImageEdit2511Int4Metrics;\n}\n',
        encoding='utf-8',
    )
    (MODEL_ROOT / 'manifest.json').write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    (MODEL_ROOT / 'index.html').write_text(render_catalog(entries, metrics_payload), encoding='utf-8')


def main() -> None:
    entries = hydrate_variants()
    generate_all_inhouse_report(entries)
    generate_top3_report(entries)
    sync_report_links()
    write_catalog(entries)
    print(f'Refreshed all-inhouse/top3 reports and catalog under {MODEL_ROOT}')


if __name__ == '__main__':
    main()
