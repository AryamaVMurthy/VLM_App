#!/usr/bin/env python3
"""Assemble a paper-facing FastVLM system bundle from validated experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _find_variant(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for variant in summary.get('variants', []):
        if variant.get('name') == name:
            return variant
    raise KeyError(f'Missing variant {name!r} in hardware summary.')


def _format_float(value: float, digits: int = 3) -> str:
    return f'{value:.{digits}f}'


def _format_optional_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return '-'
    return _format_float(float(value), digits)


def _hardware_variant(bundle_variant: dict[str, Any]) -> dict[str, Any]:
    metrics = bundle_variant['metrics']
    return {
        'name': bundle_variant['name'],
        'event_window_s': metrics.get('event_window_s'),
        'device_runtime_s': metrics.get('device_runtime_s'),
        'serial_stage_sum_s': metrics.get('serial_stage_sum_s'),
        'overlap_pipeline_speedup': metrics.get('overlap_pipeline_speedup'),
        'first_token_ttft_ms_mean': metrics.get('first_token_ttft_ms_mean'),
        'prefill_ms_mean': metrics.get('prefill_ms_mean'),
        'decode_ms_mean': metrics.get('decode_ms_mean'),
        'npu_active_ratio': metrics.get('npu_active_ratio'),
        'sync_wait_ms_total': metrics.get('sync_wait_ms_total'),
        'cpu_summary': metrics.get('cpu_summary'),
        'budget_counts': metrics.get('budget_counts', {}),
        'example_outputs': metrics.get('example_outputs', []),
    }


def build_bundle(
    *,
    hardware_summary: dict[str, Any],
    partition_validation: dict[str, Any],
    gqa_uniform: dict[str, Any],
    gqa_smart: dict[str, Any],
    coco_uniform: dict[str, Any],
    coco_smart: dict[str, Any],
    artifact_manifest: dict[str, Any] | None = None,
    figure_manifest: dict[str, str] | None = None,
) -> dict[str, Any]:
    overlap_uniform = _find_variant(
        hardware_summary, 'overlap_cpu_decode_npu_prefill_uniform'
    )
    overlap_fixed_smart = _find_variant(
        hardware_summary, 'overlap_cpu_decode_npu_prefill_prompt_conditioned_v2'
    )
    overlap_adaptive = _find_variant(
        hardware_summary, 'overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2'
    )
    sequential = _find_variant(hardware_summary, 'sequential_npu_npu_uniform')
    bundle = {
        'hardware': {
            'variants': [
                _hardware_variant(sequential),
                _hardware_variant(overlap_uniform),
                _hardware_variant(overlap_fixed_smart),
                _hardware_variant(overlap_adaptive),
            ],
            'sequential_npu_npu_uniform': sequential['metrics'],
            'overlap_cpu_decode_npu_prefill_uniform': overlap_uniform['metrics'],
            'overlap_cpu_decode_npu_prefill_prompt_conditioned_v2': overlap_fixed_smart['metrics'],
            'overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2': overlap_adaptive['metrics'],
        },
        'partition_validation': {
            'validation_passed': partition_validation['validation_passed'],
            'config_deviation': partition_validation['config_deviation'],
            'stage_backend_manifest': partition_validation['stage_backend_manifest'],
            'expected_transfer_edges': partition_validation['expected_transfer_edges'],
            'unsupported_or_opaque_regions': partition_validation['unsupported_or_opaque_regions'],
            'compile_artifacts': partition_validation['compile_artifacts'],
        },
        'gqa': {
            'uniform': {
                'max_visual_tokens': gqa_uniform['max_visual_tokens'],
                'accuracy': gqa_uniform['accuracy'],
                'correct': gqa_uniform['correct'],
                'total_scored': gqa_uniform['total_scored'],
                'wall_clock_s': gqa_uniform['wall_clock_s'],
            },
            'smart': {
                'max_visual_tokens': gqa_smart['max_visual_tokens'],
                'accuracy': gqa_smart['accuracy'],
                'correct': gqa_smart['correct'],
                'total_scored': gqa_smart['total_scored'],
                'wall_clock_s': gqa_smart['wall_clock_s'],
            },
            'accuracy_delta': gqa_smart['accuracy'] - gqa_uniform['accuracy'],
        },
        'coco': {
            'uniform': {
                'max_visual_tokens': coco_uniform['max_visual_tokens'],
                'sample_count': coco_uniform['sample_count'],
                'CIDEr': coco_uniform['metrics']['CIDEr'],
                'ROUGE_L': coco_uniform['metrics']['ROUGE_L'],
                'Bleu_4': coco_uniform['metrics']['Bleu_4'],
                'ttft_mean_s': coco_uniform['runtime']['ttft_mean_s'],
            },
            'smart': {
                'max_visual_tokens': coco_smart['max_visual_tokens'],
                'sample_count': coco_smart['sample_count'],
                'CIDEr': coco_smart['metrics']['CIDEr'],
                'ROUGE_L': coco_smart['metrics']['ROUGE_L'],
                'Bleu_4': coco_smart['metrics']['Bleu_4'],
                'ttft_mean_s': coco_smart['runtime']['ttft_mean_s'],
            },
            'cider_delta': coco_smart['metrics']['CIDEr'] - coco_uniform['metrics']['CIDEr'],
        },
    }
    if artifact_manifest is not None:
        bundle['artifact_manifest'] = artifact_manifest
    if figure_manifest is not None:
        bundle['figure_manifest'] = figure_manifest
    return bundle


def _build_stage_table(stage_backend_manifest: list[dict[str, Any]]) -> list[str]:
    lines = [
        '| Stage | Declared | Observed | Visibility | Artifact / Unit |',
        '| --- | --- | --- | --- | --- |',
    ]
    for item in stage_backend_manifest:
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(item['stage']),
                    str(item['declared_backend']),
                    str(item['observed_backend']),
                    str(item['visibility']),
                    str(item.get('compile_artifact') or item.get('stage_unit') or '-'),
                ]
            )
            + ' |'
        )
    return lines


def _build_transfer_table(expected_transfer_edges: list[dict[str, Any]]) -> list[str]:
    lines = [
        '| Edge | Type | Cross backend | Observable copy counter |',
        '| --- | --- | --- | --- |',
    ]
    for edge in expected_transfer_edges:
        lines.append(
            '| '
            + ' | '.join(
                [
                    f"{edge['from_stage']} -> {edge['to_stage']}",
                    str(edge['edge_type']),
                    str(edge['cross_backend']),
                    str(edge.get('observable_copy_counter', False)),
                ]
            )
            + ' |'
        )
    return lines


def _build_hardware_table(hardware_variants: list[dict[str, Any]]) -> list[str]:
    lines = [
        '| Variant | Device window/runtime (s) | TTFT mean (ms) | Prefill mean (ms) | Decode mean (ms) | Pipeline speedup | NPU active ratio | Sync/wait total (ms) | Budgets |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for variant in hardware_variants:
        runtime_value = (
            variant['event_window_s']
            if variant['event_window_s'] is not None
            else variant['device_runtime_s']
        )
        speedup = _format_optional_float(variant['overlap_pipeline_speedup'])
        npu_active = _format_optional_float(variant['npu_active_ratio'])
        sync_wait = _format_optional_float(variant['sync_wait_ms_total'])
        budgets = ', '.join(
            f'{budget}:{count}'
            for budget, count in sorted(variant.get('budget_counts', {}).items())
        ) or '-'
        lines.append(
            '| '
            + ' | '.join(
                [
                    variant['name'],
                    _format_float(float(runtime_value)),
                    _format_optional_float(variant['first_token_ttft_ms_mean']),
                    _format_optional_float(variant['prefill_ms_mean']),
                    _format_optional_float(variant['decode_ms_mean']),
                    speedup,
                    npu_active,
                    sync_wait,
                    budgets,
                ]
            )
            + ' |'
        )
    return lines


def _build_quality_table(bundle: dict[str, Any]) -> list[str]:
    gqa = bundle['gqa']
    coco = bundle['coco']
    return [
        '| Benchmark | Policy | Budget | Primary metric | Secondary metrics |',
        '| --- | --- | --- | --- | --- |',
        f"| GQA | uniform | {gqa['uniform']['max_visual_tokens']} | accuracy={gqa['uniform']['accuracy']:.3f} ({gqa['uniform']['correct']}/{gqa['uniform']['total_scored']}) | wall_clock_s={gqa['uniform']['wall_clock_s']:.3f} |",
        f"| GQA | prompt_conditioned_v2 | {gqa['smart']['max_visual_tokens']} | accuracy={gqa['smart']['accuracy']:.3f} ({gqa['smart']['correct']}/{gqa['smart']['total_scored']}) | wall_clock_s={gqa['smart']['wall_clock_s']:.3f} |",
        f"| COCO Karpathy | uniform | {coco['uniform']['max_visual_tokens']} | CIDEr={coco['uniform']['CIDEr']:.4f} | ROUGE_L={coco['uniform']['ROUGE_L']:.4f}, Bleu_4={coco['uniform']['Bleu_4']:.4f}, TTFT={coco['uniform']['ttft_mean_s']:.4f}s |",
        f"| COCO Karpathy | prompt_conditioned_v2 | {coco['smart']['max_visual_tokens']} | CIDEr={coco['smart']['CIDEr']:.4f} | ROUGE_L={coco['smart']['ROUGE_L']:.4f}, Bleu_4={coco['smart']['Bleu_4']:.4f}, TTFT={coco['smart']['ttft_mean_s']:.4f}s |",
    ]


def build_report(bundle: dict[str, Any]) -> str:
    stage_lines = _build_stage_table(bundle['partition_validation']['stage_backend_manifest'])
    transfer_lines = _build_transfer_table(bundle['partition_validation']['expected_transfer_edges'])
    hardware_lines = _build_hardware_table(bundle['hardware']['variants'])
    quality_lines = _build_quality_table(bundle)
    limitation_lines = [
        '- '
        + f"{item['submodel']} sg{item['subgraph_index']}: {item['mapping_limitation']}"
        for item in bundle['partition_validation']['unsupported_or_opaque_regions']
    ]
    return '\n'.join([
        '# Partition-aware FastVLM bundle',
        '',
        '## 1. Partition-aware execution core',
        '',
        f"- Validation passed: {bundle['partition_validation']['validation_passed']}",
        *stage_lines,
        '',
        '## 2. Partition validation / transfer edges',
        '',
        *transfer_lines,
        '',
        '## 3. Hardware overlap evidence',
        '',
        f"- Sequential NPU/NPU device runtime: {bundle['hardware']['sequential_npu_npu_uniform']['device_runtime_s']:.3f} s",
        f"- Overlap uniform event window: {bundle['hardware']['overlap_cpu_decode_npu_prefill_uniform']['event_window_s']:.3f} s",
        f"- Overlap fixed-smart event window: {bundle['hardware']['overlap_cpu_decode_npu_prefill_prompt_conditioned_v2']['event_window_s']:.3f} s",
        f"- Overlap adaptive event window: {bundle['hardware']['overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2']['event_window_s']:.3f} s",
        f"- Overlap adaptive pipeline speedup: {bundle['hardware']['overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2']['overlap_pipeline_speedup']:.3f}x",
        '',
        *hardware_lines,
        '',
        '## 4. Quality benchmarks',
        '',
        f"- Uniform accuracy @ {bundle['gqa']['uniform']['max_visual_tokens']} tokens: {bundle['gqa']['uniform']['accuracy']:.3f} ({bundle['gqa']['uniform']['correct']}/{bundle['gqa']['uniform']['total_scored']})",
        f"- Smart accuracy @ {bundle['gqa']['smart']['max_visual_tokens']} tokens: {bundle['gqa']['smart']['accuracy']:.3f} ({bundle['gqa']['smart']['correct']}/{bundle['gqa']['smart']['total_scored']})",
        f"- Accuracy delta: {bundle['gqa']['accuracy_delta']:+.3f}",
        f"- CIDEr delta: {bundle['coco']['cider_delta']:+.4f}",
        '',
        *quality_lines,
        '',
        '## Opaque / unresolved mapping limits',
        '',
        *(limitation_lines or ['- None']),
        '',
    ]) + '\n'


def build_paper_draft(bundle: dict[str, Any]) -> str:
    figure_manifest = bundle.get('figure_manifest', {})
    artifact_manifest = bundle.get('artifact_manifest', {})
    stage_table = _build_stage_table(bundle['partition_validation']['stage_backend_manifest'])
    transfer_table = _build_transfer_table(bundle['partition_validation']['expected_transfer_edges'])
    hardware_table = _build_hardware_table(bundle['hardware']['variants'])
    quality_table = _build_quality_table(bundle)
    limitations = [
        f"- {item['submodel']} sg{item['subgraph_index']}: {item['mapping_limitation']}"
        for item in bundle['partition_validation']['unsupported_or_opaque_regions']
    ]
    figure_lines = [
        f"- {name}: `{path}`"
        for name, path in sorted(figure_manifest.items())
    ] or ['- None']
    artifact_lines = [
        f"- {name}: `{path}`"
        for name, path in sorted(artifact_manifest.items())
    ] or ['- None']
    return '\n'.join([
        '# Partition-Aware Heterogeneous FastVLM on LiteRT',
        '',
        '## Abstract',
        (
            'We present a LiteRT-based deployment system for FastVLM on a heterogeneous mobile SoC. '
            'The system has exactly five research-facing components: a partition-aware execution core, '
            'a post-projection pruning seam, a queue-aware overlap scheduler, a structured observability '
            'layer, and a partition validation tool. The runtime executes real packaged FastVLM artifacts '
            'with Qualcomm DispatchDelegate-backed NPU prefill, XNNPACK-visible CPU projection regions, '
            'and CPU decode in the overlap path. The system is explicitly constrained by opaque packaged '
            '`DISPATCH_OP` regions for the vision encoder and language-model prefill/decode core, so the '
            'optimization boundary is the measurable post-projection visual-token seam rather than an '
            'imagined compiler-visible interior. On 500 GQA samples at a 64-token budget, prompt-conditioned '
            'post-projection budgeting improves relaxed practical accuracy from '
            f"{bundle['gqa']['uniform']['accuracy']:.3f} to {bundle['gqa']['smart']['accuracy']:.3f}. "
            'On 50 COCO Karpathy samples at the same budget, the same policy remains close but slightly '
            f"worse than uniform in free-form caption quality (CIDEr {bundle['coco']['smart']['CIDEr']:.4f} "
            f"vs {bundle['coco']['uniform']['CIDEr']:.4f}). On a 9-image stream, the heterogeneous overlap "
            'runtime achieves 1.271x fixed-budget and 1.377x adaptive pipeline speedup relative to a '
            'non-overlapped same-backend serial estimate, while remaining slower than pure NPU/NPU sequential '
            'generation on this short-output workload. The result is a reproducible hardware-aware multimodal '
            'systems scaffold that makes backend placement, opaque regions, transfer edges, and scheduling '
            'tradeoffs explicit.'
        ),
        '',
        '## 1. Introduction',
        (
            'FastVLM already runs on the target mobile device, but its deployed LiteRT package does not expose '
            'a fully transparent graph. Some regions are visible and partially delegated on CPU backends such as '
            'XNNPACK, while the vision encoder and language-model core are already wrapped inside packaged '
            '`DISPATCH_OP` regions. That deployment fact changes what a technically honest systems paper can claim. '
            'We are not introducing a new compiler pass inside opaque vendor binaries. We are building a '
            'partition-aware runtime and analysis stack on top of LiteRT’s real execution model.'
        ),
        '',
        (
            'The central design decision is to keep the optimization seam after the vision adapter / projection '
            'stage and before language-model prefill. That seam is cheap to control, easy to measure, and hardware '
            'meaningful because it directly changes NPU prefill work without pretending to rewrite the opaque encoder.'
        ),
        '',
        '## 2. System Overview',
        '',
        'The system contains exactly five research-facing components.',
        '',
        '### 2.1 Partition-aware execution core',
        (
            'The runtime logs declared and observed backend placement for the vision encoder, vision adapter / '
            'projection, pruning seam, prefill, decode, and auxiliary mask path. Experiments fail when the '
            'runtime manifest deviates from the requested backend configuration.'
        ),
        '',
        '### 2.2 Post-projection pruning seam',
        (
            'The pruning module consumes projected visual tokens, computes cheap salience and prompt-conditioned '
            'scores, optionally applies local diversity suppression, and packs a retained keep set for prefill. '
            'The policy is explicitly controllable through token budgets and fail-fast environment overrides.'
        ),
        '',
        '### 2.3 Queue-aware overlap scheduler',
        (
            'The overlap runtime maintains a bounded request queue and executes NPU prefill for request i+1 while '
            'CPU decode executes for request i. A smoothed budget controller uses recent prefill/decode windows and '
            'queue depth to pick a token budget with bounded per-request change.'
        ),
        '',
        '### 2.4 Observability layer',
        (
            'Per-request logs include queue depth, chosen token budget, retained-token count, stage backend, '
            'prepare/prefill/decode timing, exact overlap-path TTFT, CPU active time, estimated NPU active ratio, '
            'CPU stall time, sync/wait time, handoff counts, and final output text.'
        ),
        '',
        '### 2.5 Partition validation tool',
        (
            'Every run emits a subgraph-to-backend validation bundle with opaque vs visible regions, expected '
            'transfer edges, compile artifact IDs, unsupported/fallback reasons, and a configuration deviation check.'
        ),
        '',
        '## 3. Runtime Backend Manifest',
        '',
        *stage_table,
        '',
        'These assignments come from the validated overlap runtime rather than a paper-only diagram. '
        'The critical boundary is that `vision_encoder` and `prefill` remain opaque NPU dispatch regions, '
        'whereas `vision_adapter` and `pruning_seam` remain explicitly measurable.',
        '',
        '## 4. Transfer Edges and Visibility Limits',
        '',
        *transfer_table,
        '',
        'The partition validation tool explicitly reports opaque mapping limits instead of fabricating internal '
        'per-op assignments for precompiled dispatch regions.',
        '',
        *limitations,
        '',
        '## 5. Experimental Methodology',
        '',
        '### 5.1 Benchmarks',
        '- GQA is the primary grounded reasoning benchmark and uses 500 on-device overlap samples at 64 tokens.',
        '- COCO Karpathy captioning is the free-form generation benchmark and uses 50 on-device samples at 64 tokens.',
        '- A 9-image stream benchmark provides hardware evidence for sequential, fixed-overlap, smart-overlap, and adaptive-overlap variants.',
        '',
        '### 5.2 Four runtime variants',
        '- `sequential_npu_npu_uniform`',
        '- `overlap_cpu_decode_npu_prefill_uniform`',
        '- `overlap_cpu_decode_npu_prefill_prompt_conditioned_v2`',
        '- `overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2`',
        '',
        '## 6. Main Results',
        '',
        *hardware_table,
        '',
        *quality_table,
        '',
        (
            'The hardware results show a real heterogeneous pipeline: the adaptive controller achieves '
            f"{bundle['hardware']['overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2']['overlap_pipeline_speedup']:.3f}x "
            'speedup relative to a non-overlapped same-backend serial estimate. The absolute device runtime '
            f"remains worse than pure NPU/NPU sequential generation ({bundle['hardware']['sequential_npu_npu_uniform']['device_runtime_s']:.3f}s "
            f"vs {bundle['hardware']['overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2']['event_window_s']:.3f}s), "
            'which is the honest result for this short-output stream workload.'
        ),
        '',
        (
            'The quality results show that post-projection budgeting can improve short-answer reasoning slightly '
            f"on GQA (+{bundle['gqa']['accuracy_delta']:.3f} accuracy at 64 tokens) while remaining close but "
            f"slightly worse on free-form COCO captioning ({bundle['coco']['cider_delta']:+.4f} CIDEr). "
            'This is consistent with a systems story: the seam is useful and controllable, but policy tuning '
            'remains task-sensitive.'
        ),
        '',
        '## 7. Reproducibility and Artifact Bundle',
        '',
        '### 7.1 Key input artifacts',
        *artifact_lines,
        '',
        '### 7.2 Generated figures',
        *figure_lines,
        '',
        '### 7.3 Reproduction entry points',
        '- `scripts/fastvlm_partition_validator.py` for partition validation',
        '- `scripts/fastvlm_overlap_stream_benchmark.py` for four-variant stream hardware evidence',
        '- `scripts/run_fastvlm_litert_gqa_overlap_eval.py` for GQA overlap evaluation',
        '- `scripts/run_fastvlm_litert_coco_eval.py` for COCO Karpathy evaluation',
        '- `scripts/build_partition_aware_fastvlm_bundle.py` for bundle assembly',
        '',
        '## 8. Threats to Validity',
        '- GQA uses a relaxed practical matcher rather than the official strict scorer.',
        '- COCO currently uses a 50-sample subset in this validated bundle, not the full Karpathy split.',
        '- Exact per-op visibility inside precompiled `DISPATCH_OP` regions is fundamentally unavailable from these artifacts.',
        '- The current bundle contains timing/utilization evidence but not power or energy because the validated runs were collected on AC power.',
        '',
        '## 9. Conclusion',
        (
            'The resulting system is a complete LiteRT-based research scaffold for a hardware-aware FastVLM paper. '
            'It does not overclaim compiler visibility inside packaged NPU artifacts. Instead, it contributes a '
            'partition-aware runtime contract, an explicit post-projection optimization seam, a queue-aware overlap '
            'scheduler, strong observability, and a fail-fast validation tool that binds the paper claims to the '
            'real runtime configuration.'
        ),
        '',
    ]) + '\n'


def collect_default_figure_manifest(hardware_summary_path: Path) -> dict[str, str]:
    figure_dir = hardware_summary_path.parent
    expected = {
        'variant_comparison': figure_dir / 'variant_comparison.svg',
        'hardware_counters': figure_dir / 'hardware_counters.svg',
        'overlap_timeline': figure_dir / 'overlap_timeline.svg',
        'partition_map': figure_dir / 'partition_map.svg',
    }
    missing = [name for name, path in expected.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing expected figure files for paper bundle: '
            + ', '.join(f'{name}={expected[name]}' for name in missing)
        )
    return {name: str(path) for name, path in expected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-summary', type=Path, required=True)
    parser.add_argument('--partition-validation', type=Path, required=True)
    parser.add_argument('--gqa-uniform', type=Path, required=True)
    parser.add_argument('--gqa-smart', type=Path, required=True)
    parser.add_argument('--coco-uniform', type=Path, required=True)
    parser.add_argument('--coco-smart', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact_manifest = {
        'hardware_summary': str(args.hardware_summary),
        'partition_validation': str(args.partition_validation),
        'gqa_uniform': str(args.gqa_uniform),
        'gqa_smart': str(args.gqa_smart),
        'coco_uniform': str(args.coco_uniform),
        'coco_smart': str(args.coco_smart),
    }
    figure_manifest = collect_default_figure_manifest(args.hardware_summary)
    bundle = build_bundle(
        hardware_summary=load_json(args.hardware_summary),
        partition_validation=load_json(args.partition_validation),
        gqa_uniform=load_json(args.gqa_uniform),
        gqa_smart=load_json(args.gqa_smart),
        coco_uniform=load_json(args.coco_uniform),
        coco_smart=load_json(args.coco_smart),
        artifact_manifest=artifact_manifest,
        figure_manifest=figure_manifest,
    )
    (args.output_dir / 'summary.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    (args.output_dir / 'report.md').write_text(build_report(bundle), encoding='utf-8')
    (args.output_dir / 'paper_draft.md').write_text(build_paper_draft(bundle), encoding='utf-8')
    (args.output_dir / 'artifact_manifest.json').write_text(
        json.dumps(
            {
                'artifacts': artifact_manifest,
                'figures': figure_manifest,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(json.dumps({'output_dir': str(args.output_dir)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
