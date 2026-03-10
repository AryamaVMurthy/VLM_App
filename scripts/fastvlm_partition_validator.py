#!/usr/bin/env python3
"""Validate FastVLM heterogeneous partitioning against runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_host_edge_diff import parse_event_stream

DEFAULT_PARTITION_JSON = (
    SCRIPT_DIR.parent
    / "artifacts/analysis/cases_paper_bundle_20260310_030627/heterogeneous_partition_mapping.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts" / "analysis"
DEFAULT_REQUIRED_STAGES = (
    "vision_encoder",
    "vision_adapter",
    "pruning_seam",
    "prefill",
    "decode",
    "aux_mask",
)


def sha256_prefix(path: Path, prefix_chars: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:prefix_chars]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_index(stage_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for event in stage_events:
        stage = str(event.get("stage", "")).strip()
        if not stage:
            raise RuntimeError(f"Encountered STAGE_BACKEND event without stage: {event}")
        indexed[stage] = event
    return indexed


def derive_transfer_edges(stage_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def observed(stage: str) -> str:
        return str(stage_index[stage]["observed_backend"])

    return [
        {
            "from_stage": "vision_encoder",
            "to_stage": "vision_adapter",
            "cross_backend": observed("vision_encoder") != observed("vision_adapter"),
            "edge_type": "implicit_runtime_tensor_handoff",
            "observable_copy_counter": False,
        },
        {
            "from_stage": "vision_adapter",
            "to_stage": "pruning_seam",
            "cross_backend": observed("vision_adapter") != observed("pruning_seam"),
            "edge_type": "host_tensor_pass",
            "observable_copy_counter": False,
        },
        {
            "from_stage": "pruning_seam",
            "to_stage": "prefill",
            "cross_backend": observed("pruning_seam") != observed("prefill"),
            "edge_type": "packed_visual_token_transfer",
            "observable_copy_counter": False,
        },
        {
            "from_stage": "prefill",
            "to_stage": "decode",
            "cross_backend": observed("prefill") != observed("decode"),
            "edge_type": "prefill_decode_handoff",
            "observable_copy_counter": True,
        },
    ]


def summarize_limitations(partition_map: dict[str, Any]) -> list[dict[str, Any]]:
    limitations: list[dict[str, Any]] = []
    for submodel, entry in partition_map.items():
        for subgraph in entry.get("subgraphs", []):
            limitation = subgraph.get("mapping_limitation")
            if not limitation:
                continue
            limitations.append(
                {
                    "submodel": submodel,
                    "subgraph_index": subgraph.get("index"),
                    "runtime_backend": subgraph.get("runtime_backend"),
                    "mapping_limitation": limitation,
                    "per_op_mapping_possible": bool(
                        subgraph.get("per_op_mapping_possible", False)
                    ),
                }
            )
    return limitations


def build_validation_bundle(
    *,
    run_log: Path,
    partition_json: Path,
    model_path: Path,
    decode_model_path: Path,
    expected_prefill_backend: str,
    expected_decode_backend: str,
) -> dict[str, Any]:
    parsed = parse_event_stream(run_log)
    events = parsed.events
    overlap_config = next(
        (event for event in events if event.get("type") == "OVERLAP_CONFIG"), None
    )
    if overlap_config is None:
        raise RuntimeError(f"Missing OVERLAP_CONFIG event in {run_log}")
    stage_events = [event for event in events if event.get("type") == "STAGE_BACKEND"]
    if not stage_events:
        raise RuntimeError(
            f"Missing STAGE_BACKEND events in {run_log}; rebuild and rerun the overlap runtime."
        )
    stage_index = _stage_index(stage_events)
    missing = [stage for stage in DEFAULT_REQUIRED_STAGES if stage not in stage_index]
    if missing:
        raise RuntimeError(
            f"Missing required STAGE_BACKEND events in {run_log}: {missing}"
        )

    partition_map = load_json(partition_json)
    deviations: list[str] = []
    if str(overlap_config.get("prefill_backend")) != expected_prefill_backend:
        deviations.append(
            f"prefill_backend runtime={overlap_config.get('prefill_backend')} expected={expected_prefill_backend}"
        )
    if str(overlap_config.get("decode_backend")) != expected_decode_backend:
        deviations.append(
            f"decode_backend runtime={overlap_config.get('decode_backend')} expected={expected_decode_backend}"
        )

    prefill_stage = stage_index["prefill"]
    decode_stage = stage_index["decode"]
    if str(prefill_stage.get("declared_backend")) != expected_prefill_backend:
        deviations.append(
            f"prefill stage declared_backend={prefill_stage.get('declared_backend')} expected={expected_prefill_backend}"
        )
    if str(decode_stage.get("declared_backend")) != expected_decode_backend:
        deviations.append(
            f"decode stage declared_backend={decode_stage.get('declared_backend')} expected={expected_decode_backend}"
        )

    if Path(str(prefill_stage.get("compile_artifact", ""))).name not in {
        model_path.name,
        "",
    }:
        deviations.append(
            "prefill compile_artifact does not match provided model path basename"
        )
    if Path(str(decode_stage.get("compile_artifact", ""))).name not in {
        decode_model_path.name,
        "",
    }:
        deviations.append(
            "decode compile_artifact does not match provided decode model path basename"
        )

    expected_partition_pairs = {
        "vision_encoder": "NPU_DISPATCHDELEGATE",
        "vision_adapter": "CPU_XNNPACK",
    }
    for stage_name, expected_backend in expected_partition_pairs.items():
        observed_backend = str(stage_index[stage_name].get("observed_backend"))
        if observed_backend != expected_backend:
            deviations.append(
                f"{stage_name} observed_backend={observed_backend} expected={expected_backend}"
            )

    handoff_export_count = len(
        [event for event in events if event.get("type") == "HANDOFF_EXPORT"]
    )
    handoff_import_count = len(
        [event for event in events if event.get("type") == "HANDOFF_IMPORT"]
    )

    return {
        "validation_passed": not deviations,
        "config_deviation": deviations,
        "run_log": str(run_log),
        "recovered_split_events": parsed.recovered_split_events,
        "declared_experiment_config": {
            "prefill_backend": expected_prefill_backend,
            "decode_backend": expected_decode_backend,
        },
        "observed_runtime_config": overlap_config,
        "compile_artifacts": {
            "prefill_model": {
                "path": str(model_path),
                "basename": model_path.name,
                "sha256_prefix": sha256_prefix(model_path),
            },
            "decode_model": {
                "path": str(decode_model_path),
                "basename": decode_model_path.name,
                "sha256_prefix": sha256_prefix(decode_model_path),
            },
        },
        "stage_backend_manifest": [stage_index[stage] for stage in DEFAULT_REQUIRED_STAGES],
        "subgraph_backend_assignment": partition_map,
        "expected_transfer_edges": derive_transfer_edges(stage_index),
        "handoff_counters": {
            "export_event_count": handoff_export_count,
            "import_event_count": handoff_import_count,
        },
        "unsupported_or_opaque_regions": summarize_limitations(partition_map),
    }


def build_report(bundle: dict[str, Any]) -> str:
    lines = [
        "# FastVLM Partition Validation",
        "",
        f"- Validation passed: {bundle['validation_passed']}",
        f"- Run log: {bundle['run_log']}",
        f"- Recovered split events: {bundle['recovered_split_events']}",
        "",
        "## Compile Artifacts",
        "",
        f"- Prefill model: `{bundle['compile_artifacts']['prefill_model']['basename']}` sha256={bundle['compile_artifacts']['prefill_model']['sha256_prefix']}",
        f"- Decode model: `{bundle['compile_artifacts']['decode_model']['basename']}` sha256={bundle['compile_artifacts']['decode_model']['sha256_prefix']}",
        "",
        "## Stage Backend Manifest",
        "",
    ]
    for item in bundle["stage_backend_manifest"]:
        lines.append(
            "- "
            + f"{item['stage']}: declared={item['declared_backend']} observed={item['observed_backend']} "
            + f"visibility={item['visibility']} unit={item['stage_unit']} source={item['source']}"
        )
    lines.extend(
        [
            "",
            "## Expected Transfer Edges",
            "",
        ]
    )
    for edge in bundle["expected_transfer_edges"]:
        lines.append(
            "- "
            + f"{edge['from_stage']} -> {edge['to_stage']}: "
            + f"type={edge['edge_type']} cross_backend={edge['cross_backend']} "
            + f"observable_copy_counter={edge['observable_copy_counter']}"
        )
    lines.extend(["", "## Limitations / Opaque Regions", ""])
    for item in bundle["unsupported_or_opaque_regions"]:
        lines.append(
            "- "
            + f"{item['submodel']} sg{item['subgraph_index']} backend={item['runtime_backend']}: "
            + f"{item['mapping_limitation']}"
        )
    if bundle["config_deviation"]:
        lines.extend(["", "## Config Deviations", ""])
        for item in bundle["config_deviation"]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--partition-json", type=Path, default=DEFAULT_PARTITION_JSON)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--decode-model-path", type=Path, required=True)
    parser.add_argument("--expected-prefill-backend", default="npu")
    parser.add_argument("--expected-decode-backend", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_log.is_file():
        raise FileNotFoundError(f"Missing run log: {args.run_log}")
    if not args.partition_json.is_file():
        raise FileNotFoundError(f"Missing partition json: {args.partition_json}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Missing model path: {args.model_path}")
    if not args.decode_model_path.is_file():
        raise FileNotFoundError(f"Missing decode model path: {args.decode_model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    bundle = build_validation_bundle(
        run_log=args.run_log,
        partition_json=args.partition_json,
        model_path=args.model_path,
        decode_model_path=args.decode_model_path,
        expected_prefill_backend=args.expected_prefill_backend,
        expected_decode_backend=args.expected_decode_backend,
    )
    (args.output_dir / "partition_validation.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(build_report(bundle), encoding="utf-8")
    if not bundle["validation_passed"]:
        raise SystemExit(
            "Partition validation failed; see "
            f"{args.output_dir / 'partition_validation.json'}"
        )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
