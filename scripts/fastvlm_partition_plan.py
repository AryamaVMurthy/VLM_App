#!/usr/bin/env python3
"""Build an explicit FastVLM CPU/NPU partition plan from logs and graph dumps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from legacy_fastvlm_paths import LEGACY_GRAPH_INSPECT_ROOT


MODEL_TYPE_PATTERN = re.compile(r"model_type:\s+(TF_LITE_[A-Z_]+)")
DELEGATE_PATTERN = re.compile(
    r"Replacing (?P<replaced>\d+) out of (?P<total>\d+) node\(s\) with delegate "
    r"\((?P<delegate>DispatchDelegate|TfLiteXNNPackDelegate)\) node, yielding "
    r"(?P<partitions>\d+) partitions for subgraph (?P<subgraph>\d+)\."
)
PRECOMPILED_MODEL_PATTERN = re.compile(
    r"model is pre-compiled\. Plugins won't be applied\."
)
JIT_MODEL_PATTERN = re.compile(r"JIT compilation changed model, reserializing\.\.\.")
SECTION_PATTERN = re.compile(r"Section\d+_TFLiteModel_(?P<section>.+)\.full\.txt$")
SUBGRAPH_PATTERN = re.compile(r"LiteRtSubgraph\s*:\s*\[\s*#ops=(?P<ops>\d+)")
OP_PATTERN = re.compile(r"LiteRtOp\s*:\s*\[\s*(?P<name>[^\]]+)\s*\]")


def section_to_model_type(section_name: str) -> str:
    return section_name.upper()


def parse_graph_file(graph_path: Path) -> dict[str, object]:
    subgraphs: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in graph_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        subgraph_match = SUBGRAPH_PATTERN.search(line)
        if subgraph_match is not None:
            current = {
                "op_count": int(subgraph_match.group("ops")),
                "ops": [],
            }
            subgraphs.append(current)
            continue
        op_match = OP_PATTERN.search(line)
        if op_match is not None and current is not None:
            current["ops"].append(op_match.group("name"))
    return {"subgraphs": subgraphs}


def load_graphs(graph_dir: Path) -> dict[str, dict[str, object]]:
    graphs: dict[str, dict[str, object]] = {}
    for graph_path in sorted(graph_dir.glob("Section*_TFLiteModel_*.full.txt")):
        match = SECTION_PATTERN.search(graph_path.name)
        if match is None:
            continue
        model_type = section_to_model_type(match.group("section"))
        graphs[model_type] = parse_graph_file(graph_path)
    return graphs


def parse_log(log_path: Path) -> dict[str, list[dict[str, int | str]]]:
    current_model: str | None = None
    pending_models: list[str] = []
    delegates: dict[str, list[dict[str, int | str]]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        model_match = MODEL_TYPE_PATTERN.search(line)
        if model_match is not None:
            current_model = None
            pending_model = model_match.group(1)
            pending_models.append(pending_model)
            delegates.setdefault(pending_model, [])
            continue
        if PRECOMPILED_MODEL_PATTERN.search(line) is not None:
            if pending_models:
                current_model = pending_models.pop(0)
                delegates.setdefault(current_model, [])
            continue
        if JIT_MODEL_PATTERN.search(line) is not None:
            if pending_models:
                current_model = pending_models.pop()
                delegates.setdefault(current_model, [])
            continue
        delegate_match = DELEGATE_PATTERN.search(line)
        if delegate_match is None:
            continue
        if current_model is None:
            if len(pending_models) == 1:
                current_model = pending_models.pop()
                delegates.setdefault(current_model, [])
            else:
                continue
        delegates[current_model].append(
            {
                "delegate": delegate_match.group("delegate"),
                "subgraph": int(delegate_match.group("subgraph")),
                "replaced": int(delegate_match.group("replaced")),
                "total": int(delegate_match.group("total")),
                "partitions": int(delegate_match.group("partitions")),
            }
        )
    return delegates


def build_partition_plan(log_path: Path, graph_dir: Path) -> dict[str, object]:
    graphs = load_graphs(graph_dir)
    delegates = parse_log(log_path)
    models: dict[str, object] = {}
    for model_type, graph in graphs.items():
        delegate_by_subgraph = {
            entry["subgraph"]: entry for entry in delegates.get(model_type, [])
        }
        subgraphs: list[dict[str, object]] = []
        for index, subgraph in enumerate(graph["subgraphs"]):
            entry = delegate_by_subgraph.get(index)
            ops = [{"name": op_name, "backend": "UNKNOWN"} for op_name in subgraph["ops"]]
            backend = "UNKNOWN"
            mapping_limitation = None
            per_op_mapping_possible = False
            if entry is not None:
                delegate = entry["delegate"]
                is_full = entry["replaced"] == entry["total"]
                contains_dispatch = any("DISPATCH_OP" in op["name"] for op in ops)
                if delegate == "DispatchDelegate":
                    backend = "NPU" if is_full else "NPU_PARTIAL"
                else:
                    backend = "CPU" if is_full else "CPU_PARTIAL"

                if is_full and not contains_dispatch:
                    per_op_mapping_possible = True
                    for op in ops:
                        op["backend"] = backend
                elif contains_dispatch:
                    mapping_limitation = (
                        "Per-op mapping is impossible because the delegated "
                        "subgraph is a precompiled DISPATCH_OP."
                    )
                else:
                    mapping_limitation = (
                        "Per-op mapping is impossible because the runtime log "
                        "only reports a partial delegated partition."
                    )
            else:
                mapping_limitation = (
                    "No delegate replacement line was found for this subgraph; "
                    "backend stays unresolved from runtime logs alone."
                )
            subgraphs.append(
                {
                    "index": index,
                    "backend": backend,
                    "delegate_summary": entry,
                    "per_op_mapping_possible": per_op_mapping_possible,
                    "mapping_limitation": mapping_limitation,
                    "ops": ops,
                }
            )
        models[model_type] = {"subgraphs": subgraphs}
    return {"log_path": str(log_path), "graph_dir": str(graph_dir), "models": models}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Runtime log path")
    parser.add_argument(
        "--graph-dir",
        default=str(LEGACY_GRAPH_INSPECT_ROOT / "full"),
        help="Directory containing Section*_TFLiteModel_*.full.txt files",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    plan = build_partition_plan(Path(args.log).resolve(), Path(args.graph_dir).resolve())
    output = json.dumps(plan, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
