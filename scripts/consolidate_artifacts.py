#!/usr/bin/env python3
"""Consolidate stale pre-GraphPilot artifact directories under a retention policy."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from legacy_fastvlm_paths import LEGACY_FASTVLM_ARTIFACT_ROOT, REPO_ROOT


RETENTION_POLICY_VERSION = "2026-03-13"
STALE_TOP_LEVEL_DIRS = (
    "analysis",
    "aux_experiments",
    "cases_benchmark",
    "coco_eval",
    "gqa_eval",
    "graph_inspect",
    "graph_inspect_auxmaskrope_runtime",
    "host_edge_diff",
    "local_qairt_repo",
    "logs",
    "partition_plans",
    "prefill_experiments",
    "regeneration",
    "tmp",
)
KEEP_TOP_LEVEL_DIRS = ("graphpilot_edge", "models")


def consolidate_artifacts(repo_root: Path, dry_run: bool) -> dict[str, object]:
    artifacts_root = repo_root / "artifacts"
    if not artifacts_root.is_dir():
        raise RuntimeError(f"Missing artifacts root: {artifacts_root}")

    archive_root = artifacts_root / LEGACY_FASTVLM_ARTIFACT_ROOT.name
    moved: dict[str, str] = {}
    skipped: list[str] = []

    if archive_root.exists() and not archive_root.is_dir():
        raise RuntimeError(f"Archive root exists but is not a directory: {archive_root}")

    for keep_name in KEEP_TOP_LEVEL_DIRS:
        keep_path = artifacts_root / keep_name
        if keep_path.exists() and archive_root == keep_path:
            raise RuntimeError(f"Archive root collides with kept artifact directory: {keep_path}")

    if not dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)

    for name in STALE_TOP_LEVEL_DIRS:
        source = artifacts_root / name
        if not source.exists():
            skipped.append(name)
            continue
        target = archive_root / name
        if target.exists():
            raise RuntimeError(
                f"Archive destination already exists for {name}: {target}. "
                "Resolve the collision explicitly before consolidating again."
            )
        moved[name] = str(target.relative_to(repo_root))
        if not dry_run:
            shutil.move(str(source), str(target))

    manifest = {
        "retention_policy_version": RETENTION_POLICY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "archive_root": str(archive_root.relative_to(repo_root)),
        "kept": list(KEEP_TOP_LEVEL_DIRS),
        "moved": moved,
        "skipped_missing": skipped,
        "dry_run": dry_run,
    }

    manifest_path = archive_root / "consolidation_manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["manifest_path"] = str(manifest_path if not dry_run else archive_root / "consolidation_manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = consolidate_artifacts(repo_root=args.repo_root.resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
