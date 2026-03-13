from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    module_path = REPO_ROOT / 'scripts' / 'consolidate_artifacts.py'
    spec = importlib.util.spec_from_file_location('consolidate_artifacts', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArtifactRetentionPolicyTest(unittest.TestCase):
    def test_scripts_no_longer_point_to_unarchived_legacy_artifact_roots(self) -> None:
        legacy_roots = (
            "artifacts/analysis",
            "artifacts/aux_experiments",
            "artifacts/cases_benchmark",
            "artifacts/coco_eval",
            "artifacts/gqa_eval",
            "artifacts/graph_inspect",
            "artifacts/graph_inspect_auxmaskrope_runtime",
            "artifacts/host_edge_diff",
            "artifacts/local_qairt_repo",
            "artifacts/logs",
            "artifacts/partition_plans",
            "artifacts/prefill_experiments",
            "artifacts/regeneration",
            "artifacts/tmp",
        )
        offenders: list[str] = []
        for path in (REPO_ROOT / 'scripts').glob('*'):
            if not path.is_file() or path.suffix == '.pyc':
                continue
            content = path.read_text(encoding='utf-8', errors='replace')
            for root in legacy_roots:
                if root in content:
                    offenders.append(f'{path.relative_to(REPO_ROOT)}:{root}')
        self.assertEqual([], offenders, f'Legacy scripts still reference unconsolidated artifact roots: {offenders}')

    def test_retention_docs_exist(self) -> None:
        required = [
            REPO_ROOT / 'docs' / 'ARTIFACT_RETENTION.md',
            REPO_ROOT / 'artifacts' / 'README.md',
        ]
        missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
        self.assertEqual([], missing, f'Missing retention-policy docs: {missing}')

    def test_consolidation_moves_stale_dirs_and_writes_manifest(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            artifacts = repo_root / 'artifacts'
            (artifacts / 'graphpilot_edge').mkdir(parents=True)
            (artifacts / 'models').mkdir()
            for name in ('analysis', 'cases_benchmark', 'logs'):
                path = artifacts / name
                path.mkdir()
                (path / 'marker.txt').write_text(name, encoding='utf-8')

            summary = module.consolidate_artifacts(repo_root=repo_root, dry_run=False)

            archive_root = artifacts / 'legacy_fastvlm'
            self.assertTrue(archive_root.is_dir())
            for name in ('analysis', 'cases_benchmark', 'logs'):
                self.assertFalse((artifacts / name).exists())
                self.assertTrue((archive_root / name / 'marker.txt').is_file())
            self.assertTrue((artifacts / 'graphpilot_edge').is_dir())
            self.assertTrue((artifacts / 'models').is_dir())

            manifest = pathlib.Path(summary['manifest_path'])
            self.assertTrue(manifest.is_file())
            manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(sorted(manifest_data['moved'].keys()), ['analysis', 'cases_benchmark', 'logs'])

    def test_consolidation_fails_fast_when_archive_destination_exists(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = pathlib.Path(tmpdir)
            artifacts = repo_root / 'artifacts'
            (artifacts / 'analysis').mkdir(parents=True)
            archive_target = artifacts / 'legacy_fastvlm' / 'analysis'
            archive_target.mkdir(parents=True)

            with self.assertRaises(RuntimeError):
                module.consolidate_artifacts(repo_root=repo_root, dry_run=False)


if __name__ == '__main__':
    unittest.main()
