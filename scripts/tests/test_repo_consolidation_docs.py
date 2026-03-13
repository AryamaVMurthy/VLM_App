from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / 'README.md'

REQUIRED_FILES = [
    REPO_ROOT / 'docs' / 'INDEX.md',
    REPO_ROOT / 'docs' / 'REPO_LAYOUT.md',
    REPO_ROOT / 'artifacts' / 'graphpilot_edge' / 'README.md',
    REPO_ROOT / 'scripts' / 'README.md',
    REPO_ROOT / 'graphpilot_edge' / 'README.md',
    REPO_ROOT / 'android-app' / 'README.md',
    REPO_ROOT / 'Truth-docs' / 'README.md',
]

REQUIRED_README_REFERENCES = [
    'docs/INDEX.md',
    'docs/REPO_LAYOUT.md',
    'artifacts/graphpilot_edge/README.md',
    'artifacts/graphpilot_edge/papers/graphpilot_cases_20260312_195225/main.pdf',
]


class RepoConsolidationDocsTest(unittest.TestCase):
    def test_required_docs_exist(self) -> None:
        missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_FILES if not path.exists()]
        self.assertEqual([], missing, f'Missing required repo docs: {missing}')

    def test_root_readme_points_to_canonical_docs(self) -> None:
        content = README.read_text(encoding='utf-8')
        missing_refs = [ref for ref in REQUIRED_README_REFERENCES if ref not in content]
        self.assertEqual([], missing_refs, f'Root README missing canonical references: {missing_refs}')


if __name__ == '__main__':
    unittest.main()
