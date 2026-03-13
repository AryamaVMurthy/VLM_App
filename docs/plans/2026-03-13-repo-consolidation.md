# Repository Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate the repository into a clear GraphPilot-Edge-first structure without breaking canonical artifact paths.

**Architecture:** Keep the existing physical directory layout stable, and consolidate through a new root README, repo-map docs, folder-level READMEs, and local-junk ignore rules. Add a small regression test that fails if the consolidated documentation surface goes stale.

**Tech Stack:** Markdown, Python unittest, gitignore.

---

### Task 1: Add a failing documentation regression test

**Files:**
- Create: `scripts/tests/test_repo_consolidation_docs.py`

**Step 1: Write the failing test**
- Assert that the root `README.md` references `docs/INDEX.md`, `docs/REPO_LAYOUT.md`, `artifacts/graphpilot_edge/README.md`, and `artifacts/graphpilot_edge/papers/graphpilot_cases_20260312_195225/main.pdf`.
- Assert that the following files exist:
  - `docs/INDEX.md`
  - `docs/REPO_LAYOUT.md`
  - `artifacts/graphpilot_edge/README.md`
  - `scripts/README.md`
  - `graphpilot_edge/README.md`
  - `android-app/README.md`
  - `Truth-docs/README.md`

**Step 2: Run the test to verify it fails**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_repo_consolidation_docs -v`
Expected: FAIL because the new files and README references do not exist yet.

### Task 2: Rewrite the root repo surface

**Files:**
- Modify: `README.md`
- Create: `docs/INDEX.md`
- Create: `docs/REPO_LAYOUT.md`

**Step 1: Replace stale CPU-only README language with GraphPilot-Edge-first content**
- Explain the repo purpose, current verified scope, canonical paper/checkpoint outputs, and major entry points.

**Step 2: Add docs index and repo layout docs**
- `docs/INDEX.md`: quick navigation for developers/reviewers.
- `docs/REPO_LAYOUT.md`: describe each top-level directory, canonical paths, and what not to move.

**Step 3: Run the documentation regression test**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_repo_consolidation_docs -v`
Expected: still FAIL until folder-level READMEs are added.

### Task 3: Add folder-level instructions for the main working surfaces

**Files:**
- Create: `artifacts/graphpilot_edge/README.md`
- Create: `scripts/README.md`
- Create: `graphpilot_edge/README.md`
- Create: `android-app/README.md`
- Create: `Truth-docs/README.md`

**Step 1: Document the role of each folder**
- `artifacts/graphpilot_edge`: canonical checkpoint, reports, experiments, papers, registries.
- `scripts`: builders, experiments, calibration, paper generation.
- `graphpilot_edge`: offline planner/simulator/runtime-support Python package.
- `android-app`: deployed runtime and instrumentation surface.
- `Truth-docs`: claim boundary and revision source.

**Step 2: Re-run the documentation regression test**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_repo_consolidation_docs -v`
Expected: PASS.

### Task 4: Clean local-noise defaults

**Files:**
- Modify: `.gitignore`

**Step 1: Add local-only ignore entries**
- Add `.venv/`
- Add `tmp/`
- Keep existing artifact exceptions intact.

**Step 2: Run regression tests**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_repo_consolidation_docs -v`
Expected: PASS.

### Task 5: Run full relevant verification and commit

**Files:**
- No new files beyond the above.

**Step 1: Run relevant script tests**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v`
Expected: PASS.

**Step 2: Manual verification**
- Read the new root `README.md`.
- Read `docs/INDEX.md` and `docs/REPO_LAYOUT.md`.
- Confirm the paper path and canonical artifact paths are still correct.

**Step 3: Commit**
- Commit only the consolidation/docs/test changes.
