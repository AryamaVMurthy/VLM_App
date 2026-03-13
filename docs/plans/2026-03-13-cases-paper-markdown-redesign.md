# GraphPilot CASES Paper Markdown-First Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the GraphPilot CASES paper from a markdown-first source, improve the figure/table story, and generate a clean 12-page IEEE PDF with manual layout verification.

**Architecture:** Introduce a markdown paper source and a constrained markdown-to-LaTeX renderer, then rebuild the CASES figure builder around stronger checkpoint-pinned visuals. The final PDF remains IEEEtran-based but derives its prose and section structure from markdown, not embedded Python strings.

**Tech Stack:** Python 3, unittest, IEEEtran LaTeX, Poppler (`pdftoppm`, `pdfinfo`), existing GraphPilot artifact summaries.

---

### Task 1: Add failing tests for markdown-first paper build outputs

**Files:**
- Modify: `scripts/tests/test_build_graphpilot_cases_paper.py`
- Create: `scripts/tests/test_build_graphpilot_cases_markdown.py`
- Test: `scripts/tests/test_build_graphpilot_cases_paper.py`
- Test: `scripts/tests/test_build_graphpilot_cases_markdown.py`

**Step 1: Write the failing tests**
- Add tests that require:
  - markdown source discovery under `papers/graphpilot_cases_markdown/`
  - a rendered markdown artifact path in the paper build summary
  - generated LaTeX section files sourced from markdown, not hardcoded prose only
  - fail-fast behavior on unsupported markdown constructs

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_cases_paper scripts.tests.test_build_graphpilot_cases_markdown -v
```
Expected: FAIL because markdown-first support does not exist yet.

**Step 3: Implement minimal code to satisfy discovery contracts**
- Add the smallest loader/parser stubs needed to make tests move from missing-feature failure toward passing behavior.

**Step 4: Run tests to verify they pass**
Run the same command and confirm PASS.

### Task 2: Create markdown paper source and controlled renderer

**Files:**
- Create: `papers/graphpilot_cases_markdown/paper.md`
- Create: `papers/graphpilot_cases_markdown/README.md`
- Create: `scripts/graphpilot_cases_markdown.py`
- Modify: `scripts/build_graphpilot_cases_paper.py`
- Test: `scripts/tests/test_build_graphpilot_cases_markdown.py`

**Step 1: Write the failing test**
- Add tests for heading/paragraph/list rendering and raw-LaTeX figure include passthrough.

**Step 2: Run the test to verify it fails**
Run the targeted unittest command and confirm a controlled failure.

**Step 3: Write minimal implementation**
- Implement a strict markdown subset parser/renderer.
- Fail fast on unsupported constructs.
- Make the paper builder render markdown into LaTeX section files before compiling.

**Step 4: Run tests to verify they pass**
Run the targeted tests and confirm PASS.

### Task 3: Redesign figure and table generation around stronger visuals

**Files:**
- Modify: `scripts/build_graphpilot_cases_figures.py`
- Modify: `scripts/tests/test_build_graphpilot_cases_figures.py`
- Inspect: `artifacts/graphpilot_edge/analysis/graphpilot_characterization_20260312_193218/summary.json`
- Inspect: `artifacts/graphpilot_edge/analysis/graphpilot_cost_calibration_20260312_193216/summary.json`
- Inspect: `artifacts/graphpilot_edge/experiments/graphpilot_experiment_batch_20260312_192509/summary.json`

**Step 1: Write the failing tests**
- Require the new composite figures and any renamed outputs used by the new markdown paper.

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_cases_figures -v
```
Expected: FAIL because the new outputs do not exist yet.

**Step 3: Implement minimal figure changes**
- Replace weak zero-information plots with curated composites:
  - workflow universe coverage
  - real-device primary workflow results
  - family residuals + sim-to-real deltas
  - selected baseline score matrix
  - informative ablation/sensitivity overview

**Step 4: Run tests to verify they pass**
Run the targeted figure tests and confirm PASS.

### Task 4: Regenerate the canonical paper evidence bundle

**Files:**
- Modify: `scripts/build_graphpilot_cases_paper.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Generate under: `artifacts/graphpilot_edge/papers/`
- Generate under: `artifacts/graphpilot_edge/reports/`

**Step 1: Build figures from the canonical checkpoint**
Run the figure builder against the latest canonical checkpoint manifest.

**Step 2: Build the markdown-first CASES paper**
Run the paper builder and confirm it emits markdown bundle, LaTeX tree, summary, and PDF.

**Step 3: Regenerate the artifact pack**
Run the artifact-pack builder so the new paper becomes the canonical report surface.

**Step 4: Verify outputs exist**
Check that the expected summary JSON, PDF, markdown bundle, and figure bundle all exist.

### Task 5: Render and manually review every page

**Files:**
- Generate under: `tmp/pdfs/`
- Inspect: final paper PDF pages

**Step 1: Render PDF pages to PNGs**
Run:
```bash
mkdir -p tmp/pdfs/final_cases_review
pdftoppm -png artifacts/graphpilot_edge/papers/<latest>/main.pdf tmp/pdfs/final_cases_review/page
```

**Step 2: Manually inspect each page**
Check for:
- overlaps
- clipped text
- large random whitespace blocks
- bad table alignment
- unreadable plot labels

**Step 3: Fix the smallest root cause and rebuild**
Iterate on LaTeX layout or figure sizing only after identifying the exact cause.

**Step 4: Re-render and re-check**
Repeat until the PDF is clean.

### Task 6: Run full verification and capture final paper outputs

**Files:**
- Modify if needed: `scripts/tests/*`
- Update generated outputs under `artifacts/graphpilot_edge/`

**Step 1: Run full Python and script verification**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```

**Step 2: Re-run Android regression only if paper changes touched runtime evidence wiring**
Run the existing GraphPilot Android tests if artifact pointers or runtime-facing builders changed.

**Step 3: Verify PDF page count and summary**
Run `pdfinfo` and confirm the paper build summary records the expected outputs.

**Step 4: Commit**
```bash
git add docs/plans papers/graphpilot_cases_markdown scripts artifacts/graphpilot_edge/reports artifacts/graphpilot_edge/papers
 git commit -m "feat: rebuild CASES paper with markdown-first pipeline"
```
