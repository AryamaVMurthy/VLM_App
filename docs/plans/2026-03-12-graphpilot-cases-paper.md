# GraphPilot-Edge CASES Paper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining evidence gaps from `Truth-docs/graphpilot_edge_revision_report.pdf` and produce a submission-grade IEEE ESWEEK CASES paper package from one canonical GraphPilot checkpoint.

**Architecture:** Keep the current GraphPilot runtime and artifact system as the single implementation path. Add the missing workload/baseline/evidence closure needed for the CASES claim, then generate one checkpoint-pinned paper build with figures, tables, and final audit derived from the same manifest. The paper output should be produced from code and artifacts, not manual copy-paste.

**Tech Stack:** Python 3, existing GraphPilot scripts and JSON registries, Android/QIDK validation, Beads, Matplotlib/SVG figure generation, Pandoc or LaTeX toolchain for IEEE two-column PDF generation.

---

### Task 1: Create the CASES paper Beads epic and child tasks

**Files:**
- Modify: `.beads/` issue graph via `bd` CLI
- Reference: `docs/plans/2026-03-12-graphpilot-cases-paper-design.md`
- Reference: `Truth-docs/graphpilot_edge_revision_report.pdf`

**Step 1: Create the epic**
Run:
```bash
bd create "GraphPilot CASES paper completion" -t epic -p 1 --description "Submission-grade IEEE ESWEEK CASES closure for GraphPilot-Edge from a single canonical checkpoint."
```
Expected: creates a new epic ID.

**Step 2: Create child tasks**
Create children for:
- truth-surface audit
- workload-universe closure
- method-class baseline proxies
- experiment rerun and checkpoint rebuild
- CASES figures/tables builder
- IEEE paper source and PDF build
- final verification and push

Example:
```bash
bd create "Close CASES workload universe" --parent <epic_id> -p 1
```
Expected: all required child beads exist and are linked to the epic.

**Step 3: Claim the first ready child**
Run:
```bash
bd ready
bd update <child_id> --claim
```
Expected: the first task is claimed.

**Step 4: Sync**
Run:
```bash
bd sync
```
Expected: bead state is persisted to git.

**Step 5: Commit**
```bash
git add .beads
 git commit -m "chore: add graphpilot CASES paper beads"
```

### Task 2: Add failing tests for checkpoint-pinned CASES paper generation

**Files:**
- Create: `scripts/build_graphpilot_cases_paper.py`
- Create: `scripts/tests/test_build_graphpilot_cases_paper.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Modify: `scripts/tests/test_build_graphpilot_artifact_pack.py`

**Step 1: Write the failing tests**
Create tests that assert:
- paper build fails fast if `--checkpoint-manifest` is missing
- paper build fails fast if required figures/tables are absent from the checkpoint
- paper build emits a CASES paper directory with `main.tex`, `references.bib`, `figures/`, `tables.tex`, and `paper.pdf` or explicit toolchain error
- artifact pack summary records the paper build outputs

Example test skeleton:
```python
def test_cases_paper_builder_requires_checkpoint(self):
    with self.assertRaises(SystemExit):
        module.main([])
```

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_cases_paper scripts.tests.test_build_graphpilot_artifact_pack -v
```
Expected: FAIL because the CASES paper builder does not exist yet.

**Step 3: Implement minimal CASES paper builder**
- Create `scripts/build_graphpilot_cases_paper.py`
- Input: `--checkpoint-manifest`
- Output root: `artifacts/graphpilot_edge/papers/graphpilot_cases_<timestamp>/`
- Emit:
  - `main.tex`
  - `sections/*.tex`
  - `figures/` copied/symlinked from checkpoint-pinned artifact pack
  - `tables.tex`
  - `metadata.json`
- If `latexmk`/`pdflatex` is unavailable, error out explicitly with remediation; do not silently skip PDF build

**Step 4: Thread paper outputs into artifact-pack summary**
- Extend `scripts/build_graphpilot_artifact_pack.py` so it can record the generated CASES paper directory and PDF path when provided

**Step 5: Run tests to verify they pass**
Run the same unittest command from Step 2.
Expected: PASS.

**Step 6: Commit**
```bash
git add scripts/build_graphpilot_cases_paper.py scripts/tests/test_build_graphpilot_cases_paper.py scripts/build_graphpilot_artifact_pack.py scripts/tests/test_build_graphpilot_artifact_pack.py
 git commit -m "feat: add checkpoint-pinned CASES paper builder"
```

### Task 3: Close workload-universe gaps from the revision report

**Files:**
- Modify: `configs/graphpilot_edge/workload_universe.json`
- Modify: `graphpilot_edge/workload_universe.py`
- Modify: `graphpilot_edge/model_graph_simulator.py`
- Modify: `scripts/generate_graphpilot_workload_registry.py`
- Modify: `scripts/tests/test_generate_graphpilot_workload_registry.py`
- Modify: `tests/graphpilot_edge/test_workload_universe.py`

**Step 1: Write the failing tests**
Add tests that assert the workload config includes at least:
- `mmmu_vlm_reasoning`
- `docvqa_vlm_document`
- `chartqa_vlm_chart`
- `mobile_actions_planner`
- at least one mixed-criticality continuous workload with foreground + background jobs
- at least one explicit fallback-penalty stress workload

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_workload_universe scripts.tests.test_generate_graphpilot_workload_registry -v
```
Expected: FAIL because the workload universe does not yet cover the revision-report set.

**Step 3: Implement the missing workload entries**
- Add the missing workloads to `configs/graphpilot_edge/workload_universe.json`
- Extend scenario construction in `graphpilot_edge/workload_universe.py`
- Extend `graphpilot_edge/model_graph_simulator.py` for any missing family tags/knob surfaces
- Keep them simulator-scoped unless real-device support-safe execution exists

**Step 4: Regenerate workload registry**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/generate_graphpilot_workload_registry.py
```
Expected: a new `artifacts/graphpilot_edge/registries/workload_universe_registry.json`

**Step 5: Run tests to verify they pass**
Run the unittest command from Step 2.
Expected: PASS.

**Step 6: Commit**
```bash
git add configs/graphpilot_edge/workload_universe.json graphpilot_edge/workload_universe.py graphpilot_edge/model_graph_simulator.py scripts/generate_graphpilot_workload_registry.py scripts/tests/test_generate_graphpilot_workload_registry.py tests/graphpilot_edge/test_workload_universe.py artifacts/graphpilot_edge/registries/workload_universe_registry.json
 git commit -m "feat: expand graphpilot workload universe for CASES"
```

### Task 4: Add faithful method-class proxy baselines and tests

**Files:**
- Modify: `graphpilot_edge/baselines.py`
- Modify: `scripts/generate_graphpilot_baseline_registry.py`
- Modify: `scripts/run_graphpilot_characterization.py`
- Modify: `tests/graphpilot_edge/test_baselines.py`
- Modify: `scripts/tests/test_generate_graphpilot_baseline_registry.py`
- Modify: `scripts/tests/test_run_graphpilot_characterization.py`
- Reference: `docs/plans/2026-03-12-graphpilot-paper-inspired-baseline-proxies.md`

**Step 1: Write the failing tests**
Add tests that baseline registry contains and characterization can score:
- `band_like`
- `adms_like`
- `puzzle_like`
- `twill_like`
- `heteroinfer_like`
- `agent_xpu_like`
- `hero_like`

Example assertion:
```python
self.assertIn("band_like", payload["baseline_ids"])
```

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_baselines scripts.tests.test_generate_graphpilot_baseline_registry scripts.tests.test_run_graphpilot_characterization -v
```
Expected: FAIL because those baseline IDs do not exist yet.

**Step 3: Implement the proxy baseline policies**
- Add lightweight method-class proxies to `graphpilot_edge/baselines.py`
- Document approximations in baseline metadata
- Ensure they are checkpoint-visible in `baseline_policy_registry.json`
- Ensure characterization compares GraphPilot against them honestly, including ties/losses

**Step 4: Regenerate baseline registry**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/generate_graphpilot_baseline_registry.py
```
Expected: updated baseline registry with the new IDs.

**Step 5: Run tests to verify they pass**
Run the unittest command from Step 2.
Expected: PASS.

**Step 6: Commit**
```bash
git add graphpilot_edge/baselines.py scripts/generate_graphpilot_baseline_registry.py scripts/run_graphpilot_characterization.py tests/graphpilot_edge/test_baselines.py scripts/tests/test_generate_graphpilot_baseline_registry.py scripts/tests/test_run_graphpilot_characterization.py artifacts/graphpilot_edge/registries/baseline_policy_registry.json
 git commit -m "feat: add graphpilot paper proxy baselines"
```

### Task 5: Re-run the CASES experiment matrix from one checkpoint

**Files:**
- Modify: `scripts/build_graphpilot_checkpoint.py`
- Modify: `scripts/calibrate_graphpilot_cost_model.py`
- Modify: `scripts/run_graphpilot_characterization.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Outputs under: `artifacts/graphpilot_edge/checkpoints/`, `artifacts/graphpilot_edge/analysis/`, `artifacts/graphpilot_edge/experiments/`, `artifacts/graphpilot_edge/reports/`

**Step 1: Generate a fresh canonical checkpoint**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_checkpoint.py
```
Expected: a new `artifacts/graphpilot_edge/checkpoints/graphpilot_checkpoint_<timestamp>/summary.json`

**Step 2: Run calibration from the checkpoint**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/calibrate_graphpilot_cost_model.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: a new calibration summary with family residuals and bias tables.

**Step 3: Run characterization and baselines from the same checkpoint**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_characterization.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: a new characterization summary with continuous-stream studies, fallback penalties, baseline comparisons, and ablations.

**Step 4: Build artifact pack from the checkpoint**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_artifact_pack.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: a new artifact pack summary that references only checkpoint-pinned evidence.

**Step 5: Manually verify the truth surface**
Check:
```bash
python3 - <<'PY'
import json
from pathlib import Path
pack = json.loads(Path('<artifact_pack_summary_json>').read_text())
print(pack['checkpoint_manifest'])
print(pack['report'])
print(pack['paper_draft'])
PY
```
Expected: all outputs point to the same checkpoint.

**Step 6: Commit**
```bash
git add artifacts/graphpilot_edge/checkpoints artifacts/graphpilot_edge/analysis artifacts/graphpilot_edge/experiments artifacts/graphpilot_edge/reports scripts/build_graphpilot_checkpoint.py scripts/calibrate_graphpilot_cost_model.py scripts/run_graphpilot_characterization.py scripts/build_graphpilot_artifact_pack.py
 git commit -m "feat: refresh graphpilot CASES evidence set"
```

### Task 6: Build CASES figures and tables from the canonical checkpoint

**Files:**
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Create: `scripts/build_graphpilot_cases_figures.py`
- Create: `scripts/tests/test_build_graphpilot_cases_figures.py`
- Output under: `artifacts/graphpilot_edge/papers/` or checkpoint-pinned figure dir

**Step 1: Write failing tests**
Add tests that the figure builder emits at least:
- `architecture_overview.svg`
- `sim_real_calibration.svg`
- `workflow_primary_results.svg`
- `continuous_stream_results.svg`
- `baseline_comparison.svg`
- `ablation_breakdown.svg`
- `fallback_penalty.svg`
- `tables.tex`

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_cases_figures -v
```
Expected: FAIL because the builder does not exist yet.

**Step 3: Implement figure/table builder**
- Create `scripts/build_graphpilot_cases_figures.py`
- Consume `--checkpoint-manifest`
- Fail fast if a required metric section is absent
- Emit SVG/PDF-ready figures and LaTeX table fragments

**Step 4: Run tests to verify they pass**
Run the unittest command from Step 2.
Expected: PASS.

**Step 5: Commit**
```bash
git add scripts/build_graphpilot_cases_figures.py scripts/tests/test_build_graphpilot_cases_figures.py scripts/build_graphpilot_artifact_pack.py
 git commit -m "feat: add graphpilot CASES figures and tables"
```

### Task 7: Write the IEEE ESWEEK CASES paper source and build the PDF

**Files:**
- Create: `papers/graphpilot_cases_2026/main.tex`
- Create: `papers/graphpilot_cases_2026/sections/abstract.tex`
- Create: `papers/graphpilot_cases_2026/sections/introduction.tex`
- Create: `papers/graphpilot_cases_2026/sections/related_work.tex`
- Create: `papers/graphpilot_cases_2026/sections/system_design.tex`
- Create: `papers/graphpilot_cases_2026/sections/simulator.tex`
- Create: `papers/graphpilot_cases_2026/sections/methodology.tex`
- Create: `papers/graphpilot_cases_2026/sections/results.tex`
- Create: `papers/graphpilot_cases_2026/sections/limitations.tex`
- Create: `papers/graphpilot_cases_2026/sections/conclusion.tex`
- Create: `papers/graphpilot_cases_2026/references.bib`
- Modify: `scripts/build_graphpilot_cases_paper.py`

**Step 1: Add a failing paper-build test**
Extend `scripts/tests/test_build_graphpilot_cases_paper.py` so it expects a populated LaTeX tree and explicit PDF build status.

**Step 2: Run test to verify it fails**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_cases_paper -v
```
Expected: FAIL because the paper source tree is missing.

**Step 3: Write the paper source**
- Use IEEEtran-compatible two-column layout
- Keep the claim aligned to the verified system
- Pull figures/tables from the checkpoint-pinned builder
- Include explicit limitations and realized-vs-simulated distinction

**Step 4: Build the PDF**
Run:
```bash
python3 scripts/build_graphpilot_cases_paper.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: a built `paper.pdf`, or an explicit toolchain failure with remediation.

**Step 5: Manually inspect the PDF**
Check:
- page count
- figure readability
- table overflow
- claim consistency vs final audit

**Step 6: Commit**
```bash
git add papers/graphpilot_cases_2026 scripts/build_graphpilot_cases_paper.py scripts/tests/test_build_graphpilot_cases_paper.py
 git commit -m "feat: add graphpilot CASES paper source"
```

### Task 8: Final verification, push, and release checkpoint

**Files:**
- Outputs under: `artifacts/graphpilot_edge/checkpoints/`, `artifacts/graphpilot_edge/reports/`, `artifacts/graphpilot_edge/papers/`

**Step 1: Run the full verification suite**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
ANDROID_HOME=/home/aryamavmurthy/android-sdk ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk ./gradlew app:testDebugUnitTest --tests 'com.qidk.fastvlm.core.graphpilot.*'
ANDROID_HOME=/home/aryamavmurthy/android-sdk ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk ./gradlew app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest
```
Expected: PASS / BUILD SUCCESSFUL.

**Step 2: Verify paper-to-checkpoint consistency**
Run:
```bash
python3 - <<'PY'
import json
from pathlib import Path
pack = json.loads(Path('<artifact_pack_summary_json>').read_text())
assert Path(pack['checkpoint_manifest']).exists()
assert Path(pack['paper_draft']).exists()
assert Path(pack['final_audit_report']).exists()
print('ok')
PY
```
Expected: `ok`

**Step 3: Push final state**
Run:
```bash
git push origin codex/litert-deepdive
bd sync
```
Expected: remote branch updated and beads synced.

**Step 4: Final commit if needed**
```bash
git add artifacts/graphpilot_edge/checkpoints artifacts/graphpilot_edge/reports artifacts/graphpilot_edge/papers
 git commit -m "feat: finalize graphpilot CASES submission package"
```
