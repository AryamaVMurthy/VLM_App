# GraphPilot Truth Surface and Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze a canonical pre-expansion GraphPilot checkpoint, calibrate the simulator against actual SM8750 device/model-family evidence using the revision PDF formulas, then rerun baselines/ablations and regenerate a single-source artifact pack.

**Architecture:** Add an explicit checkpoint manifest that pins every report and artifact to one exact evidence set instead of auto-selecting the latest files. Extend the calibration pipeline from workflow-level residuals to family/stage/backend calibration driven by measured profiler and device artifacts, then require baseline and artifact generation to consume that calibrated manifest. The final audit must derive open/partial/pass status from the manifest and explicit completion gates, not optimistic heuristics.

**Tech Stack:** Python 3, existing GraphPilot scripts, JSON registries, Beads, Android/QIDK evidence artifacts, Poppler text extraction for the revision PDF.

---

### Task 1: Add failing tests for the canonical checkpoint manifest

**Files:**
- Create: `scripts/tests/test_build_graphpilot_checkpoint.py`
- Modify: `scripts/tests/test_build_graphpilot_artifact_pack.py`

**Step 1: Write the failing tests**
- Add a test that builds a checkpoint manifest from explicit inputs and asserts the manifest records exact paths for experiment, sustained, calibration, characterization, tuning, memory-admission, backend matrix, profiler registry, candidate registry, baseline registry, workload registry, and state ledger.
- Add a test that `build_graphpilot_artifact_pack.py` refuses to mix latest-* auto discovery when `--checkpoint-manifest` is provided and instead uses only the pinned manifest paths.
- Add a test that the final audit written from a checkpoint manifest reports `PARTIAL` when open beads or incomplete baseline IDs are present.

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_checkpoint scripts.tests.test_build_graphpilot_artifact_pack -v
```
Expected: FAIL because no checkpoint builder or manifest-aware artifact-pack path exists yet.

**Step 3: Implement minimal checkpoint builder**
- Create a new script `scripts/build_graphpilot_checkpoint.py`.
- Emit `artifacts/graphpilot_edge/checkpoints/graphpilot_checkpoint_<timestamp>/summary.json` plus `report.md`.
- Fail fast if any required input path is missing.
- Include explicit `open_beads`, `closed_beads`, `verification_commands`, `truth_source_pdf`, and `canonical_evidence_paths` sections.

**Step 4: Make artifact-pack builder consume the checkpoint manifest**
- Add `--checkpoint-manifest` to `scripts/build_graphpilot_artifact_pack.py`.
- When present, use only manifest paths; do not call latest-file helpers for those inputs.
- Thread manifest metadata into `report.md`, `paper_draft.md`, `paper_tables.md`, and `final_audit_report.md`.

**Step 5: Run tests to verify they pass**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_checkpoint scripts.tests.test_build_graphpilot_artifact_pack -v
```
Expected: PASS.

**Step 6: Commit**
```bash
git add scripts/build_graphpilot_checkpoint.py scripts/build_graphpilot_artifact_pack.py scripts/tests/test_build_graphpilot_checkpoint.py scripts/tests/test_build_graphpilot_artifact_pack.py
git commit -m "feat: add canonical graphpilot checkpoint manifest"
```

### Task 2: Add failing tests for family-level calibration driven by revision-PDF formulas

**Files:**
- Modify: `scripts/tests/test_calibrate_graphpilot_cost_model.py` or create if missing
- Modify: `scripts/tests/test_run_graphpilot_characterization.py`
- Modify: `scripts/tests/test_generate_graphpilot_candidate_plans.py`

**Step 1: Write the failing tests**
- Add a test that calibration emits per-family/per-stage/per-backend surrogate parameters, not only workflow-level residuals.
- Add a test that calibration records launch overhead, transfer bias, thermal factors, contention factors, and residual error by family.
- Add a test that candidate-plan generation can consume an explicit checkpoint calibration path instead of latest auto-discovery.

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_calibrate_graphpilot_cost_model scripts.tests.test_generate_graphpilot_candidate_plans scripts.tests.test_run_graphpilot_characterization -v
```
Expected: FAIL because the current calibration only emits workflow-level surrogates.

**Step 3: Implement family-level calibration**
- Extend `scripts/calibrate_graphpilot_cost_model.py` to build calibration buckets by workload family / stage family / backend.
- Use explicit measured evidence from profiler registry and experiment summaries.
- Record residuals honestly; do not collapse disagreement into fake success.
- Save machine-readable family calibration tables and a compact report.

**Step 4: Thread calibrated surrogates into planning and characterization**
- Update `scripts/generate_graphpilot_candidate_plans.py` to prefer checkpoint-pinned calibration.
- Update characterization generation to expose calibration quality by family.
- Update any helper that still auto-picks latest calibration without manifest control.

**Step 5: Run tests to verify they pass**
Run the same unittest command from Step 2.
Expected: PASS.

**Step 6: Commit**
```bash
git add scripts/calibrate_graphpilot_cost_model.py scripts/generate_graphpilot_candidate_plans.py scripts/tests/test_calibrate_graphpilot_cost_model.py scripts/tests/test_generate_graphpilot_candidate_plans.py scripts/tests/test_run_graphpilot_characterization.py
git commit -m "feat: add family-level graphpilot calibration"
```

### Task 3: Add failing tests for honest baseline and ablation closure

**Files:**
- Modify: `graphpilot_edge/baselines.py`
- Modify: `scripts/generate_graphpilot_baseline_registry.py`
- Modify: `scripts/run_graphpilot_characterization.py`
- Modify: `tests/graphpilot_edge/test_baselines.py`
- Modify: `scripts/tests/test_generate_graphpilot_baseline_registry.py`
- Modify: `scripts/tests/test_run_graphpilot_characterization.py`

**Step 1: Write the failing tests**
- Add tests that baseline registry includes `no_fallback_aware`, `no_memory_kv`, `no_knob_tuning`, and `no_thermal_adaptation`.
- Add tests that characterization fails fast if required baseline IDs are missing from the registry.
- Add tests that baseline comparisons report when GraphPilot ties or loses, instead of implying a win.

**Step 2: Run tests to verify they fail**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_baselines scripts.tests.test_generate_graphpilot_baseline_registry scripts.tests.test_run_graphpilot_characterization -v
```
Expected: FAIL because these baseline IDs and gating checks do not exist yet.

**Step 3: Implement the missing ablation baselines**
- Add the ablation baseline policies to `graphpilot_edge/baselines.py`.
- Add baseline registry emission for them.
- Update characterization so GraphPilot vs baseline reporting is honest and workload-by-workload.

**Step 4: Run tests to verify they pass**
Run the same unittest command from Step 2.
Expected: PASS.

**Step 5: Commit**
```bash
git add graphpilot_edge/baselines.py scripts/generate_graphpilot_baseline_registry.py scripts/run_graphpilot_characterization.py tests/graphpilot_edge/test_baselines.py scripts/tests/test_generate_graphpilot_baseline_registry.py scripts/tests/test_run_graphpilot_characterization.py
git commit -m "feat: close graphpilot ablation baseline set"
```

### Task 4: Generate the canonical checkpoint, rerun calibration/baselines/ablations, and rebuild the artifact pack

**Files:**
- Modify: `docs/plans/2026-03-12-graphpilot-truth-surface-and-calibration.md`
- Outputs under: `artifacts/graphpilot_edge/checkpoints/`, `artifacts/graphpilot_edge/analysis/`, `artifacts/graphpilot_edge/reports/`

**Step 1: Generate the checkpoint manifest**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_checkpoint.py
```
Expected: writes a new checkpoint summary and report under `artifacts/graphpilot_edge/checkpoints/`.

**Step 2: Re-run calibration from the checkpoint evidence**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/calibrate_graphpilot_cost_model.py --checkpoint-manifest <checkpoint_summary_json> --output-root artifacts/graphpilot_edge/analysis
```
Expected: writes a new family-calibrated summary.

**Step 3: Rebuild baseline registry and characterization from the checkpoint**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/generate_graphpilot_baseline_registry.py --checkpoint-manifest <checkpoint_summary_json> --workloads $(env -u PYTHONHOME -u PYTHONPATH python3 - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('configs/graphpilot_edge/workload_universe.json').read_text())
print(' '.join(item['workload_id'] for item in cfg['workloads']))
PY)

env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_characterization.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: writes new baseline and characterization summaries pinned to the same checkpoint.

**Step 4: Build the artifact pack from the checkpoint manifest**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_artifact_pack.py --checkpoint-manifest <checkpoint_summary_json>
```
Expected: writes a new artifact pack whose `summary.json` references exactly the checkpoint-pinned evidence and whose final audit matches open beads and remaining missing baselines honestly.

**Step 5: Full verification**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```
Expected: PASS.

**Step 6: Commit**
```bash
git add docs/plans/2026-03-12-graphpilot-truth-surface-and-calibration.md artifacts/graphpilot_edge/checkpoints artifacts/graphpilot_edge/analysis artifacts/graphpilot_edge/reports
git commit -m "feat: freeze graphpilot truth surface and recalibrate artifacts"
```
