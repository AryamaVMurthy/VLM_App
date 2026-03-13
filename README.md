# GraphPilot-Edge

GraphPilot-Edge is a profiler-driven runtime, calibrated simulator, and artifact surface for continuous multimodal assistant DAGs on heterogeneous mobile SoCs.

The repository now contains three aligned surfaces:
- the deployed Android runtime under `android-app/`
- the offline planner / simulator / calibration code under `graphpilot_edge/`
- the checkpoint-pinned evidence surface under `artifacts/graphpilot_edge/`

This repo does **not** use directory reshuffles or moving artifact outputs as a cleanup strategy. Canonical checkpoint, report, and paper paths are part of the audited evidence surface and should remain stable.

## Current Verified Scope

The current verified GraphPilot-Edge system anchors on three support-safe workflows:
- Workflow A: `ASR -> planner -> responder -> TTS`
- Workflow B: `ASR -> planner -> VLM -> responder -> TTS`
- Workflow C: `ASR -> planner -> (VLM || retrieval) -> responder -> TTS`

Current mainline deployment facts:
- FastVLM remains on LiteRT NPU.
- Text, retrieval, and speech stages remain on support-safe CPU paths unless a stronger backend path is explicitly validated.
- Silent fallback is not counted as successful acceleration.

## Canonical Outputs

Use these pinned outputs as the current repo truth surface:
- CASES paper PDF: `artifacts/graphpilot_edge/papers/graphpilot_cases_20260312_195225/main.pdf`
- Current checkpoint summary: `artifacts/graphpilot_edge/checkpoints/graphpilot_checkpoint_20260312_195112/summary.json`
- Current artifact pack summary: `artifacts/graphpilot_edge/reports/artifact_pack_20260312_195226/summary.json`
- Revision scope boundary: `Truth-docs/graphpilot_edge_revision_report.pdf`
- Artifact retention policy: `docs/ARTIFACT_RETENTION.md`

## Start Here

- Navigation index: `docs/INDEX.md`
- Repo layout and path-stability rules: `docs/REPO_LAYOUT.md`
- Canonical artifact surface: `artifacts/graphpilot_edge/README.md`
- Offline planner/simulator package: `graphpilot_edge/README.md`
- Android runtime surface: `android-app/README.md`
- Script entry points: `scripts/README.md`
- Claim boundary and revision source: `Truth-docs/README.md`
- Artifact retention and archive rules: `docs/ARTIFACT_RETENTION.md`

## Common Tasks

### Python verification

Use a clean Python environment invocation because this shell can inherit broken `PYTHONHOME` / `PYTHONPATH` values.

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```

### Android verification

```bash
cd android-app
ANDROID_HOME=/home/aryamavmurthy/android-sdk \
ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk \
./gradlew app:testDebugUnitTest --tests 'com.qidk.fastvlm.core.graphpilot.*'
```

For connected tests, ensure the root VLM daemon is running first:

```bash
./scripts/start_root_vlm_daemon_adb.sh
cd android-app
ANDROID_HOME=/home/aryamavmurthy/android-sdk \
ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk \
./gradlew app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest
```

### Rebuild checkpoint / paper surface

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_checkpoint.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_cases_figures.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_cases_paper.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_artifact_pack.py
```

## Guardrails

- Do not move or rename canonical artifact directories under `artifacts/graphpilot_edge/`.
- Do not replace `Truth-docs/graphpilot_edge_revision_report.pdf` with an untracked local draft.
- Fail fast on support-safety issues; do not add hidden defaults or silent degradation.
- Treat the artifact pack as evidence, not as scratch space.
