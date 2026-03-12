# GraphPilot Runtime-First Expanded Scope Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen the GraphPilot-Edge runtime first, then use the stronger online system as the anchor for the expanded simulator, workload, baseline, calibration, and paper phases.

**Architecture:** Preserve the current support-safe workflows and execution paths while expanding plan schemas, runtime scheduling, memory/KV policy, and streaming behavior in backward-compatible layers. The online brain becomes backend-aware and queue-aware first; the offline brain then expands around the new runtime contract.

**Tech Stack:** Kotlin, Android instrumentation/unit tests, Python 3 scripts, GraphPilot offline planner/simulator, Beads task graph, LiteRT/LiteRT-LM integration, JSON registries.

---

### Task 1: Revalidate the current audited runtime and lock the runtime-first scope

**Files:**
- Create: `docs/plans/2026-03-12-graphpilot-runtime-first-expanded-design.md`
- Create: `docs/plans/2026-03-12-graphpilot-runtime-first-expanded-implementation.md`
- Modify: `.beads` issue graph through `bd`

**Step 1: Verify the current GraphPilot offline unit surface**

Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
```

Expected: all tests pass.

**Step 2: Verify the current GraphPilot script test surface**

Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```

Expected: all tests pass.

**Step 3: Record runtime-first design and implementation docs**

Create the two docs above and map the phase order to the Beads issue graph.

**Step 4: Claim the active Beads issue**

Run:
```bash
bd update fvlm-i6n.1 --claim
```

Expected: issue enters `in_progress`.

### Task 2: Extend runtime plan schemas and objective fields

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotPlanStore.kt`
- Modify: `graphpilot_edge/cost_model.py`
- Modify: `graphpilot_edge/simulation.py`
- Modify: `graphpilot_edge/planner.py`
- Modify: `tests/graphpilot_edge/test_cost_model.py`
- Modify: `tests/graphpilot_edge/test_simulation.py`
- Modify: `tests/graphpilot_edge/test_planner.py`
- Modify: `scripts/tests/test_generate_graphpilot_candidate_plans.py`

**Step 1: Write failing tests for 8-term objective support**

Add tests that require:

- queue delay and deadline miss to contribute to objective score,
- predicted-cost schema to carry full objective inputs,
- planner/simulation outputs to preserve the extra fields.

**Step 2: Run the failing tests**

Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_cost_model tests.graphpilot_edge.test_simulation tests.graphpilot_edge.test_planner scripts.tests.test_generate_graphpilot_candidate_plans -v
```

Expected: failures for missing schema/objective fields.

**Step 3: Implement minimal schema and cost-model changes**

Add the new objective terms and keep backward-compatible defaults.

**Step 4: Re-run the focused tests**

Expected: pass.

### Task 3: Replace FIFO runtime scheduling with backend-aware scoring

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeScheduler.kt`
- Modify: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeSchedulerTest.kt`
- Create: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotSchedulerPolicyTest.kt`

**Step 1: Write failing tests for backend-aware scoring**

Add tests that require:

- per-backend queue accounting,
- first-output bias in scheduling metadata,
- deadline/slack-aware admission ranking,
- no regression of the current single-request behavior.

**Step 2: Run the focused JVM tests**

Run:
```bash
./gradlew :android-app:app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotRuntimeSchedulerTest
```

Expected: failures for the new policy cases.

**Step 3: Implement the scheduler changes**

Add explicit scoring inputs, backend-resource state, and queue metadata while preserving current coordinator entrypoints.

**Step 4: Re-run the focused tests**

Expected: pass.

### Task 4: Extend runtime memory/KV policy and plan-bank switching inputs

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotMemoryAdmissionController.kt`
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- Create: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotThermalPlanBank.kt`
- Modify: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotMemoryAdmissionControllerTest.kt`
- Modify: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeMemoryManagerTest.kt`

**Step 1: Write failing tests for richer degradation and thermal state selection**

Require:

- KV-aware policy fields,
- explicit degradation actions in runtime observations,
- plan-bank state switching decisions with hysteresis.

**Step 2: Run the focused JVM tests**

Run:
```bash
./gradlew :android-app:app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotMemoryAdmissionControllerTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotRuntimeMemoryManagerTest
```

Expected: failures for the new policy surfaces.

**Step 3: Implement the minimal controller and state-selection logic**

**Step 4: Re-run the focused tests**

Expected: pass.

### Task 5: Generalize runtime stream-edge handling

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingResponder.kt`
- Create: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingEdgePolicyTest.kt`

**Step 1: Write failing tests for STT-to-planner and generic stream-edge behavior**

Require:

- token/chunk edge recognition from plan metadata,
- no regression to responder-to-TTS streaming,
- explicit fail-fast behavior if a requested stream edge is unsupported online.

**Step 2: Run the focused JVM tests**

Run:
```bash
./gradlew :android-app:app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotStreamingResponderTest
```

Expected: failures for the new stream-edge cases.

**Step 3: Implement the smallest runtime extension that preserves current flows**

**Step 4: Re-run the focused tests**

Expected: pass.

### Task 6: Run regression verification for the runtime-first slice

**Files:**
- Modify: `docs/plans/2026-03-12-graphpilot-runtime-first-expanded-design.md`
- Modify: `docs/plans/2026-03-12-graphpilot-runtime-first-expanded-implementation.md`
- Update Beads issues with notes

**Step 1: Run GraphPilot offline tests again**

Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```

Expected: pass.

**Step 2: Run focused Android JVM tests**

Run:
```bash
./gradlew :android-app:app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotRuntimeSchedulerTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotMemoryAdmissionControllerTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotRuntimeMemoryManagerTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotStreamingResponderTest
```

Expected: pass.

**Step 3: Update Beads with evidence and close completed sub-issues**

Use `bd update ... --append-notes` or `bd close ...` with verification evidence.

**Step 4: Commit**

```bash
git add docs/plans android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot graphpilot_edge tests/graphpilot_edge scripts/tests
git commit -m "Strengthen GraphPilot runtime scheduling and plan schema"
```

### Task 7: Continue with simulator, workloads, baselines, calibration, characterization, and final reports

**Files:**
- Modify all phase-specific files referenced by Beads tasks `fvlm-i6n.6` through `fvlm-i6n.12`

**Step 1: Claim the next ready Beads issue**

Run:
```bash
bd ready
bd update <next-id> --claim
```

**Step 2: Execute in dependency order**

Priority order:

1. `fvlm-i6n.6` hardware simulator
2. `fvlm-i6n.7` model-graph simulator
3. `fvlm-i6n.8` workload universe
4. `fvlm-i6n.9` baseline suite
5. `fvlm-i6n.10` sim-to-real calibration
6. `fvlm-i6n.11` characterization and ablations
7. `fvlm-i6n.12` final artifact/report/paper/audit

**Step 3: After each task, verify, update Beads, commit, and continue**

Expected: the work proceeds phase-by-phase without losing the regression anchor.
