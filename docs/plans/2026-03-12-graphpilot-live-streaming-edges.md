# GraphPilot Live Streaming Edges Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a support-safe live streaming edge to GraphPilot so responder tokens stream into chunked TTS when the plan declares `responder.primary -> tts.primary` in `TOKEN` mode.

**Architecture:** Preserve the existing support-safe runtime and current backends. Extend the LiteRT-LM text adapter with real delta streaming via `sendMessageAsync`, add a small streaming speech interface around `StreamingTtsCoordinator`, and let `GraphPilotCoordinator` execute workflow A with either full-response TTS or token-streamed TTS based on the plan registry.

**Tech Stack:** Kotlin, Android coroutines, LiteRT-LM, Android TTS, Gradle unit + instrumentation tests, GraphPilot artifact reporting.

---

### Task 1: Add failing tests for text streaming and streamed-TTS decision logic

**Files:**
- Modify: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinatorTest.kt`
- Create: `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingResponderTest.kt`

**Step 1: Write the failing tests**
- Add a pure Kotlin test for a streaming responder helper that receives deltas and forwards them to a speech streamer.
- Add a plan-gating test that only enables streamed TTS when the plan contains `responder.primary -> tts.primary` with `mode=token`.

**Step 2: Run tests to verify they fail**
Run: `cd android-app && ./gradlew --max-workers=2 :app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotStreamingResponderTest --tests com.qidk.fastvlm.core.graphpilot.GraphPilotCoordinatorTest`
Expected: FAIL because no streaming responder helper or stream-edge gating exists.

**Step 3: Write minimal implementation**
- Add the smallest new abstractions needed to support streaming responder orchestration.
- Do not change actual backend behavior yet.

**Step 4: Run tests to verify they pass**
Run the same command.
Expected: PASS.

### Task 2: Add a real LiteRT-LM streaming API to the text stage adapter

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/text/LiteRtLmTextStageAdapter.kt`
- Modify: `android-app/app/src/androidTest/java/com/qidk/fastvlm/text/LiteRtLmTextStageInstrumentedTest.kt`

**Step 1: Write the failing instrumentation test**
- Add a focused device test that requires non-empty streamed deltas and a non-empty final assembled response from the responder model.

**Step 2: Run it to verify it fails**
Run the single instrumentation test.
Expected: FAIL because only full-response generation exists.

**Step 3: Write minimal implementation**
- Add `generateStreaming(...)` using `sendMessageAsync`.
- Fail fast if no non-empty delta is emitted or if final text is empty.

**Step 4: Run the focused instrumentation test**
Expected: PASS.

### Task 3: Integrate streamed responder -> chunked TTS in GraphPilotCoordinator

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- Possibly create: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingResponder.kt`
- Test: `android-app/app/src/androidTest/java/com/qidk/fastvlm/graphpilot/GraphPilotCoordinatorInstrumentedTest.kt`

**Step 1: Write the failing tests**
- Add a coordinator/device test for workflow A that asserts streamed-TTS observability fields/logs are emitted when the plan stream edge is `TOKEN`.
- Add unit coverage for stream-edge gating.

**Step 2: Run tests to verify they fail**
Run focused JVM + instrumentation tests.
Expected: FAIL because coordinator still uses full-response TTS.

**Step 3: Write minimal implementation**
- If the plan declares `responder.primary -> tts.primary` token streaming, start a streaming TTS session, feed responder deltas into `StreamingTtsCoordinator`, finalize the session, and record TTFT/TTFS-style timestamps.
- Keep the existing full-response TTS path as the explicit non-streaming path.
- No hidden fallback: if streaming is requested and the streaming path fails, surface the error.

**Step 4: Run tests to verify they pass**
Run the focused JVM and instrumentation tests.
Expected: PASS.

### Task 4: Update reporting and regenerate GraphPilot artifacts

**Files:**
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Possibly modify: `artifacts/graphpilot_edge/state/state_ledger.json`

**Step 1: Write/report assertions**
- Add tests requiring the artifact pack to surface streaming-edge evidence when present.

**Step 2: Implement minimal report updates**
- Include streamed responder / TTS evidence in the report and paper draft.

**Step 3: Run tests**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_artifact_pack`
Expected: PASS.

### Task 5: Verify end to end and checkpoint

**Files:**
- Update: Beads issue `fvlm-kzg.1`
- Update: `artifacts/graphpilot_edge/state/state_ledger.json`

**Step 1: Full verification**
Run:
- `cd android-app && ./gradlew --max-workers=2 :app:testDebugUnitTest --tests com.qidk.fastvlm.core.graphpilot.*`
- `cd android-app && ./gradlew --max-workers=2 :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest`
- `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_artifact_pack`

**Step 2: Regenerate artifact pack**
- Rebuild the GraphPilot artifact pack with the new streaming-edge evidence.

**Step 3: Update Beads and ledger**
- Record actual evidence paths and close `fvlm-kzg.1` only after verification is fresh.
