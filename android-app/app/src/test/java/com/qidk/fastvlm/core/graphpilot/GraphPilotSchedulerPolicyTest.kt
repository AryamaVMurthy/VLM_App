package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.test.runTest
import org.junit.Test

class GraphPilotSchedulerPolicyTest {
  private val cpuPlan =
    GraphPilotExecutionPlan(
      plan_id = "cool:workflow_a:cpu",
      workflow_template = "workflow_a_voice_only",
      state_id = "cool",
      backend_map =
        mapOf(
          "asr.primary" to "cpu",
          "planner.primary" to "cpu",
          "responder.primary" to "cpu",
          "tts.primary" to "cpu",
        ),
      predictedCost =
        GraphPilotPredictedCost(
          streamMakespanMs = 1000.0,
          p95QueueDelayMs = 250.0,
          deadlineMissRate = 0.0,
          ttftMs = 400.0,
        ),
    )

  private val npuPlan =
    GraphPilotExecutionPlan(
      plan_id = "cool:workflow_b:npu",
      workflow_template = "workflow_b_voice_vision",
      state_id = "cool",
      backend_map = mapOf("vlm.fastvlm.primary" to "npu"),
      predictedCost =
        GraphPilotPredictedCost(
          streamMakespanMs = 800.0,
          p95QueueDelayMs = 50.0,
          deadlineMissRate = 0.0,
          ttftMs = 100.0,
        ),
    )

  private val lowPriorityCpuPlan =
    GraphPilotExecutionPlan(
      plan_id = "cool:workflow_a:low_priority",
      workflow_template = "workflow_a_voice_only",
      state_id = "cool",
      backend_map =
        mapOf(
          "asr.primary" to "cpu",
          "planner.primary" to "cpu",
          "responder.primary" to "cpu",
          "tts.primary" to "cpu",
        ),
      predictedCost =
        GraphPilotPredictedCost(
          streamMakespanMs = 1200.0,
          p95QueueDelayMs = 600.0,
          deadlineMissRate = 0.0,
          ttftMs = 900.0,
        ),
    )

  private val highPriorityCpuPlan =
    GraphPilotExecutionPlan(
      plan_id = "cool:workflow_a:high_priority",
      workflow_template = "workflow_a_voice_only",
      state_id = "cool",
      backend_map =
        mapOf(
          "asr.primary" to "cpu",
          "planner.primary" to "cpu",
          "responder.primary" to "cpu",
          "tts.primary" to "cpu",
        ),
      predictedCost =
        GraphPilotPredictedCost(
          streamMakespanMs = 900.0,
          p95QueueDelayMs = 50.0,
          deadlineMissRate = 0.25,
          ttftMs = 50.0,
        ),
    )

  @Test
  fun disjointBackendRequestStartsWhileCpuRequestIsActive() = runTest {
    val scheduler = GraphPilotRuntimeScheduler(maxQueuedRequests = 0, clockMs = { 0L })
    val releaseCpu = CompletableDeferred<Unit>()

    val first =
      async {
        scheduler.runWithAdmission("workflow_a_voice_only", cpuPlan) {
          releaseCpu.await()
          "cpu"
        }
      }

    testScheduler.advanceUntilIdle()

    val (npuObservation, value) =
      scheduler.runWithAdmission("workflow_b_voice_vision", npuPlan) {
        "npu"
      }

    assertThat(value).isEqualTo("npu")
    assertThat(npuObservation.queueDepthAtAdmission).isEqualTo(0)
    assertThat(npuObservation.predictedQueueDelayMs).isEqualTo(0.0)
    assertThat(npuObservation.requiredBackends).containsExactly("npu")

    releaseCpu.complete(Unit)
    first.await()
  }

  @Test
  fun higherPriorityPendingCpuRequestStartsBeforeLowerPriorityRequest() = runTest {
    val scheduler = GraphPilotRuntimeScheduler(maxQueuedRequests = 3, clockMs = { 0L })
    val releaseActive = CompletableDeferred<Unit>()
    val releaseLow = CompletableDeferred<Unit>()
    val releaseHigh = CompletableDeferred<Unit>()
    val startOrder = mutableListOf<String>()

    val active =
      async {
        scheduler.runWithAdmission("workflow_a_voice_only", cpuPlan) {
          releaseActive.await()
          "active"
        }
      }

    val low =
      async {
        scheduler.runWithAdmission("workflow_a_voice_only", lowPriorityCpuPlan) {
          startOrder += "low"
          releaseLow.await()
          "low"
        }
      }

    val high =
      async {
        scheduler.runWithAdmission("workflow_a_voice_only", highPriorityCpuPlan) {
          startOrder += "high"
          releaseHigh.await()
          "high"
        }
      }

    testScheduler.advanceUntilIdle()
    assertThat(startOrder).isEmpty()

    releaseActive.complete(Unit)
    testScheduler.advanceUntilIdle()

    assertThat(startOrder).containsExactly("high")

    releaseHigh.complete(Unit)
    high.await()
    testScheduler.advanceUntilIdle()

    assertThat(startOrder).containsExactly("high", "low").inOrder()

    releaseLow.complete(Unit)
    low.await()
    active.await()
  }
}
