package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.test.runTest
import org.junit.Test

class GraphPilotRuntimeSchedulerTest {
  private val plan =
    GraphPilotExecutionPlan(
      plan_id = "cool:workflow_a:test",
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
        ),
    )

  @Test
  fun firstRequestStartsImmediately() = runTest {
    val scheduler = GraphPilotRuntimeScheduler(maxQueuedRequests = 1, clockMs = { 0L })

    val (observation, value) =
      scheduler.runWithAdmission(
        workflowId = "workflow_a_voice_only",
        plan = plan,
      ) { "ok" }

    assertThat(value).isEqualTo("ok")
    assertThat(observation.queueDepthAtAdmission).isEqualTo(0)
    assertThat(observation.predictedQueueDelayMs).isEqualTo(0.0)
    assertThat(observation.queueWaitMs).isEqualTo(0L)
    assertThat(observation.admissionDecision).isEqualTo(GraphPilotAdmissionDecision.ADMIT)
  }

  @Test
  fun secondRequestQueuesBehindActiveRequest() = runTest {
    val scheduler = GraphPilotRuntimeScheduler(maxQueuedRequests = 2, clockMs = { 0L })
    val releaseFirst = CompletableDeferred<Unit>()

    val first =
      async {
        scheduler.runWithAdmission(
          workflowId = "workflow_a_voice_only",
          plan = plan,
        ) { observation ->
          releaseFirst.await()
          observation
        }
      }

    val second =
      async {
        scheduler.runWithAdmission(
          workflowId = "workflow_a_voice_only",
          plan = plan,
        ) { observation ->
          observation
        }
      }

    testScheduler.advanceUntilIdle()
    assertThat(second.isCompleted).isFalse()

    releaseFirst.complete(Unit)
    val (firstObservation, _) = first.await()
    val (secondObservation, _) = second.await()

    assertThat(firstObservation.queueDepthAtAdmission).isEqualTo(0)
    assertThat(secondObservation.queueDepthAtAdmission).isEqualTo(1)
    assertThat(secondObservation.predictedQueueDelayMs).isEqualTo(1000.0)
    assertThat(secondObservation.queueWaitMs).isAtLeast(0L)
  }

  @Test(expected = GraphPilotAdmissionException::class)
  fun rejectsRequestWhenQueueIsFull() = runTest {
    val scheduler = GraphPilotRuntimeScheduler(maxQueuedRequests = 0, clockMs = { 0L })
    val releaseFirst = CompletableDeferred<Unit>()

    val first =
      async {
        scheduler.runWithAdmission(
          workflowId = "workflow_a_voice_only",
          plan = plan,
        ) {
          releaseFirst.await()
          "first"
        }
      }

    testScheduler.advanceUntilIdle()

    try {
      scheduler.runWithAdmission(
        workflowId = "workflow_a_voice_only",
        plan = plan,
      ) { "second" }
    } finally {
      releaseFirst.complete(Unit)
      first.await()
    }
  }
}
