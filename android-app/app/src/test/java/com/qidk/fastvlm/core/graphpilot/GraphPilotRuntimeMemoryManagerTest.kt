package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Test

class GraphPilotRuntimeMemoryManagerTest {
  private val controller = GraphPilotMemoryAdmissionController(latencyPenalty = 0.01)
  private val estimator = GraphPilotMemoryProfileEstimator()

  @Test
  fun admitsAndReservesMemoryUntilRequestCompletes() = runBlocking {
    val manager =
      GraphPilotRuntimeMemoryManager(
        budget = GraphPilotMemoryBudget(totalBytes = mebibytes(192), marginBytes = mebibytes(16)),
        controller = controller,
      )
    val profile = estimator.estimate("workflow_a_voice_only", predictedCost = null)

    val deferred =
      async {
        manager.runWithAdmission("workflow_a_voice_only", "plan-a", profile) { observation ->
          assertThat(observation.activeReservedBytesAtAdmission).isEqualTo(0L)
          delay(150)
          observation
        }
      }

    delay(30)
    val active = manager.activeReservedBytes()
    assertThat(active).isGreaterThan(0L)

    val observation = deferred.await()
    assertThat(observation.decision.type).isEqualTo(GraphPilotMemoryDecisionType.ADMIT)
    assertThat(manager.activeReservedBytes()).isEqualTo(0L)
  }

  @Test
  fun rejectsSecondRequestWhenReservedMemoryExhaustsBudget() = runBlocking {
    val manager =
      GraphPilotRuntimeMemoryManager(
        budget = GraphPilotMemoryBudget(totalBytes = mebibytes(192), marginBytes = mebibytes(16)),
        controller = controller,
      )
    val profile = estimator.estimate("workflow_a_voice_only", predictedCost = null)

    val first =
      async {
        manager.runWithAdmission("workflow_a_voice_only", "plan-a", profile) {
          delay(150)
        }
      }
    delay(30)

    val error =
      runCatching {
        manager.runWithAdmission("workflow_a_voice_only", "plan-a", profile) { }
      }.exceptionOrNull()

    assertThat(error).isInstanceOf(GraphPilotMemoryAdmissionException::class.java)
    val typed = error as GraphPilotMemoryAdmissionException
    assertThat(typed.observation.decision.type).isEqualTo(GraphPilotMemoryDecisionType.REJECT)
    assertThat(typed.observation.reason).contains("memory budget")

    first.await()
  }
}
