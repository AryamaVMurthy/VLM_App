package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class GraphPilotMemoryProfileEstimatorTest {
  private val estimator = GraphPilotMemoryProfileEstimator()

  @Test
  fun fallsBackToExplicitWorkflowHeuristicsWhenPredictedPeakMemoryIsUnavailable() {
    val profile = estimator.estimate("workflow_b_voice_vision", predictedCost = null)

    assertThat(profile.workflowId).isEqualTo("workflow_b_voice_vision")
    assertThat(profile.requiredBytes).isGreaterThan(0L)
    assertThat(profile.estimationSource).isEqualTo("workflow_heuristic_v1")
    assertThat(profile.degradationOptions.map { it.id }).contains("reduce_responder_max_tokens")
  }

  @Test
  fun usesPredictedPeakMemoryWhenAvailable() {
    val predictedCost =
      GraphPilotPredictedCost(
        streamMakespanMs = 1000.0,
        p95QueueDelayMs = 100.0,
        deadlineMissRate = 0.0,
        peakMemoryBytes = mebibytes(64),
      )

    val profile = estimator.estimate("workflow_a_voice_only", predictedCost)

    assertThat(profile.baseBytes).isEqualTo(mebibytes(64))
    assertThat(profile.estimationSource).isEqualTo("predicted_cost_peak_memory")
  }
}
