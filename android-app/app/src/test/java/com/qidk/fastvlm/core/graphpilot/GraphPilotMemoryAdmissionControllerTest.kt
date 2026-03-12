package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class GraphPilotMemoryAdmissionControllerTest {
  private val budget = GraphPilotMemoryBudget(totalBytes = 1_000L, marginBytes = 100L)
  private val controller = GraphPilotMemoryAdmissionController(latencyPenalty = 0.01)

  @Test
  fun admitsWhenRequestedBytesFitBudget() {
    val decision =
      controller.decide(
        profile =
          GraphPilotRequestMemoryProfile(
            workflowId = "workflow_a_voice_only",
            baseBytes = 600L,
            kvBytes = 100L,
            estimationSource = "unit_test",
          ),
        budget = budget,
      )

    assertThat(decision.type).isEqualTo(GraphPilotMemoryDecisionType.ADMIT)
    assertThat(decision.requiredBytes).isEqualTo(700L)
    assertThat(decision.appliedActions).isEmpty()
  }

  @Test
  fun degradesUsingLowestHarmRatioUntilBudgetFits() {
    val decision =
      controller.decide(
        profile =
          GraphPilotRequestMemoryProfile(
            workflowId = "workflow_b_voice_vision",
            baseBytes = 700L,
            kvBytes = 250L,
            estimationSource = "unit_test",
            degradationOptions =
              listOf(
                GraphPilotDegradationOption(
                  id = "reduce_vlm_tokens",
                  freedBytes = 200L,
                  addedLatencyMs = 5.0,
                  qualityLoss = 0.2,
                  vlmVisualTokenBudget = 128,
                ),
                GraphPilotDegradationOption(
                  id = "reduce_responder_max_tokens",
                  freedBytes = 100L,
                  addedLatencyMs = 2.0,
                  qualityLoss = 0.5,
                  responderMaxTokens = 24,
                ),
              ),
          ),
        budget = budget,
      )

    assertThat(decision.type).isEqualTo(GraphPilotMemoryDecisionType.DEGRADE)
    assertThat(decision.appliedActions.map { it.id }).containsExactly("reduce_vlm_tokens")
    assertThat(decision.effectiveRequiredBytes).isEqualTo(750L)
    assertThat(decision.effectiveVlmVisualTokenBudget).isEqualTo(128)
  }

  @Test
  fun rejectsWhenEvenDegradedRequestDoesNotFitBudget() {
    val decision =
      controller.decide(
        profile =
          GraphPilotRequestMemoryProfile(
            workflowId = "workflow_c_voice_vision_retrieval",
            baseBytes = 900L,
            kvBytes = 500L,
            estimationSource = "unit_test",
            degradationOptions =
              listOf(
                GraphPilotDegradationOption(
                  id = "reduce_retrieval_top_k",
                  freedBytes = 100L,
                  addedLatencyMs = 1.0,
                  qualityLoss = 0.1,
                  retrievalTopK = 1,
                )
              ),
          ),
        budget = budget,
      )

    assertThat(decision.type).isEqualTo(GraphPilotMemoryDecisionType.REJECT)
    assertThat(decision.effectiveRequiredBytes).isEqualTo(1300L)
    assertThat(decision.reason).contains("memory budget")
  }
}
