package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class GraphPilotCoordinatorTest {
  @Test
  fun buildAdmissionLogMessageIncludesPredictedStreamMetrics() {
    val plan =
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
            streamMakespanMs = 12345.0,
            p95E2eMs = 12000.0,
            p95TtfsMs = 7000.0,
            avgEnergyMj = 88.0,
            copyBytes = 4096L,
            qualityLoss = 0.1,
            p95QueueDelayMs = 456.0,
            deadlineMissRate = 0.25,
          ),
      )
    val observation =
      GraphPilotQueueObservation(
        requestId = 7L,
        workflowId = "workflow_a_voice_only",
        planId = plan.plan_id,
        queueDepthAtAdmission = 1,
        predictedQueueDelayMs = 1000.0,
        predictedStreamMakespanMs = 12345.0,
        predictedDeadlineMiss = false,
        deadlineMs = null,
        admittedAtMs = 1000L,
        admissionDecision = GraphPilotAdmissionDecision.ADMIT,
        queueWaitMs = 0L,
      )

    val message = buildAdmissionLogMessage("workflow_a_voice_only", plan, observation)

    assertThat(message).contains("workflow=workflow_a_voice_only")
    assertThat(message).contains("request_id=7")
    assertThat(message).contains("queue_depth_at_admission=1")
    assertThat(message).contains("predicted_queue_delay_at_admission_ms=1000.0")
    assertThat(message).contains("predicted_stream_makespan_ms=12345.0")
    assertThat(message).contains("predicted_p95_e2e_ms=12000.0")
    assertThat(message).contains("predicted_p95_ttfs_ms=7000.0")
    assertThat(message).contains("predicted_avg_energy_mj=88.0")
    assertThat(message).contains("predicted_copy_bytes=4096")
    assertThat(message).contains("predicted_quality_loss=0.1")
    assertThat(message).contains("predicted_p95_queue_ms=456.0")
    assertThat(message).contains("predicted_deadline_miss_rate=0.25")
  }

  @Test
  fun buildMemoryAdmissionLogMessageIncludesDecisionFields() {
    val plan =
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
      )
    val profile =
      GraphPilotRequestMemoryProfile(
        workflowId = "workflow_a_voice_only",
        baseBytes = mebibytes(64),
        kvBytes = mebibytes(32),
        estimationSource = "workflow_heuristic_v1",
      )
    val decision =
      GraphPilotMemoryDecision(
        type = GraphPilotMemoryDecisionType.DEGRADE,
        requiredBytes = mebibytes(96),
        effectiveRequiredBytes = mebibytes(64),
        appliedActions =
          listOf(
            GraphPilotDegradationOption(
              id = "reduce_responder_max_tokens",
              freedBytes = mebibytes(32),
              addedLatencyMs = 0.0,
              qualityLoss = 0.1,
              responderMaxTokens = 24,
            )
          ),
        totalAddedLatencyMs = 0.0,
        totalQualityLoss = 0.1,
        effectiveResponderMaxTokens = 24,
        reason = "Request exceeded usable memory budget and was degraded to fit.",
      )
    val observation =
      GraphPilotMemoryObservation(
        workflowId = "workflow_a_voice_only",
        planId = plan.plan_id,
        profile = profile,
        budget = GraphPilotMemoryBudget(totalBytes = mebibytes(128), marginBytes = mebibytes(16)),
        activeReservedBytesAtAdmission = mebibytes(24),
        reservedBytesAfterAdmission = mebibytes(64),
        decision = decision,
      )

    val message = buildMemoryAdmissionLogMessage(plan, observation)

    assertThat(message).contains("workflow=workflow_a_voice_only")
    assertThat(message).contains("memory_source=workflow_heuristic_v1")
    assertThat(message).contains("active_reserved_bytes=25165824")
    assertThat(message).contains("decision=DEGRADE")
    assertThat(message).contains("total_added_latency_ms=0.0")
    assertThat(message).contains("total_quality_loss=0.1")
    assertThat(message).contains("applied_actions=reduce_responder_max_tokens")
    assertThat(message).contains("effective_responder_max_tokens=24")
  }

  @Test
  fun buildStreamFallbackLogMessageIncludesFallbackReason() {
    val plan =
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
      )

    val message =
      buildStreamFallbackLogMessage(
        workflowId = "workflow_a_voice_only",
        plan = plan,
        fromStageId = "asr.primary",
        toStageId = "planner.primary",
        fallbackReason = "missing_chunk_size_for_chunk_stream_edge",
      )

    assertThat(message).contains("GRAPHPILOT_STREAM_FALLBACK")
    assertThat(message).contains("workflow=workflow_a_voice_only")
    assertThat(message).contains("from_stage=asr.primary")
    assertThat(message).contains("to_stage=planner.primary")
    assertThat(message).contains("fallback_reason=missing_chunk_size_for_chunk_stream_edge")
  }

  @Test(expected = IllegalStateException::class)
  fun buildAdmissionLogMessageFailsFastWhenPredictedStreamMetricsMissing() {
    val plan =
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
      )

    buildAdmissionLogMessage(
      "workflow_a_voice_only",
      plan,
      GraphPilotQueueObservation(
        requestId = 1L,
        workflowId = "workflow_a_voice_only",
        planId = plan.plan_id,
        queueDepthAtAdmission = 0,
        predictedQueueDelayMs = 0.0,
        predictedStreamMakespanMs = 0.0,
        predictedDeadlineMiss = false,
        deadlineMs = null,
        admittedAtMs = 1000L,
        admissionDecision = GraphPilotAdmissionDecision.ADMIT,
      ),
    )
  }

  @Test
  fun planDeclaresTokenStreamEdgeForResponderToTts() {
    val plan =
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
        stream_edges =
          listOf(
            GraphPilotStreamEdge(from = "planner.primary", to = "responder.primary", streamMode = "full"),
            GraphPilotStreamEdge(from = "responder.primary", to = "tts.primary", streamMode = "token"),
          ),
      )

    assertThat(plan.hasTokenStreamEdge("responder.primary", "tts.primary")).isTrue()
    assertThat(plan.hasTokenStreamEdge("planner.primary", "responder.primary")).isFalse()
  }
}
