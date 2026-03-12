package com.qidk.fastvlm.core.graphpilot

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.speech.tts.TextToSpeech
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.qidk.fastvlm.core.bridge.FastVlmNativeBridge
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.metrics.MetricsStore
import com.qidk.fastvlm.core.model.BackendTarget
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.model.VqaMetrics
import com.qidk.fastvlm.core.model.VqaRequest
import com.qidk.fastvlm.core.speech.AndroidTtsSpeaker
import com.qidk.fastvlm.core.speech.StreamingTtsCoordinator
import com.qidk.fastvlm.core.speech.WhisperSttEngine
import com.qidk.fastvlm.core.text.LiteRtLmTextStageAdapter
import com.qidk.fastvlm.core.text.TextStageModelSpec
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull

private const val TAG = "GraphPilotCoordinator"

internal fun buildAdmissionLogMessage(
  workflowId: String,
  plan: GraphPilotExecutionPlan,
  observation: GraphPilotQueueObservation,
): String {
  val predictedCost = plan.requirePredictedStreamCost()
  return "GRAPHPILOT_ADMISSION workflow=$workflowId plan_id=${plan.plan_id} state_id=${plan.state_id} " +
    "request_id=${observation.requestId} admission_ms=${observation.admittedAtMs} " +
    "queue_depth_at_admission=${observation.queueDepthAtAdmission} " +
    "predicted_queue_delay_at_admission_ms=${observation.predictedQueueDelayMs} " +
    "predicted_deadline_miss=${observation.predictedDeadlineMiss} " +
    "predicted_stream_makespan_ms=${predictedCost.streamMakespanMs} " +
    "predicted_p95_e2e_ms=${predictedCost.p95E2eMs ?: -1.0} " +
    "predicted_p95_ttfs_ms=${predictedCost.p95TtfsMs ?: -1.0} " +
    "predicted_avg_energy_mj=${predictedCost.avgEnergyMj ?: -1.0} " +
    "predicted_copy_bytes=${predictedCost.copyBytes ?: -1L} " +
    "predicted_quality_loss=${predictedCost.qualityLoss ?: -1.0} " +
    "predicted_p95_queue_ms=${predictedCost.p95QueueDelayMs} " +
    "predicted_deadline_miss_rate=${predictedCost.deadlineMissRate}"
}

internal fun buildMemoryAdmissionLogMessage(
  plan: GraphPilotExecutionPlan,
  observation: GraphPilotMemoryObservation,
): String {
  return "GRAPHPILOT_MEMORY_ADMISSION workflow=${observation.workflowId} plan_id=${plan.plan_id} state_id=${plan.state_id} " +
    "memory_source=${observation.profile.estimationSource} " +
    "required_bytes=${observation.decision.requiredBytes} " +
    "effective_required_bytes=${observation.decision.effectiveRequiredBytes} " +
    "active_reserved_bytes=${observation.activeReservedBytesAtAdmission} " +
    "reserved_after_admission_bytes=${observation.reservedBytesAfterAdmission} " +
    "decision=${observation.decision.type} " +
    "reason=${observation.reason.replace(' ', '_')} " +
    "total_added_latency_ms=${observation.decision.totalAddedLatencyMs} " +
    "total_quality_loss=${observation.decision.totalQualityLoss} " +
    "applied_actions=${observation.decision.appliedActions.joinToString(",") { it.id }.ifBlank { "none" }} " +
    "effective_responder_max_tokens=${observation.decision.effectiveResponderMaxTokens ?: -1} " +
    "effective_retrieval_top_k=${observation.decision.effectiveRetrievalTopK ?: -1}"
}

internal fun buildStreamFallbackLogMessage(
  workflowId: String,
  plan: GraphPilotExecutionPlan,
  fromStageId: String,
  toStageId: String,
  fallbackReason: String,
): String {
  return "GRAPHPILOT_STREAM_FALLBACK workflow=$workflowId plan_id=${plan.plan_id} state_id=${plan.state_id} " +
    "from_stage=$fromStageId to_stage=$toStageId fallback_reason=$fallbackReason " +
    "remediation=regenerate_candidate_plan_registry_with_explicit_chunk_sizes_or_disable_the_chunk_stream_edge"
}

private data class GraphPilotExecutionKnobs(
  val responderMaxTokens: Int = 48,
  val retrievalTopK: Int = 2,
  val vlmMaxOutputTokens: Int = 32,
)

private fun GraphPilotMemoryDecision.toExecutionKnobs(): GraphPilotExecutionKnobs {
  return GraphPilotExecutionKnobs(
    responderMaxTokens = effectiveResponderMaxTokens ?: 48,
    retrievalTopK = effectiveRetrievalTopK ?: 2,
    vlmMaxOutputTokens = 32,
  )
}

data class GraphPilotRunResult(
  val workflowId: String,
  val planId: String,
  val stateId: String,
  val requestId: Long,
  val queueDepthAtAdmission: Int,
  val predictedQueueDelayAtAdmissionMs: Double,
  val predictedDeadlineMissAtAdmission: Boolean,
  val queueWaitMs: Long,
  val predictedStreamMakespanMs: Double?,
  val predictedP95QueueDelayMs: Double?,
  val predictedDeadlineMissRate: Double?,
  val memoryDecision: String,
  val memoryProfileSource: String,
  val memoryRequiredBytes: Long,
  val memoryEffectiveRequiredBytes: Long,
  val memoryActiveReservedBytesAtAdmission: Long,
  val memoryAppliedActions: List<String>,
  val effectiveResponderMaxTokens: Int?,
  val effectiveRetrievalTopK: Int?,
  val transcript: String,
  val plannerText: String,
  val vlmText: String?,
  val retrievalText: String?,
  val finalResponse: String,
  val stageBackends: Map<String, String>,
  val stageTimingsMs: Map<String, Long>,
  val totalMs: Long,
  val ttftMs: Long?,
  val ttsFirstChunkQueuedMs: Long?,
  val ttsFirstAudioMs: Long?,
  val vlmPrefillMs: Long,
  val vlmDecodeMs: Long,
)

class GraphPilotCoordinator(
  private val context: Context,
  private val runtimeScheduler: GraphPilotRuntimeScheduler = GraphPilotRuntimeScheduler(),
  private val memoryBudget: GraphPilotMemoryBudget = GraphPilotMemoryBudget(totalBytes = mebibytes(512), marginBytes = mebibytes(64)),
  private val memoryController: GraphPilotMemoryAdmissionController = GraphPilotMemoryAdmissionController(latencyPenalty = 0.01),
  private val memoryProfileEstimator: GraphPilotMemoryProfileEstimator = GraphPilotMemoryProfileEstimator(),
  private val thermalPlanBank: GraphPilotThermalPlanBank = GraphPilotThermalPlanBank(),
  private val thermalSlowdownProvider: suspend () -> Double = { 1.0 },
) : AutoCloseable {
  private val json = JsonCodec.instance
  private val planStore = GraphPilotPlanStore()
  private val memoryManager = GraphPilotRuntimeMemoryManager(budget = memoryBudget, controller = memoryController)
  private val stt = WhisperSttEngine(context)
  private val textStages = LiteRtLmTextStageAdapter(context)
  private val bridge = FastVlmNativeBridge(context, MetricsStore(context, json))
  private val retrieval = GraphPilotRetrievalBridge(context)
  private val tts = AndroidTtsSpeaker(context)

  private data class GraphPilotStreamedSpeechResult(
    val responderText: String,
    val ttftMs: Long?,
    val ttsFirstChunkQueuedMs: Long?,
    val ttsFirstAudioMs: Long?,
  )

  private suspend fun <T> runWithMemoryAdmission(
    workflowId: String,
    plan: GraphPilotExecutionPlan,
    block: suspend (GraphPilotMemoryObservation) -> T,
  ): T {
    val profile = memoryProfileEstimator.estimate(workflowId, plan.predictedCost)
    return try {
      memoryManager.runWithAdmission(workflowId, plan.plan_id, profile) { observation ->
        Log.i(TAG, buildMemoryAdmissionLogMessage(plan, observation))
        block(observation)
      }
    } catch (e: GraphPilotMemoryAdmissionException) {
      Log.i(TAG, buildMemoryAdmissionLogMessage(plan, e.observation))
      throw e
    }
  }

  private suspend fun resolveStateId(requestedStateId: String): String {
    if (requestedStateId != "auto") {
      return requestedStateId
    }
    val activeReservedBytes = memoryManager.activeReservedBytes()
    val freeBytes = (memoryBudget.usableBytes - activeReservedBytes).coerceAtLeast(0L)
    return thermalPlanBank.selectState(
      slowdownFactor = thermalSlowdownProvider(),
      freeBytes = freeBytes,
      usableBytes = memoryBudget.usableBytes,
    ).stateId
  }

  suspend fun runWorkflowAVoiceOnly(stateId: String = "cool"): GraphPilotRunResult {
    val workflowId = "workflow_a_voice_only"
    val effectiveStateId = resolveStateId(stateId)
    val plan = planStore.loadPlan(workflowId, effectiveStateId)
    val predictedCost = plan.requirePredictedStreamCost()
    return runWithMemoryAdmission(workflowId, plan) { memoryObservation ->
      val knobs = memoryObservation.decision.toExecutionKnobs()
      runtimeScheduler.runWithAdmission(workflowId, plan) { observation ->
        val totalStartMs = System.currentTimeMillis()
        logPlanAdmission(workflowId, plan, observation)
        val wav = resolveWavFixture()
        val stageBackends = mutableStageBackends(plan, "asr.primary", "planner.primary", "responder.primary", "tts.primary")
        val timings = linkedMapOf<String, Long>()

        requireFixedBackend(plan, "asr.primary", "cpu", "WhisperSttEngine is CPU-only in GraphPilotCoordinator.")
        requireFixedBackend(plan, "tts.primary", "cpu", "AndroidTtsSpeaker is CPU-only in GraphPilotCoordinator.")
        val asrPlanner =
          runAsrPlannerWithOptionalChunking(
            workflowId = workflowId,
            plan = plan,
            wav = wav,
            plannerBackend = plan.textBackend("planner.primary"),
            timings = timings,
            totalStartMs = totalStartMs,
          )
        val transcript = asrPlanner.transcript
        val plannerText = asrPlanner.plannerText
        val streamedSpeech =
          runResponderWithOptionalStreamingToTts(
            plan = plan,
            transcript = transcript,
            plannerText = plannerText,
            vlmText = null,
            retrievalText = null,
            backend = plan.textBackend("responder.primary"),
            timings = timings,
            totalStartMs = totalStartMs,
            maxOutputTokens = knobs.responderMaxTokens,
          )
        val totalMs = System.currentTimeMillis() - totalStartMs

        Log.i(
          TAG,
          "GRAPHPILOT_METRICS workflow=$workflowId plan_id=${plan.plan_id} state_id=$effectiveStateId " +
            "request_id=${observation.requestId} queue_depth_at_admission=${observation.queueDepthAtAdmission} " +
            "queue_wait_ms=${observation.queueWaitMs} memory_decision=${memoryObservation.decision.type} " +
            "memory_effective_required_bytes=${memoryObservation.decision.effectiveRequiredBytes} " +
            "stage_backends=${encodeMap(stageBackends)} stage_timings_ms=${encodeLongMap(timings)} total_ms=$totalMs " +
            "planner_first_partial_ms=${asrPlanner.plannerFirstPartialOffsetMs ?: -1} asr_chunk_count=${asrPlanner.asrChunkCount} " +
            "ttft_ms=${streamedSpeech.ttftMs ?: -1} tts_first_chunk_queued_ms=${streamedSpeech.ttsFirstChunkQueuedMs ?: -1} " +
            "tts_first_audio_ms=${streamedSpeech.ttsFirstAudioMs ?: -1} vlm_prefill_ms=0 vlm_decode_ms=0",
        )

        GraphPilotRunResult(
          workflowId = workflowId,
          planId = plan.plan_id,
          stateId = effectiveStateId,
          requestId = observation.requestId,
          queueDepthAtAdmission = observation.queueDepthAtAdmission,
          predictedQueueDelayAtAdmissionMs = observation.predictedQueueDelayMs,
          predictedDeadlineMissAtAdmission = observation.predictedDeadlineMiss,
          queueWaitMs = observation.queueWaitMs,
          predictedStreamMakespanMs = predictedCost.streamMakespanMs,
          predictedP95QueueDelayMs = predictedCost.p95QueueDelayMs,
          predictedDeadlineMissRate = predictedCost.deadlineMissRate,
          memoryDecision = memoryObservation.decision.type.name,
          memoryProfileSource = memoryObservation.profile.estimationSource,
          memoryRequiredBytes = memoryObservation.decision.requiredBytes,
          memoryEffectiveRequiredBytes = memoryObservation.decision.effectiveRequiredBytes,
          memoryActiveReservedBytesAtAdmission = memoryObservation.activeReservedBytesAtAdmission,
          memoryAppliedActions = memoryObservation.decision.appliedActions.map { it.id },
          effectiveResponderMaxTokens = memoryObservation.decision.effectiveResponderMaxTokens,
          effectiveRetrievalTopK = memoryObservation.decision.effectiveRetrievalTopK,
          transcript = transcript,
          plannerText = plannerText,
          vlmText = null,
          retrievalText = null,
          finalResponse = streamedSpeech.responderText,
          stageBackends = stageBackends,
          stageTimingsMs = timings,
          totalMs = totalMs,
          ttftMs = streamedSpeech.ttftMs,
          ttsFirstChunkQueuedMs = streamedSpeech.ttsFirstChunkQueuedMs,
          ttsFirstAudioMs = streamedSpeech.ttsFirstAudioMs,
          vlmPrefillMs = 0L,
          vlmDecodeMs = 0L,
        )
      }.second
    }
  }

  suspend fun runWorkflowBVoiceVision(stateId: String = "cool"): GraphPilotRunResult {
    val workflowId = "workflow_b_voice_vision"
    val effectiveStateId = resolveStateId(stateId)
    val plan = planStore.loadPlan(workflowId, effectiveStateId)
    val predictedCost = plan.requirePredictedStreamCost()
    return runWithMemoryAdmission(workflowId, plan) { memoryObservation ->
      val knobs = memoryObservation.decision.toExecutionKnobs()
      runtimeScheduler.runWithAdmission(workflowId, plan) { observation ->
        val totalStartMs = System.currentTimeMillis()
        logPlanAdmission(workflowId, plan, observation)
        val wav = resolveWavFixture()
        val image = createSyntheticImage()
        val stageBackends =
          mutableStageBackends(
            plan,
            "asr.primary",
            "planner.primary",
            "vlm.fastvlm.primary",
            "responder.primary",
            "tts.primary",
          )
        val timings = linkedMapOf<String, Long>()

        requireFixedBackend(plan, "asr.primary", "cpu", "WhisperSttEngine is CPU-only in GraphPilotCoordinator.")
        requireFixedBackend(plan, "tts.primary", "cpu", "AndroidTtsSpeaker is CPU-only in GraphPilotCoordinator.")
        val asrPlanner =
          runAsrPlannerWithOptionalChunking(
            workflowId = workflowId,
            plan = plan,
            wav = wav,
            plannerBackend = plan.textBackend("planner.primary"),
            timings = timings,
            totalStartMs = totalStartMs,
          )
        val transcript = asrPlanner.transcript
        val plannerText = asrPlanner.plannerText
        val plannedVlmBackend = plan.vlmBackendTarget()
        val (vlmText, metrics, ttftMs) = runVlm(transcript, image, plannedVlmBackend, timings, maxOutputTokens = knobs.vlmMaxOutputTokens)
        val actualVlmBackend = metrics.backend_status.backend_config_actual.name.lowercase()
        check(actualVlmBackend == plannedVlmBackend.name.lowercase()) {
          "Workflow B plan '${plan.plan_id}' expected FastVLM backend='${plannedVlmBackend.name.lowercase()}', " +
            "but bridge reported backend='${metrics.backend_status.backend_config_actual}'."
        }
        stageBackends["vlm.fastvlm.primary"] = actualVlmBackend
        val streamedSpeech =
          runResponderWithOptionalStreamingToTts(
            plan = plan,
            transcript = transcript,
            plannerText = plannerText,
            vlmText = vlmText,
            retrievalText = null,
            backend = plan.textBackend("responder.primary"),
            timings = timings,
            totalStartMs = totalStartMs,
            maxOutputTokens = knobs.responderMaxTokens,
          )
        val totalMs = System.currentTimeMillis() - totalStartMs

        Log.i(
          TAG,
          "GRAPHPILOT_METRICS workflow=$workflowId plan_id=${plan.plan_id} state_id=$effectiveStateId " +
            "request_id=${observation.requestId} queue_depth_at_admission=${observation.queueDepthAtAdmission} " +
            "queue_wait_ms=${observation.queueWaitMs} memory_decision=${memoryObservation.decision.type} " +
            "memory_effective_required_bytes=${memoryObservation.decision.effectiveRequiredBytes} " +
            "stage_backends=${encodeMap(stageBackends)} stage_timings_ms=${encodeLongMap(timings)} total_ms=$totalMs " +
            "planner_first_partial_ms=${asrPlanner.plannerFirstPartialOffsetMs ?: -1} asr_chunk_count=${asrPlanner.asrChunkCount} " +
            "ttft_ms=${ttftMs ?: streamedSpeech.ttftMs ?: -1} tts_first_chunk_queued_ms=${streamedSpeech.ttsFirstChunkQueuedMs ?: -1} " +
            "tts_first_audio_ms=${streamedSpeech.ttsFirstAudioMs ?: -1} " +
            "vlm_prefill_ms=${metrics.stage_timings.prefill_ms} vlm_decode_ms=${metrics.stage_timings.decode_ms}",
        )

        GraphPilotRunResult(
          workflowId = workflowId,
          planId = plan.plan_id,
          stateId = effectiveStateId,
          requestId = observation.requestId,
          queueDepthAtAdmission = observation.queueDepthAtAdmission,
          predictedQueueDelayAtAdmissionMs = observation.predictedQueueDelayMs,
          predictedDeadlineMissAtAdmission = observation.predictedDeadlineMiss,
          queueWaitMs = observation.queueWaitMs,
          predictedStreamMakespanMs = predictedCost.streamMakespanMs,
          predictedP95QueueDelayMs = predictedCost.p95QueueDelayMs,
          predictedDeadlineMissRate = predictedCost.deadlineMissRate,
          memoryDecision = memoryObservation.decision.type.name,
          memoryProfileSource = memoryObservation.profile.estimationSource,
          memoryRequiredBytes = memoryObservation.decision.requiredBytes,
          memoryEffectiveRequiredBytes = memoryObservation.decision.effectiveRequiredBytes,
          memoryActiveReservedBytesAtAdmission = memoryObservation.activeReservedBytesAtAdmission,
          memoryAppliedActions = memoryObservation.decision.appliedActions.map { it.id },
          effectiveResponderMaxTokens = memoryObservation.decision.effectiveResponderMaxTokens,
          effectiveRetrievalTopK = memoryObservation.decision.effectiveRetrievalTopK,
          transcript = transcript,
          plannerText = plannerText,
          vlmText = vlmText,
          retrievalText = null,
          finalResponse = streamedSpeech.responderText,
          stageBackends = stageBackends,
          stageTimingsMs = timings,
          totalMs = totalMs,
          ttftMs = ttftMs ?: streamedSpeech.ttftMs,
          ttsFirstChunkQueuedMs = streamedSpeech.ttsFirstChunkQueuedMs,
          ttsFirstAudioMs = streamedSpeech.ttsFirstAudioMs,
          vlmPrefillMs = metrics.stage_timings.prefill_ms,
          vlmDecodeMs = metrics.stage_timings.decode_ms,
        )
      }.second
    }
  }

  suspend fun runWorkflowCVoiceVisionRetrieval(stateId: String = "cool"): GraphPilotRunResult = coroutineScope {
    val workflowId = "workflow_c_voice_vision_retrieval"
    val effectiveStateId = resolveStateId(stateId)
    val plan = planStore.loadPlan(workflowId, effectiveStateId)
    val predictedCost = plan.requirePredictedStreamCost()
    runWithMemoryAdmission(workflowId, plan) { memoryObservation ->
      val knobs = memoryObservation.decision.toExecutionKnobs()
      runtimeScheduler.runWithAdmission(workflowId, plan) { observation ->
      val totalStartMs = System.currentTimeMillis()
      logPlanAdmission(workflowId, plan, observation)
      val wav = resolveWavFixture()
      val image = createSyntheticImage()
      val stageBackends =
        mutableStageBackends(
          plan,
          "asr.primary",
          "planner.primary",
          "vlm.fastvlm.primary",
          "retrieval.embedder.primary",
          "responder.primary",
          "tts.primary",
        )
      val timings = linkedMapOf<String, Long>()

      requireFixedBackend(plan, "asr.primary", "cpu", "WhisperSttEngine is CPU-only in GraphPilotCoordinator.")
      requireFixedBackend(plan, "tts.primary", "cpu", "AndroidTtsSpeaker is CPU-only in GraphPilotCoordinator.")
      val asrPlanner =
        runAsrPlannerWithOptionalChunking(
          workflowId = workflowId,
          plan = plan,
          wav = wav,
          plannerBackend = plan.textBackend("planner.primary"),
          timings = timings,
          totalStartMs = totalStartMs,
        )
      val transcript = asrPlanner.transcript
      val plannerText = asrPlanner.plannerText
      val retrievalQuery = buildString {
        append(transcript)
        append("\nPlanner summary: ")
        append(plannerText)
      }

      val plannedVlmBackend = plan.vlmBackendTarget()
      val vlmDeferred = async { runVlm(transcript, image, plannedVlmBackend, timings, maxOutputTokens = knobs.vlmMaxOutputTokens) }
      val retrievalBackend = plan.retrievalBackend()
      val retrievalDeferred = async { runRetrieval(retrievalQuery, retrievalBackend, timings, topK = knobs.retrievalTopK) }

      val (vlmText, metrics, ttftMs) = vlmDeferred.await()
      val actualVlmBackend = metrics.backend_status.backend_config_actual.name.lowercase()
      check(actualVlmBackend == plannedVlmBackend.name.lowercase()) {
        "Workflow C plan '${plan.plan_id}' expected FastVLM backend='${plannedVlmBackend.name.lowercase()}', " +
          "but bridge reported backend='${metrics.backend_status.backend_config_actual}'."
      }
      stageBackends["vlm.fastvlm.primary"] = actualVlmBackend

      val retrievalText = retrievalDeferred.await()
      stageBackends["retrieval.embedder.primary"] = retrievalBackend.name.lowercase()
      val streamedSpeech =
        runResponderWithOptionalStreamingToTts(
          plan = plan,
          transcript = transcript,
          plannerText = plannerText,
          vlmText = vlmText,
          retrievalText = retrievalText,
          backend = plan.textBackend("responder.primary"),
          timings = timings,
          totalStartMs = totalStartMs,
          maxOutputTokens = knobs.responderMaxTokens,
        )
      val totalMs = System.currentTimeMillis() - totalStartMs

      Log.i(
        TAG,
        "GRAPHPILOT_METRICS workflow=$workflowId plan_id=${plan.plan_id} state_id=$effectiveStateId " +
          "request_id=${observation.requestId} queue_depth_at_admission=${observation.queueDepthAtAdmission} " +
          "queue_wait_ms=${observation.queueWaitMs} memory_decision=${memoryObservation.decision.type} " +
          "memory_effective_required_bytes=${memoryObservation.decision.effectiveRequiredBytes} " +
          "stage_backends=${encodeMap(stageBackends)} " +
          "stage_timings_ms=${encodeLongMap(timings)} total_ms=$totalMs " +
          "planner_first_partial_ms=${asrPlanner.plannerFirstPartialOffsetMs ?: -1} asr_chunk_count=${asrPlanner.asrChunkCount} " +
          "ttft_ms=${ttftMs ?: streamedSpeech.ttftMs ?: -1} tts_first_chunk_queued_ms=${streamedSpeech.ttsFirstChunkQueuedMs ?: -1} " +
          "tts_first_audio_ms=${streamedSpeech.ttsFirstAudioMs ?: -1} " +
          "vlm_prefill_ms=${metrics.stage_timings.prefill_ms} vlm_decode_ms=${metrics.stage_timings.decode_ms}",
      )

      GraphPilotRunResult(
        workflowId = workflowId,
        planId = plan.plan_id,
        stateId = effectiveStateId,
        requestId = observation.requestId,
        queueDepthAtAdmission = observation.queueDepthAtAdmission,
        predictedQueueDelayAtAdmissionMs = observation.predictedQueueDelayMs,
        predictedDeadlineMissAtAdmission = observation.predictedDeadlineMiss,
        queueWaitMs = observation.queueWaitMs,
        predictedStreamMakespanMs = predictedCost.streamMakespanMs,
        predictedP95QueueDelayMs = predictedCost.p95QueueDelayMs,
        predictedDeadlineMissRate = predictedCost.deadlineMissRate,
        memoryDecision = memoryObservation.decision.type.name,
        memoryProfileSource = memoryObservation.profile.estimationSource,
        memoryRequiredBytes = memoryObservation.decision.requiredBytes,
        memoryEffectiveRequiredBytes = memoryObservation.decision.effectiveRequiredBytes,
        memoryActiveReservedBytesAtAdmission = memoryObservation.activeReservedBytesAtAdmission,
        memoryAppliedActions = memoryObservation.decision.appliedActions.map { it.id },
        effectiveResponderMaxTokens = memoryObservation.decision.effectiveResponderMaxTokens,
        effectiveRetrievalTopK = memoryObservation.decision.effectiveRetrievalTopK,
        transcript = transcript,
        plannerText = plannerText,
        vlmText = vlmText,
        retrievalText = retrievalText,
        finalResponse = streamedSpeech.responderText,
        stageBackends = stageBackends,
        stageTimingsMs = timings,
        totalMs = totalMs,
        ttftMs = ttftMs ?: streamedSpeech.ttftMs,
        ttsFirstChunkQueuedMs = streamedSpeech.ttsFirstChunkQueuedMs,
        ttsFirstAudioMs = streamedSpeech.ttsFirstAudioMs,
        vlmPrefillMs = metrics.stage_timings.prefill_ms,
        vlmDecodeMs = metrics.stage_timings.decode_ms,
      )
      }.second
    }
  }

  override fun close() {
    runBlocking {
      runCatching { tts.release() }
      runCatching { stt.release() }
      runCatching { textStages.close() }
      runCatching { bridge.close() }
    }
  }

  private suspend fun prepareAsr() {
    val whisperConfigPath = stageConfig("whisper_stt.json")
    stt.ensureInitialized(whisperConfigPath)
    stt.warmup()
  }

  private suspend fun runAsrPlannerWithOptionalChunking(
    workflowId: String,
    plan: GraphPilotExecutionPlan,
    wav: File,
    plannerBackend: Backend,
    timings: MutableMap<String, Long>,
    totalStartMs: Long,
  ): GraphPilotAsrPlannerPipelineResult {
    prepareAsr()
    val pcm = readWavPcm16Mono16k(wav)
    val clip = pcm.copyOfRange(0, minOf(pcm.size, 24_000))
    val chunkSizeMs =
      if (plan.hasChunkStreamEdge("asr.primary", "planner.primary")) {
        val configuredChunkSize = plan.chunkSizeOrNull("asr.primary")
        if (configuredChunkSize == null) {
          Log.w(
            TAG,
            buildStreamFallbackLogMessage(
              workflowId = workflowId,
              plan = plan,
              fromStageId = "asr.primary",
              toStageId = "planner.primary",
              fallbackReason = "missing_chunk_size_for_chunk_stream_edge",
            ),
          )
        }
        configuredChunkSize
      } else {
        null
      }
    val result =
      runAsrPlannerPipeline(
        audioData = clip,
        chunkSizeMs = chunkSizeMs,
        totalStartMs = totalStartMs,
        transcribe = { chunk -> stt.transcribe(chunk) },
        plan = { transcript -> generatePlannerText(transcript, plannerBackend) },
      )
    timings["asr.primary"] = result.asrElapsedMs
    timings["planner.primary"] = result.plannerElapsedMs
    return result
  }

  private suspend fun runAsr(wav: File, timings: MutableMap<String, Long>): String {
    prepareAsr()
    val pcm = readWavPcm16Mono16k(wav)
    val clip = pcm.copyOfRange(0, minOf(pcm.size, 24_000))
    val startMs = System.currentTimeMillis()
    val transcript = stt.transcribe(clip).trim()
    timings["asr.primary"] = System.currentTimeMillis() - startMs
    check(transcript.isNotEmpty()) { "ASR produced an empty transcript." }
    return transcript
  }

  private suspend fun generatePlannerText(
    transcript: String,
    backend: Backend,
  ): String {
    val spec = TextStageModelSpec.planner(backend)
    textStages.ensureInitialized(spec)
    val plannerText =
      textStages.generate(
        prompt = "User request transcript: $transcript\nReturn a compact plan in one short sentence.",
        maxOutputTokens = 32,
      )
    check(plannerText.isNotBlank()) { "Planner produced an empty plan." }
    return plannerText
  }

  private suspend fun runPlanner(transcript: String, timings: MutableMap<String, Long>): String {
    return runPlanner(transcript, Backend.CPU, timings)
  }

  private suspend fun runPlanner(
    transcript: String,
    backend: Backend,
    timings: MutableMap<String, Long>,
  ): String {
    val startMs = System.currentTimeMillis()
    val plannerText = generatePlannerText(transcript, backend)
    timings["planner.primary"] = System.currentTimeMillis() - startMs
    return plannerText
  }

  private suspend fun runResponderWithOptionalStreamingToTts(
    plan: GraphPilotExecutionPlan,
    transcript: String,
    plannerText: String,
    vlmText: String?,
    retrievalText: String?,
    backend: Backend,
    timings: MutableMap<String, Long>,
    totalStartMs: Long,
    maxOutputTokens: Int = 48,
  ): GraphPilotStreamedSpeechResult {
    val prompt = buildResponderPrompt(transcript, plannerText, vlmText, retrievalText)
    if (!plan.hasTokenStreamEdge("responder.primary", "tts.primary")) {
      val responderText = runResponder(prompt, backend, timings, maxOutputTokens)
      val ttsAudioMs = runTts(responderText, timings, totalStartMs)
      return GraphPilotStreamedSpeechResult(
        responderText = responderText,
        ttftMs = null,
        ttsFirstChunkQueuedMs = null,
        ttsFirstAudioMs = ttsAudioMs,
      )
    }
    return runResponderStreamingToTts(prompt, backend, timings, totalStartMs, maxOutputTokens)
  }

  private fun buildResponderPrompt(
    transcript: String,
    plannerText: String,
    vlmText: String?,
    retrievalText: String?,
  ): String {
    return buildString {
      append("User transcript: ")
      append(transcript)
      append("\nPlanner summary: ")
      append(plannerText)
      if (!vlmText.isNullOrBlank()) {
        append("\nVision output: ")
        append(vlmText)
      }
      if (!retrievalText.isNullOrBlank()) {
        append("\nRetrieval evidence: ")
        append(retrievalText)
      }
      append("\nRespond directly in one or two short sentences.")
    }
  }

  private suspend fun runResponder(
    prompt: String,
    backend: Backend,
    timings: MutableMap<String, Long>,
    maxOutputTokens: Int = 48,
  ): String {
    val spec = TextStageModelSpec.responder(backend)
    val startMs = System.currentTimeMillis()
    textStages.ensureInitialized(spec)
    val responderText = textStages.generate(prompt, maxOutputTokens = maxOutputTokens)
    timings["responder.primary"] = System.currentTimeMillis() - startMs
    check(responderText.isNotBlank()) { "Responder produced an empty response." }
    return responderText
  }

  private suspend fun runResponderStreamingToTts(
    prompt: String,
    backend: Backend,
    timings: MutableMap<String, Long>,
    totalStartMs: Long,
    maxOutputTokens: Int,
  ): GraphPilotStreamedSpeechResult = coroutineScope {
    val spec = TextStageModelSpec.responder(backend)
    textStages.ensureInitialized(spec)
    tts.ensureInitialized()

    var firstChunkQueuedAtMsValue: Long? = null
    var firstAudioAtMsValue: Long? = null
    val firstChunkQueuedAtMs = CompletableDeferred<Long>()
    val firstAudioAtMs = CompletableDeferred<Long>()
    val speechIdle = CompletableDeferred<Unit>()
    val streamingTts =
      StreamingTtsCoordinator(
        ttsSpeaker = tts,
        scope = this,
        onFirstChunkQueued = { queuedAtMs ->
          firstChunkQueuedAtMsValue = queuedAtMs
          if (!firstChunkQueuedAtMs.isCompleted) {
            firstChunkQueuedAtMs.complete(queuedAtMs)
          }
        },
        onFirstAudioStart = { startedAtMs ->
          firstAudioAtMsValue = startedAtMs
          if (!firstAudioAtMs.isCompleted) {
            firstAudioAtMs.complete(startedAtMs)
          }
        },
        onSpeechActive = {},
        onSpeechIdle = {
          if (!speechIdle.isCompleted) {
            speechIdle.complete(Unit)
          }
        },
      )
    val speechStreamer =
      object : GraphPilotSpeechStreamer {
        override fun startSession() {
          streamingTts.startSession()
        }

        override fun appendDelta(delta: String) {
          streamingTts.appendDelta(delta)
        }

        override fun finishStreaming() {
          streamingTts.finishStreaming()
        }

        override fun abortStreaming() {
          streamingTts.abortStreaming()
          if (!speechIdle.isCompleted) {
            speechIdle.complete(Unit)
          }
        }
      }

    val responderStartMs = System.currentTimeMillis()
    val response =
      GraphPilotStreamingResponder()
        .generateAndStream(
          prompt = prompt,
          maxOutputTokens = maxOutputTokens,
          generator = textStages,
          speechStreamer = speechStreamer,
        )
    timings["responder.primary"] = System.currentTimeMillis() - responderStartMs
    withTimeout(30_000L) {
      speechIdle.await()
    }
    timings["tts.primary"] = maxOf(0L, System.currentTimeMillis() - responderStartMs)
    GraphPilotStreamedSpeechResult(
      responderText = response.text,
      ttftMs = response.firstTokenOffsetMs,
      ttsFirstChunkQueuedMs = firstChunkQueuedAtMsValue?.minus(totalStartMs),
      ttsFirstAudioMs = firstAudioAtMsValue?.minus(totalStartMs),
    )
  }

  private suspend fun runRetrieval(
    query: String,
    backend: Backend,
    timings: MutableMap<String, Long>,
    topK: Int = 2,
  ): String {
    val startMs = System.currentTimeMillis()
    val payload = retrieval.retrieve(query, topK = topK, backend = backend)
    timings["retrieval.embedder.primary"] = System.currentTimeMillis() - startMs
    check(payload.hits.isNotEmpty()) { "Retrieval returned zero hits." }
    return payload.hits.joinToString(separator = " || ") { hit ->
      "${hit.title}: ${hit.text}"
    }
  }

  private suspend fun runVlm(
    transcript: String,
    image: File,
    backend: BackendTarget,
    timings: MutableMap<String, Long>,
    maxOutputTokens: Int = 32,
  ): Triple<String, VqaMetrics, Long?> {
    val fastVlmConfigPath =
      when (backend) {
        BackendTarget.NPU -> stageConfig("fastvlm_phase1_npu.json")
        BackendTarget.CPU -> stageConfig("fastvlm_phase1.json")
      }
    val initResult = bridge.nativeInit(fastVlmConfigPath)
    check(initResult.ok) { "FastVLM bridge init failed: ${initResult.message}" }
    check(initResult.preferred_backend == backend) {
      "Workflow expected FastVLM ${backend.name} init, but bridge reported preferred_backend=${initResult.preferred_backend}."
    }

    val answer = StringBuilder()
    val done = CompletableDeferred<Unit>()
    var metrics: VqaMetrics? = null
    var firstTokenAtMs: Long? = null
    val requestStartMs = System.currentTimeMillis()

    val request =
      VqaRequest(
        question = transcript,
        image_path = image.absolutePath,
        preferred_backend = backend,
        allow_fallback = false,
        max_output_tokens = maxOutputTokens,
        frame_capture_ms = 0L,
        preprocess_ms = 0L,
      )

    bridge.nativeRunVqa(json.encodeToString(VqaRequest.serializer(), request)) { raw ->
      val event = json.decodeFromString(VqaEvent.serializer(), raw)
      when (event.type) {
        VqaEventType.TOKEN -> {
          if (firstTokenAtMs == null) {
            firstTokenAtMs = System.currentTimeMillis()
          }
          event.token?.let { answer.append(it) }
        }
        VqaEventType.METRICS -> metrics = event.metrics
        VqaEventType.DONE -> if (!done.isCompleted) done.complete(Unit)
        VqaEventType.ERROR -> {
          val message = event.error?.message ?: "unknown_error"
          if (!done.isCompleted) {
            done.completeExceptionally(IllegalStateException("VLM failed: $message"))
          }
        }
        else -> Unit
      }
    }

    withTimeout(30_000L) {
      done.await()
    }
    val elapsedMs = System.currentTimeMillis() - requestStartMs
    timings["vlm.fastvlm.primary"] = elapsedMs
    val vlmText = answer.toString().trim()
    check(vlmText.isNotEmpty()) { "VLM produced an empty response." }
    val metricsResult = metrics ?: error("Missing VLM metrics event.")
    return Triple(vlmText, metricsResult, firstTokenAtMs?.minus(requestStartMs))
  }

  private fun mutableStageBackends(
    plan: GraphPilotExecutionPlan,
    vararg stageIds: String,
  ): LinkedHashMap<String, String> {
    val map = linkedMapOf<String, String>()
    for (stageId in stageIds) {
      map[stageId] = plan.requireStageBackend(stageId)
    }
    return map
  }

  private fun requireFixedBackend(
    plan: GraphPilotExecutionPlan,
    stageId: String,
    expectedBackend: String,
    remediation: String,
  ) {
    val actualBackend = plan.requireStageBackend(stageId)
    check(actualBackend == expectedBackend) {
      "GraphPilot plan '${plan.plan_id}' requested ${stageId} backend='${actualBackend}', " +
        "but GraphPilotCoordinator only supports backend='${expectedBackend}' for that stage. Remediation: ${remediation}"
    }
  }

  private fun logPlanAdmission(
    workflowId: String,
    plan: GraphPilotExecutionPlan,
    observation: GraphPilotQueueObservation,
  ) {
    Log.i(TAG, buildAdmissionLogMessage(workflowId, plan, observation))
  }

  private suspend fun runTts(
    response: String,
    timings: MutableMap<String, Long>,
    totalStartMs: Long,
  ): Long? {
    tts.ensureInitialized()
    val startMs = System.currentTimeMillis()
    val firstAudioStart = CompletableDeferred<Long>()
    tts.speakChunk(
      text = response.take(180),
      queueMode = TextToSpeech.QUEUE_FLUSH,
      onAudioStart = { startedAt ->
        if (!firstAudioStart.isCompleted) {
          firstAudioStart.complete(startedAt)
        }
      },
      awaitCompletion = true,
    )
    timings["tts.primary"] = System.currentTimeMillis() - startMs
    return withTimeout(5_000L) { firstAudioStart.await() } - totalStartMs
  }

  private fun encodeMap(values: Map<String, String>): String {
    return values.entries.joinToString(",") { (key, value) -> "$key:$value" }
  }

  private fun encodeLongMap(values: Map<String, Long>): String {
    return values.entries.joinToString(",") { (key, value) -> "$key:$value" }
  }

  private fun stageConfig(assetName: String): String {
    val configDir = File(context.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      error("Failed to create config dir '${configDir.absolutePath}'")
    }
    val target = File(configDir, assetName)
    context.assets.open(assetName).use { input ->
      target.outputStream().use { output -> input.copyTo(output) }
    }
    return target.absolutePath
  }

  private fun resolveWavFixture(): File {
    val candidates =
      listOf(
        File("/data/local/tmp/graphpilot_edge/whisper/jfk.wav"),
        File("/data/local/tmp/vlm_phase1/stt_sample.wav"),
        File("/sdcard/Download/stt_sample.wav"),
      )
    return candidates.firstOrNull { it.exists() }
      ?: error(
        "Missing wav fixture. Remediation: adb push a 16kHz mono PCM wav to /data/local/tmp/graphpilot_edge/whisper/jfk.wav or /data/local/tmp/vlm_phase1/stt_sample.wav",
      )
  }

  private fun createSyntheticImage(): File {
    val out = File(context.cacheDir, "graphpilot_workflow_b_input.jpg")
    val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    canvas.drawColor(Color.rgb(244, 248, 252))
    val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(20, 32, 48)
      textSize = 42f
    }
    val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(48, 80, 120)
      textSize = 30f
    }
    canvas.drawText("GraphPilot Edge", 48f, 180f, titlePaint)
    canvas.drawText("Synthetic workflow-B image", 48f, 235f, subPaint)
    FileOutputStream(out).use { fos ->
      if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 92, fos)) {
        error("Failed to encode synthetic JPEG at '${out.absolutePath}'")
      }
    }
    bitmap.recycle()
    return out
  }

  private fun readWavPcm16Mono16k(file: File): FloatArray {
    val bytes = file.readBytes()
    require(bytes.size >= 44) { "Invalid wav: header too short (${bytes.size} bytes)" }
    require(String(bytes, 0, 4) == "RIFF") { "Invalid wav: missing RIFF header" }
    require(String(bytes, 8, 4) == "WAVE") { "Invalid wav: missing WAVE header" }

    var offset = 12
    var dataOffset = -1
    var dataSize = -1
    var numChannels = -1
    var sampleRate = -1
    var bitsPerSample = -1

    while (offset + 8 <= bytes.size) {
      val id = String(bytes, offset, 4)
      val size = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
      val payloadOffset = offset + 8
      if (id == "fmt " && size >= 16 && payloadOffset + size <= bytes.size) {
        val fmt = ByteBuffer.wrap(bytes, payloadOffset, size).order(ByteOrder.LITTLE_ENDIAN)
        val audioFormat = fmt.short.toInt() and 0xFFFF
        numChannels = fmt.short.toInt() and 0xFFFF
        sampleRate = fmt.int
        fmt.int
        fmt.short
        bitsPerSample = fmt.short.toInt() and 0xFFFF
        require(audioFormat == 1) { "Unsupported wav format=$audioFormat; require PCM(1)" }
      } else if (id == "data" && payloadOffset + size <= bytes.size) {
        dataOffset = payloadOffset
        dataSize = size
        break
      }
      offset = payloadOffset + size + (size and 1)
    }

    require(dataOffset >= 0 && dataSize > 0) { "Invalid wav: data chunk not found" }
    require(numChannels == 1) { "Unsupported channel count=$numChannels; require mono" }
    require(sampleRate == 16_000) { "Unsupported sample_rate=$sampleRate; require 16000Hz" }
    require(bitsPerSample == 16) { "Unsupported bit depth=$bitsPerSample; require 16-bit PCM" }

    val sampleCount = dataSize / 2
    val pcm = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
    val out = FloatArray(sampleCount)
    for (i in 0 until sampleCount) {
      out[i] = (pcm.short / 32767.0f).coerceIn(-1f, 1f)
    }
    return out
  }
}
