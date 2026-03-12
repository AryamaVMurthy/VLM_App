package com.qidk.fastvlm.core.graphpilot

import kotlinx.coroutines.sync.withLock

enum class GraphPilotMemoryDecisionType {
  ADMIT,
  DEGRADE,
  REJECT,
}

data class GraphPilotMemoryBudget(
  val totalBytes: Long,
  val marginBytes: Long,
) {
  init {
    require(totalBytes > 0) { "totalBytes must be > 0" }
    require(marginBytes >= 0) { "marginBytes must be >= 0" }
    require(totalBytes > marginBytes) { "totalBytes must exceed marginBytes" }
  }

  val usableBytes: Long
    get() = totalBytes - marginBytes
}

data class GraphPilotDegradationOption(
  val id: String,
  val freedBytes: Long,
  val addedLatencyMs: Double,
  val qualityLoss: Double,
  val responderMaxTokens: Int? = null,
  val retrievalTopK: Int? = null,
  val vlmVisualTokenBudget: Int? = null,
) {
  init {
    require(id.isNotBlank()) { "Degradation option id cannot be blank" }
    require(freedBytes > 0) { "freedBytes must be > 0" }
    require(addedLatencyMs >= 0.0) { "addedLatencyMs must be >= 0" }
    require(qualityLoss >= 0.0) { "qualityLoss must be >= 0" }
  }
}

data class GraphPilotRequestMemoryProfile(
  val workflowId: String,
  val baseBytes: Long,
  val kvBytes: Long,
  val estimationSource: String,
  val degradationOptions: List<GraphPilotDegradationOption> = emptyList(),
) {
  init {
    require(workflowId.isNotBlank()) { "workflowId cannot be blank" }
    require(baseBytes >= 0) { "baseBytes must be >= 0" }
    require(kvBytes >= 0) { "kvBytes must be >= 0" }
    require(estimationSource.isNotBlank()) { "estimationSource cannot be blank" }
  }

  val requiredBytes: Long
    get() = baseBytes + kvBytes
}

data class GraphPilotMemoryDecision(
  val type: GraphPilotMemoryDecisionType,
  val requiredBytes: Long,
  val effectiveRequiredBytes: Long,
  val appliedActions: List<GraphPilotDegradationOption>,
  val effectiveResponderMaxTokens: Int? = null,
  val effectiveRetrievalTopK: Int? = null,
  val effectiveVlmVisualTokenBudget: Int? = null,
  val reason: String,
)

data class GraphPilotMemoryObservation(
  val workflowId: String,
  val planId: String,
  val profile: GraphPilotRequestMemoryProfile,
  val budget: GraphPilotMemoryBudget,
  val activeReservedBytesAtAdmission: Long,
  val reservedBytesAfterAdmission: Long,
  val decision: GraphPilotMemoryDecision,
) {
  val reason: String
    get() = decision.reason
}

class GraphPilotMemoryAdmissionException(
  message: String,
  val observation: GraphPilotMemoryObservation,
) : IllegalStateException(message)

class GraphPilotMemoryAdmissionController(
  private val latencyPenalty: Double = 1.0,
) {
  fun decide(
    profile: GraphPilotRequestMemoryProfile,
    budget: GraphPilotMemoryBudget,
  ): GraphPilotMemoryDecision {
    val requiredBytes = profile.requiredBytes
    if (requiredBytes <= budget.usableBytes) {
      return GraphPilotMemoryDecision(
        type = GraphPilotMemoryDecisionType.ADMIT,
        requiredBytes = requiredBytes,
        effectiveRequiredBytes = requiredBytes,
        appliedActions = emptyList(),
        reason = "Request fits within usable memory budget.",
      )
    }

    val sortedOptions =
      profile.degradationOptions.sortedBy { option ->
        (option.qualityLoss + latencyPenalty * option.addedLatencyMs) / option.freedBytes.toDouble()
      }

    val applied = mutableListOf<GraphPilotDegradationOption>()
    var freedBytes = 0L
    var responderMaxTokens: Int? = null
    var retrievalTopK: Int? = null
    var vlmVisualTokenBudget: Int? = null

    for (option in sortedOptions) {
      applied += option
      freedBytes += option.freedBytes
      responderMaxTokens =
        when {
          option.responderMaxTokens == null -> responderMaxTokens
          responderMaxTokens == null -> option.responderMaxTokens
          else -> minOf(responderMaxTokens, option.responderMaxTokens)
        }
      retrievalTopK =
        when {
          option.retrievalTopK == null -> retrievalTopK
          retrievalTopK == null -> option.retrievalTopK
          else -> minOf(retrievalTopK, option.retrievalTopK)
        }
      vlmVisualTokenBudget =
        when {
          option.vlmVisualTokenBudget == null -> vlmVisualTokenBudget
          vlmVisualTokenBudget == null -> option.vlmVisualTokenBudget
          else -> minOf(vlmVisualTokenBudget, option.vlmVisualTokenBudget)
        }
      val effectiveRequiredBytes = (requiredBytes - freedBytes).coerceAtLeast(0L)
      if (effectiveRequiredBytes <= budget.usableBytes) {
        return GraphPilotMemoryDecision(
          type = GraphPilotMemoryDecisionType.DEGRADE,
          requiredBytes = requiredBytes,
          effectiveRequiredBytes = effectiveRequiredBytes,
          appliedActions = applied.toList(),
          effectiveResponderMaxTokens = responderMaxTokens,
          effectiveRetrievalTopK = retrievalTopK,
          effectiveVlmVisualTokenBudget = vlmVisualTokenBudget,
          reason = "Request exceeded usable memory budget and was degraded to fit.",
        )
      }
    }

    return GraphPilotMemoryDecision(
      type = GraphPilotMemoryDecisionType.REJECT,
      requiredBytes = requiredBytes,
      effectiveRequiredBytes = (requiredBytes - freedBytes).coerceAtLeast(0L),
      appliedActions = applied.toList(),
      effectiveResponderMaxTokens = responderMaxTokens,
      effectiveRetrievalTopK = retrievalTopK,
      effectiveVlmVisualTokenBudget = vlmVisualTokenBudget,
      reason = "Request exceeds memory budget even after applying all configured degradation actions.",
    )
  }
}

internal fun mebibytes(value: Long): Long = value * 1024L * 1024L

class GraphPilotMemoryProfileEstimator {
  fun estimate(
    workflowId: String,
    predictedCost: GraphPilotPredictedCost?,
  ): GraphPilotRequestMemoryProfile {
    val heuristic = heuristicForWorkflow(workflowId)
    val predictedPeakBytes = predictedCost?.peakMemoryBytes?.takeIf { it > 0L }
    val baseBytes = predictedPeakBytes ?: heuristic.baseBytes
    val source =
      if (predictedPeakBytes != null) {
        "predicted_cost_peak_memory"
      } else {
        "workflow_heuristic_v1"
      }
    return GraphPilotRequestMemoryProfile(
      workflowId = workflowId,
      baseBytes = baseBytes,
      kvBytes = heuristic.kvBytes,
      estimationSource = source,
      degradationOptions = heuristic.degradationOptions,
    )
  }

  private fun heuristicForWorkflow(workflowId: String): WorkflowMemoryHeuristic {
    return when (workflowId) {
      "workflow_a_voice_only" ->
        WorkflowMemoryHeuristic(
          baseBytes = mebibytes(100),
          kvBytes = mebibytes(6),
          degradationOptions =
            listOf(
              GraphPilotDegradationOption(
                id = "reduce_responder_max_tokens",
                freedBytes = mebibytes(3),
                addedLatencyMs = 0.0,
                qualityLoss = 0.05,
                responderMaxTokens = 24,
              )
            ),
        )
      "workflow_b_voice_vision" ->
        WorkflowMemoryHeuristic(
          baseBytes = mebibytes(156),
          kvBytes = mebibytes(6),
          degradationOptions =
            listOf(
              GraphPilotDegradationOption(
                id = "reduce_responder_max_tokens",
                freedBytes = mebibytes(3),
                addedLatencyMs = 0.0,
                qualityLoss = 0.05,
                responderMaxTokens = 24,
              )
            ),
        )
      "workflow_c_voice_vision_retrieval" ->
        WorkflowMemoryHeuristic(
          baseBytes = mebibytes(140),
          kvBytes = mebibytes(6),
          degradationOptions =
            listOf(
              GraphPilotDegradationOption(
                id = "reduce_retrieval_top_k",
                freedBytes = mebibytes(8),
                addedLatencyMs = 0.0,
                qualityLoss = 0.02,
                retrievalTopK = 1,
              ),
              GraphPilotDegradationOption(
                id = "reduce_responder_max_tokens",
                freedBytes = mebibytes(3),
                addedLatencyMs = 0.0,
                qualityLoss = 0.05,
                responderMaxTokens = 24,
              ),
            ),
        )
      else ->
        error(
          "Missing GraphPilot memory heuristic for workflow='${workflowId}'. " +
            "Remediation: add an explicit GraphPilotMemoryProfileEstimator entry before enabling runtime memory admission for this workflow.",
        )
    }
  }
}

class GraphPilotRuntimeMemoryManager(
  private val budget: GraphPilotMemoryBudget,
  private val controller: GraphPilotMemoryAdmissionController = GraphPilotMemoryAdmissionController(),
) {
  private val lock = kotlinx.coroutines.sync.Mutex()
  private var nextReservationId = 1L
  private val activeReservations = linkedMapOf<Long, Long>()

  suspend fun <T> runWithAdmission(
    workflowId: String,
    planId: String,
    profile: GraphPilotRequestMemoryProfile,
    block: suspend (GraphPilotMemoryObservation) -> T,
  ): T {
    val handle = admit(workflowId, planId, profile)
    try {
      return block(handle.observation)
    } finally {
      release(handle.reservationId)
    }
  }

  suspend fun activeReservedBytes(): Long = lock.lockedBytes()

  private suspend fun admit(
    workflowId: String,
    planId: String,
    profile: GraphPilotRequestMemoryProfile,
  ): MemoryReservationHandle = lock.withLock {
    val activeReservedBytes = activeReservations.values.sum()
    val availableUsableBytes = budget.usableBytes - activeReservedBytes
    val decision =
      if (availableUsableBytes <= 0L) {
        GraphPilotMemoryDecision(
          type = GraphPilotMemoryDecisionType.REJECT,
          requiredBytes = profile.requiredBytes,
          effectiveRequiredBytes = profile.requiredBytes,
          appliedActions = emptyList(),
          reason =
            "Request exceeds usable memory budget because active reservations already consumed all usable memory.",
        )
      } else {
        controller.decide(
          profile = profile,
          budget =
            GraphPilotMemoryBudget(
              totalBytes = availableUsableBytes + budget.marginBytes,
              marginBytes = budget.marginBytes,
            ),
        )
      }
    val reservedBytesAfterAdmission =
      if (decision.type == GraphPilotMemoryDecisionType.REJECT) {
        activeReservedBytes
      } else {
        activeReservedBytes + decision.effectiveRequiredBytes
      }
    val observation =
      GraphPilotMemoryObservation(
        workflowId = workflowId,
        planId = planId,
        profile = profile,
        budget = budget,
        activeReservedBytesAtAdmission = activeReservedBytes,
        reservedBytesAfterAdmission = reservedBytesAfterAdmission,
        decision = decision,
      )
    if (decision.type == GraphPilotMemoryDecisionType.REJECT) {
      throw GraphPilotMemoryAdmissionException(
        "GraphPilot rejected workflow='${workflowId}' plan='${planId}' because required_bytes=${decision.requiredBytes} " +
          "did not fit within usable memory after active_reserved_bytes=${activeReservedBytes}. " +
          "Remediation: increase runtime memory budget, reduce concurrent load, or tighten degradation knobs.",
        observation,
      )
    }
    val reservationId = nextReservationId++
    activeReservations[reservationId] = decision.effectiveRequiredBytes
    MemoryReservationHandle(reservationId = reservationId, observation = observation)
  }

  private suspend fun release(reservationId: Long) = lock.withLock {
    check(activeReservations.remove(reservationId) != null) {
      "GraphPilot runtime memory manager lost reservation_id=${reservationId}. " +
        "Remediation: investigate mismatched memory admission/release lifecycle in GraphPilotCoordinator."
    }
  }

  private suspend fun kotlinx.coroutines.sync.Mutex.lockedBytes(): Long = withLock {
    activeReservations.values.sum()
  }

  private data class MemoryReservationHandle(
    val reservationId: Long,
    val observation: GraphPilotMemoryObservation,
  )
}

private data class WorkflowMemoryHeuristic(
  val baseBytes: Long,
  val kvBytes: Long,
  val degradationOptions: List<GraphPilotDegradationOption>,
)
