package com.qidk.fastvlm.core.graphpilot

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

enum class GraphPilotAdmissionDecision {
  ADMIT,
  REJECT_QUEUE_FULL,
  REJECT_DEADLINE_RISK,
}

class GraphPilotAdmissionException(
  message: String,
  val decision: GraphPilotAdmissionDecision,
) : IllegalStateException(message)

data class GraphPilotQueueObservation(
  val requestId: Long,
  val workflowId: String,
  val planId: String,
  val queueDepthAtAdmission: Int,
  val predictedQueueDelayMs: Double,
  val predictedStreamMakespanMs: Double,
  val predictedDeadlineMiss: Boolean,
  val deadlineMs: Long?,
  val admittedAtMs: Long,
  val admissionDecision: GraphPilotAdmissionDecision,
  val requiredBackends: Set<String> = emptySet(),
  val priorityScore: Double = 0.0,
  val queueWaitMs: Long = 0L,
)

private data class GraphPilotActiveRequest(
  val requestId: Long,
  val predictedServiceMs: Double,
  val startedAtMs: Long,
  val requiredBackends: Set<String>,
) {
  fun remainingMs(nowMs: Long): Double = (predictedServiceMs - (nowMs - startedAtMs)).coerceAtLeast(0.0)
}

private data class GraphPilotPendingRequest(
  val requestId: Long,
  val predictedServiceMs: Double,
  val requiredBackends: Set<String>,
  val priorityScore: Double,
  val startSignal: CompletableDeferred<Unit>,
)

data class GraphPilotSchedulerWeights(
  val latencyWeight: Double = 1.0,
  val firstOutputWeight: Double = 3.0,
  val slackWeight: Double = 2.0,
  val deadlineUrgencyWeight: Double = 500.0,
  val queuePenaltyWeight: Double = 0.01,
)

class GraphPilotRuntimeScheduler(
  private val maxQueuedRequests: Int = 2,
  private val clockMs: () -> Long = { System.currentTimeMillis() },
  private val weights: GraphPilotSchedulerWeights = GraphPilotSchedulerWeights(),
) {
  init {
    require(maxQueuedRequests >= 0) { "maxQueuedRequests must be >= 0" }
  }

  private val lock = Mutex()
  private val pending = mutableListOf<GraphPilotPendingRequest>()
  private val active = linkedMapOf<Long, GraphPilotActiveRequest>()
  private var nextRequestId = 1L

  suspend fun <T> runWithAdmission(
    workflowId: String,
    plan: GraphPilotExecutionPlan,
    deadlineMs: Long? = null,
    block: suspend (GraphPilotQueueObservation) -> T,
  ): Pair<GraphPilotQueueObservation, T> {
    val handle = admit(workflowId, plan, deadlineMs)
    handle.startSignal.await()
    val startedAtMs = clockMs()
    val observation = handle.observation.copy(queueWaitMs = startedAtMs - handle.observation.admittedAtMs)
    try {
      val value = block(observation)
      return observation to value
    } finally {
      complete(handle.requestId)
    }
  }

  private suspend fun admit(
    workflowId: String,
    plan: GraphPilotExecutionPlan,
    deadlineMs: Long?,
  ): GraphPilotAdmissionHandle = lock.withLock {
    val predictedCost = plan.requirePredictedStreamCost()
    val nowMs = clockMs()
    val requestId = nextRequestId++
    val predictedServiceMs = predictedCost.streamMakespanMs!!
    val requiredBackends = requiredBackendSet(plan)
    val priorityScore = computePriorityScore(plan, deadlineMs)
    val queueDepthAtAdmission = overlappingRequestCount(requiredBackends)
    val predictedQueueDelayMs = predictQueueDelayMs(requiredBackends, nowMs)
    val predictedCompletionMs = predictedQueueDelayMs + predictedServiceMs
    val predictedDeadlineMiss =
      deadlineMs?.let { predictedCompletionMs > it } ?: (predictedCost.deadlineMissRate!! > 0.0)

    if (queueDepthAtAdmission > maxQueuedRequests) {
      throw GraphPilotAdmissionException(
        "GraphPilot rejected workflow='${workflowId}' plan='${plan.plan_id}' because queue_depth=${queueDepthAtAdmission} exceeded maxQueuedRequests=${maxQueuedRequests}. " +
          "Remediation: increase queue capacity, reduce request rate, or choose a lower-latency plan.",
        GraphPilotAdmissionDecision.REJECT_QUEUE_FULL,
      )
    }
    if (deadlineMs != null && predictedDeadlineMiss) {
      throw GraphPilotAdmissionException(
        "GraphPilot rejected workflow='${workflowId}' plan='${plan.plan_id}' because predicted_completion_ms=${predictedCompletionMs} exceeds deadline_ms=${deadlineMs}. " +
          "Remediation: relax the deadline, lower model/knob costs, or select a faster backend plan.",
        GraphPilotAdmissionDecision.REJECT_DEADLINE_RISK,
      )
    }

    val observation =
      GraphPilotQueueObservation(
        requestId = requestId,
        workflowId = workflowId,
        planId = plan.plan_id,
        queueDepthAtAdmission = queueDepthAtAdmission,
        predictedQueueDelayMs = predictedQueueDelayMs,
        predictedStreamMakespanMs = predictedServiceMs,
        predictedDeadlineMiss = predictedDeadlineMiss,
        deadlineMs = deadlineMs,
        admittedAtMs = nowMs,
        admissionDecision = GraphPilotAdmissionDecision.ADMIT,
        requiredBackends = requiredBackends,
        priorityScore = priorityScore,
      )
    if (canStartImmediately(requiredBackends)) {
      active[requestId] =
        GraphPilotActiveRequest(
          requestId = requestId,
          predictedServiceMs = predictedServiceMs,
          startedAtMs = nowMs,
          requiredBackends = requiredBackends,
        )
      return@withLock GraphPilotAdmissionHandle(
        requestId = requestId,
        observation = observation,
        startSignal = CompletableDeferred<Unit>().also { it.complete(Unit) },
      )
    }

    val startSignal = CompletableDeferred<Unit>()
    pending.addLast(
      GraphPilotPendingRequest(
        requestId = requestId,
        predictedServiceMs = predictedServiceMs,
        requiredBackends = requiredBackends,
        priorityScore = priorityScore,
        startSignal = startSignal,
      )
    )
    pending.sortWith(
      compareByDescending<GraphPilotPendingRequest> { it.priorityScore }
        .thenBy { it.predictedServiceMs }
        .thenBy { it.requestId }
    )
    GraphPilotAdmissionHandle(
      requestId = requestId,
      observation = observation,
      startSignal = startSignal,
    )
  }

  private suspend fun complete(requestId: Long) = lock.withLock {
    val removed = active.remove(requestId)
    check(removed != null) {
      "GraphPilot runtime scheduler lost queue ownership for request_id=${requestId}. " +
        "Remediation: investigate concurrent coordinator execution and scheduler lifecycle."
    }
    var startedRequest = true
    while (startedRequest) {
      startedRequest = false
      val reservedBackends = active.values.flatMapTo(linkedSetOf()) { it.requiredBackends }
      val nextIndex =
        pending.indexOfFirst { request ->
          request.requiredBackends.none { it in reservedBackends }
        }
      if (nextIndex >= 0) {
        val next = pending.removeAt(nextIndex)
        active[next.requestId] =
          GraphPilotActiveRequest(
            requestId = next.requestId,
            predictedServiceMs = next.predictedServiceMs,
            startedAtMs = clockMs(),
            requiredBackends = next.requiredBackends,
          )
        next.startSignal.complete(Unit)
        startedRequest = true
      }
    }
  }

  private fun overlappingRequestCount(requiredBackends: Set<String>): Int {
    val activeConflicts = active.values.count { overlaps(it.requiredBackends, requiredBackends) }
    val pendingConflicts = pending.count { overlaps(it.requiredBackends, requiredBackends) }
    return activeConflicts + pendingConflicts
  }

  private fun predictQueueDelayMs(requiredBackends: Set<String>, nowMs: Long): Double {
    val activeDelay =
      active.values
        .filter { overlaps(it.requiredBackends, requiredBackends) }
        .maxOfOrNull { it.remainingMs(nowMs) }
        ?: 0.0
    val pendingDelay =
      pending
        .filter { overlaps(it.requiredBackends, requiredBackends) }
        .sumOf { it.predictedServiceMs }
    return activeDelay + pendingDelay
  }

  private fun canStartImmediately(requiredBackends: Set<String>): Boolean {
    return active.values.none { overlaps(it.requiredBackends, requiredBackends) }
  }

  private fun computePriorityScore(
    plan: GraphPilotExecutionPlan,
    deadlineMs: Long?,
  ): Double {
    val cost = plan.requirePredictedStreamCost()
    val streamMs = cost.streamMakespanMs ?: 1.0
    val firstOutputMs =
      listOfNotNull(cost.ttftMs, cost.p95TtfsMs, cost.ttfsMs, cost.streamMakespanMs)
        .filter { it > 0.0 }
        .minOrNull()
        ?: streamMs
    val slackScore =
      if (deadlineMs == null) {
        0.0
      } else {
        val slackMs = (deadlineMs - streamMs).coerceAtLeast(0.0)
        1000.0 / (1.0 + slackMs)
      }
    val deadlineUrgency = (cost.deadlineMissRate ?: 0.0) * weights.deadlineUrgencyWeight
    val queuePenalty = (cost.p95QueueDelayMs ?: 0.0) * weights.queuePenaltyWeight
    return (
      weights.latencyWeight * (1000.0 / streamMs.coerceAtLeast(1.0))
        + weights.firstOutputWeight * (1000.0 / firstOutputMs.coerceAtLeast(1.0))
        + weights.slackWeight * slackScore
        + deadlineUrgency
        - queuePenalty
      )
  }

  private fun requiredBackendSet(plan: GraphPilotExecutionPlan): Set<String> {
    return plan.backend_map.values.map { it.lowercase() }.toSet()
  }

  private fun overlaps(left: Set<String>, right: Set<String>): Boolean {
    return left.any { it in right }
  }
}

private data class GraphPilotAdmissionHandle(
  val requestId: Long,
  val observation: GraphPilotQueueObservation,
  val startSignal: CompletableDeferred<Unit>,
)
