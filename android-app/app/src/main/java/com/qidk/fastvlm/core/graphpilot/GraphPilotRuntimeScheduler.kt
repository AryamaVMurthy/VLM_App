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
  val queueWaitMs: Long = 0L,
)

private data class GraphPilotActiveRequest(
  val requestId: Long,
  val predictedServiceMs: Double,
  val startedAtMs: Long,
) {
  fun remainingMs(nowMs: Long): Double = (predictedServiceMs - (nowMs - startedAtMs)).coerceAtLeast(0.0)
}

private data class GraphPilotPendingRequest(
  val requestId: Long,
  val predictedServiceMs: Double,
  val startSignal: CompletableDeferred<Unit>,
)

class GraphPilotRuntimeScheduler(
  private val maxQueuedRequests: Int = 2,
  private val clockMs: () -> Long = { System.currentTimeMillis() },
) {
  init {
    require(maxQueuedRequests >= 0) { "maxQueuedRequests must be >= 0" }
  }

  private val lock = Mutex()
  private val pending = ArrayDeque<GraphPilotPendingRequest>()
  private var active: GraphPilotActiveRequest? = null
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
    val activeRemainingMs = active?.remainingMs(nowMs) ?: 0.0
    val queuedAheadMs = pending.sumOf { it.predictedServiceMs }
    val queueDepthAtAdmission = pending.size + if (active != null) 1 else 0
    val predictedQueueDelayMs = activeRemainingMs + queuedAheadMs
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
      )
    if (active == null && pending.isEmpty()) {
      active = GraphPilotActiveRequest(requestId = requestId, predictedServiceMs = predictedServiceMs, startedAtMs = nowMs)
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
        startSignal = startSignal,
      )
    )
    GraphPilotAdmissionHandle(
      requestId = requestId,
      observation = observation,
      startSignal = startSignal,
    )
  }

  private suspend fun complete(requestId: Long) = lock.withLock {
    val activeRequest = active
    check(activeRequest?.requestId == requestId) {
      "GraphPilot runtime scheduler lost queue ownership for request_id=${requestId}. " +
        "Remediation: investigate concurrent coordinator execution and scheduler lifecycle."
    }
    active = null
    if (pending.isNotEmpty()) {
      val next = pending.removeFirst()
      active =
        GraphPilotActiveRequest(
          requestId = next.requestId,
          predictedServiceMs = next.predictedServiceMs,
          startedAtMs = clockMs(),
        )
      next.startSignal.complete(Unit)
    }
  }
}

private data class GraphPilotAdmissionHandle(
  val requestId: Long,
  val observation: GraphPilotQueueObservation,
  val startSignal: CompletableDeferred<Unit>,
)
