package com.qidk.fastvlm.core.graphpilot

import com.google.ai.edge.litertlm.Backend
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.model.BackendTarget
import java.io.File
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

private const val DEFAULT_PLAN_REGISTRY_PATH = "/data/local/tmp/graphpilot_edge/candidate_plan_registry.json"

@Serializable
data class GraphPilotCandidatePlanRegistry(
  val plans: List<GraphPilotExecutionPlan> = emptyList(),
)

@Serializable
data class GraphPilotExecutionPlan(
  val plan_id: String,
  val workflow_template: String,
  val state_id: String,
  val backend_map: Map<String, String>,
  val stage_variant_map: Map<String, String> = emptyMap(),
  val macro_region_map: Map<String, String> = emptyMap(),
  val stream_edges: List<GraphPilotStreamEdge> = emptyList(),
  val chunk_sizes: Map<String, Int> = emptyMap(),
  val kv_policy: String = "sticky_decode",
  val buffer_policy: String = "interval_reuse_best_fit",
  val degradation_policy: String = "memory_guardrail_v1",
  @SerialName("predicted_cost") val predictedCost: GraphPilotPredictedCost? = null,
) {
  fun hasTokenStreamEdge(fromStageId: String, toStageId: String): Boolean {
    return stream_edges.any { edge ->
      edge.from == fromStageId &&
        edge.to == toStageId &&
        edge.streamMode.equals("token", ignoreCase = true)
    }
  }

  fun requirePredictedStreamCost(): GraphPilotPredictedCost {
    val cost = predictedCost
      ?: error(
        "GraphPilot execution plan '${plan_id}' is missing predicted_cost. " +
          "Remediation: regenerate candidate_plan_registry.json with stream-aware planning data and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}.",
      )
    check(cost.streamMakespanMs != null) {
      "GraphPilot execution plan '${plan_id}' is missing predicted_cost.stream_makespan_ms. " +
        "Remediation: regenerate candidate_plan_registry.json with continuous-stream simulation metrics and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}."
    }
    check(cost.p95QueueDelayMs != null) {
      "GraphPilot execution plan '${plan_id}' is missing predicted_cost.p95_queue_delay_ms. " +
        "Remediation: regenerate candidate_plan_registry.json with queue-delay metrics and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}."
    }
    check(cost.deadlineMissRate != null) {
      "GraphPilot execution plan '${plan_id}' is missing predicted_cost.deadline_miss_rate. " +
        "Remediation: regenerate candidate_plan_registry.json with deadline-miss metrics and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}."
    }
    return cost
  }

  fun requireStageBackend(stageId: String): String {
    return backend_map[stageId]?.lowercase()
      ?: error(
        "GraphPilot execution plan '${plan_id}' is missing backend_map['${stageId}']. " +
          "Remediation: regenerate candidate_plan_registry.json and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}.",
      )
  }

  fun textBackend(stageId: String): Backend {
    return when (requireStageBackend(stageId)) {
      "cpu" -> Backend.CPU
      "gpu" -> Backend.GPU
      "npu" -> Backend.NPU
      else ->
        error(
          "Unsupported text backend '${requireStageBackend(stageId)}' for stage '${stageId}' in plan '${plan_id}'.",
        )
    }
  }

  fun retrievalBackend(): Backend = textBackend("retrieval.embedder.primary")

  fun vlmBackendTarget(): BackendTarget {
    return when (requireStageBackend("vlm.fastvlm.primary")) {
      "cpu" -> BackendTarget.CPU
      "npu" -> BackendTarget.NPU
      "gpu" ->
        error(
          "GraphPilot plan '${plan_id}' requested VLM GPU, but FastVLMNativeBridge only exposes CPU/NPU. " +
            "Remediation: keep VLM on support-safe CPU/NPU paths until a GPU adapter exists.",
        )
      else ->
        error(
          "Unsupported VLM backend '${requireStageBackend("vlm.fastvlm.primary")}' in plan '${plan_id}'.",
        )
    }
  }
}

@Serializable
data class GraphPilotPredictedCost(
  @SerialName("stream_makespan_ms") val streamMakespanMs: Double? = null,
  @SerialName("p95_queue_delay_ms") val p95QueueDelayMs: Double? = null,
  @SerialName("deadline_miss_rate") val deadlineMissRate: Double? = null,
  @SerialName("memory_mb") val memoryMb: Int? = null,
  @SerialName("peak_memory_bytes") val peakMemoryBytes: Long? = null,
  @SerialName("ttft_ms") val ttftMs: Double? = null,
  @SerialName("ttfs_ms") val ttfsMs: Double? = null,
)

@Serializable
data class GraphPilotStreamEdge(
  val from: String,
  val to: String,
  @SerialName("mode") val streamMode: String,
)

class GraphPilotPlanStore(
  private val registryPath: File = File(DEFAULT_PLAN_REGISTRY_PATH),
) {
  private val json = JsonCodec.instance

  fun loadPlan(workflowId: String, stateId: String): GraphPilotExecutionPlan {
    check(registryPath.isFile) {
      "Missing GraphPilot candidate plan registry '${registryPath.absolutePath}'. " +
        "Remediation: push artifacts/graphpilot_edge/registries/candidate_plan_registry.json to ${DEFAULT_PLAN_REGISTRY_PATH} before running GraphPilot workflows."
    }
    val registry =
      try {
        json.decodeFromString<GraphPilotCandidatePlanRegistry>(registryPath.readText())
      } catch (t: Throwable) {
        throw IllegalStateException(
          "Failed to parse GraphPilot candidate plan registry '${registryPath.absolutePath}': ${t.message}",
          t,
        )
      }
    return registry.plans.firstOrNull {
      it.workflow_template == workflowId && it.state_id == stateId
    }
      ?: error(
        "No GraphPilot execution plan found for workflow='${workflowId}' state='${stateId}' in '${registryPath.absolutePath}'. " +
          "Remediation: regenerate candidate_plan_registry.json for the current backend matrix and restage it to ${DEFAULT_PLAN_REGISTRY_PATH}.",
      )
  }
}
