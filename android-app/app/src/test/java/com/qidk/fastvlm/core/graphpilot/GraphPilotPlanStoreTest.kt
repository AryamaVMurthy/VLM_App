package com.qidk.fastvlm.core.graphpilot

import com.google.ai.edge.litertlm.Backend
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.model.BackendTarget
import java.io.File
import kotlin.io.path.createTempFile
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class GraphPilotPlanStoreTest {

  @Test
  fun loadPlanParsesWorkflowAndBackendSelections() {
    val registry =
      """
      {
        "plans": [
          {
            "plan_id": "cool:workflow_b_voice_vision:test",
            "workflow_template": "workflow_b_voice_vision",
            "state_id": "cool",
            "backend_map": {
              "asr.primary": "cpu",
              "planner.primary": "gpu",
              "vlm.fastvlm.primary": "npu",
              "responder.primary": "cpu",
              "tts.primary": "cpu"
            },
            "stream_edges": [
              {
                "from": "asr.primary",
                "to": "planner.primary",
                "mode": "chunk"
              }
            ],
            "chunk_sizes": {
              "asr.primary": 800
            },
            "predicted_cost": {
              "stream_makespan_ms": 12345.0,
              "p95_e2e_ms": 12000.0,
              "p95_ttfs_ms": 7000.0,
              "avg_energy_mj": 88.0,
              "copy_bytes": 4096,
              "quality_loss": 0.1,
              "p95_queue_delay_ms": 456.0,
              "deadline_miss_rate": 0.25,
              "objective_score": 999.0
            }
          }
        ]
      }
      """.trimIndent()

    val path = createTempFile(suffix = ".json").toFile()
    path.writeText(registry)
    val plan = GraphPilotPlanStore(path).loadPlan("workflow_b_voice_vision", "cool")

    assertThat(plan.plan_id).isEqualTo("cool:workflow_b_voice_vision:test")
    assertThat(plan.textBackend("planner.primary")).isEqualTo(Backend.GPU)
    assertThat(plan.vlmBackendTarget()).isEqualTo(BackendTarget.NPU)
    assertThat(plan.hasChunkStreamEdge("asr.primary", "planner.primary")).isTrue()
    assertThat(plan.chunkSizeOrNull("asr.primary")).isEqualTo(800)
    assertThat(plan.predictedCost?.streamMakespanMs).isEqualTo(12345.0)
    assertThat(plan.predictedCost?.p95E2eMs).isEqualTo(12000.0)
    assertThat(plan.predictedCost?.p95TtfsMs).isEqualTo(7000.0)
    assertThat(plan.predictedCost?.avgEnergyMj).isEqualTo(88.0)
    assertThat(plan.predictedCost?.copyBytes).isEqualTo(4096L)
    assertThat(plan.predictedCost?.qualityLoss).isEqualTo(0.1)
    assertThat(plan.predictedCost?.p95QueueDelayMs).isEqualTo(456.0)
    assertThat(plan.predictedCost?.deadlineMissRate).isEqualTo(0.25)
    assertThat(plan.predictedCost?.objectiveScore).isEqualTo(999.0)
  }

  @Test(expected = IllegalStateException::class)
  fun requirePredictedStreamCostFailsFastWhenStreamMetricsMissing() {
    val registry =
      """
      {
        "plans": [
          {
            "plan_id": "cool:workflow_a_voice_only:test",
            "workflow_template": "workflow_a_voice_only",
            "state_id": "cool",
            "backend_map": {
              "asr.primary": "cpu",
              "planner.primary": "cpu",
              "responder.primary": "cpu",
              "tts.primary": "cpu"
            },
            "predicted_cost": {
              "makespan_ms": 1234
            }
          }
        ]
      }
      """.trimIndent()

    val path = createTempFile(suffix = ".json").toFile()
    path.writeText(registry)

    GraphPilotPlanStore(path)
      .loadPlan("workflow_a_voice_only", "cool")
      .requirePredictedStreamCost()
  }

  @Test(expected = IllegalStateException::class)
  fun loadPlanFailsFastForMalformedRegistry() {
    val path = createTempFile(suffix = ".json").toFile()
    path.writeText("{ not-json }")
    GraphPilotPlanStore(path).loadPlan("workflow_a_voice_only", "cool")
  }

  @Test(expected = IllegalStateException::class)
  fun loadPlanFailsFastWhenRegistryMissing() {
    val missing = File("/tmp/graphpilot_missing_registry.json")
    if (missing.exists()) {
      missing.delete()
    }
    GraphPilotPlanStore(missing).loadPlan("workflow_a_voice_only", "cool")
  }

  @Test(expected = IllegalStateException::class)
  fun requireChunkSizeFailsFastWhenChunkEdgeHasNoExplicitSize() {
    val plan =
      GraphPilotExecutionPlan(
        plan_id = "cool:workflow_a_voice_only:test",
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
            GraphPilotStreamEdge(
              from = "asr.primary",
              to = "planner.primary",
              streamMode = "chunk",
            )
          ),
      )

    plan.requireChunkSize("asr.primary")
  }
}
