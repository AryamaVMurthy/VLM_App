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
            "predicted_cost": {
              "stream_makespan_ms": 12345.0,
              "p95_queue_delay_ms": 456.0,
              "deadline_miss_rate": 0.25
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
    assertThat(plan.predictedCost?.streamMakespanMs).isEqualTo(12345.0)
    assertThat(plan.predictedCost?.p95QueueDelayMs).isEqualTo(456.0)
    assertThat(plan.predictedCost?.deadlineMissRate).isEqualTo(0.25)
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
}
