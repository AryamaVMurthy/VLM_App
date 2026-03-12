package com.qidk.fastvlm.graphpilot

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.graphpilot.GraphPilotCoordinator
import com.qidk.fastvlm.core.graphpilot.GraphPilotMemoryAdmissionException
import com.qidk.fastvlm.core.graphpilot.GraphPilotMemoryBudget
import com.qidk.fastvlm.core.graphpilot.GraphPilotPlanStore
import com.qidk.fastvlm.core.graphpilot.mebibytes
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GraphPilotCoordinatorInstrumentedTest {
  @Test
  fun workflowAVoiceOnly_completes() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator = GraphPilotCoordinator(context)
    coordinator.use {
      val result = coordinator.runWorkflowAVoiceOnly(stateId = "cool")
      assertThat(result.workflowId).isEqualTo("workflow_a_voice_only")
      assertThat(result.stateId).isEqualTo("cool")
      assertThat(result.planId).contains("workflow_a_voice_only")
      assertThat(result.requestId).isGreaterThan(0L)
      assertThat(result.queueDepthAtAdmission).isEqualTo(0)
      assertThat(result.predictedQueueDelayAtAdmissionMs).isAtLeast(0.0)
      assertThat(result.queueWaitMs).isAtLeast(0L)
      assertThat(result.predictedStreamMakespanMs).isNotNull()
      assertThat(result.predictedP95QueueDelayMs).isNotNull()
      assertThat(result.predictedDeadlineMissRate).isNotNull()
      assertThat(result.memoryDecision).isEqualTo("ADMIT")
      assertThat(result.memoryRequiredBytes).isGreaterThan(0L)
      assertThat(result.transcript).isNotEmpty()
      assertThat(result.finalResponse).isNotEmpty()
      assertThat(result.ttftMs).isNotNull()
      assertThat(result.ttftMs).isGreaterThan(0L)
      assertThat(result.ttsFirstAudioMs).isNotNull()
      assertThat(result.ttsFirstAudioMs).isGreaterThan(0L)
      assertThat(result.stageBackends["planner.primary"]).isEqualTo("cpu")
    }
  }

  @Test
  fun workflowBVoiceVision_completes() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator = GraphPilotCoordinator(context)
    coordinator.use {
      val result = coordinator.runWorkflowBVoiceVision(stateId = "cool")
      assertThat(result.workflowId).isEqualTo("workflow_b_voice_vision")
      assertThat(result.stateId).isEqualTo("cool")
      assertThat(result.planId).contains("workflow_b_voice_vision")
      assertThat(result.requestId).isGreaterThan(0L)
      assertThat(result.queueDepthAtAdmission).isEqualTo(0)
      assertThat(result.predictedQueueDelayAtAdmissionMs).isAtLeast(0.0)
      assertThat(result.queueWaitMs).isAtLeast(0L)
      assertThat(result.predictedStreamMakespanMs).isNotNull()
      assertThat(result.predictedP95QueueDelayMs).isNotNull()
      assertThat(result.predictedDeadlineMissRate).isNotNull()
      assertThat(result.memoryDecision).isEqualTo("ADMIT")
      assertThat(result.transcript).isNotEmpty()
      assertThat(result.vlmText).isNotEmpty()
      assertThat(result.finalResponse).isNotEmpty()
      assertThat(result.stageBackends["vlm.fastvlm.primary"]).isEqualTo("npu")
      assertThat(result.vlmPrefillMs).isGreaterThan(0L)
    }
  }

  @Test
  fun workflowCVoiceVisionRetrieval_completes() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator = GraphPilotCoordinator(context)
    val planStore = GraphPilotPlanStore()
    val expectedRetrievalBackend =
      planStore.loadPlan("workflow_c_voice_vision_retrieval", "cool").retrievalBackend().name.lowercase()
    coordinator.use {
      val result = coordinator.runWorkflowCVoiceVisionRetrieval(stateId = "cool")
      assertThat(result.workflowId).isEqualTo("workflow_c_voice_vision_retrieval")
      assertThat(result.stateId).isEqualTo("cool")
      assertThat(result.planId).contains("workflow_c_voice_vision_retrieval")
      assertThat(result.requestId).isGreaterThan(0L)
      assertThat(result.queueDepthAtAdmission).isEqualTo(0)
      assertThat(result.predictedQueueDelayAtAdmissionMs).isAtLeast(0.0)
      assertThat(result.queueWaitMs).isAtLeast(0L)
      assertThat(result.predictedStreamMakespanMs).isNotNull()
      assertThat(result.predictedP95QueueDelayMs).isNotNull()
      assertThat(result.predictedDeadlineMissRate).isNotNull()
      assertThat(result.memoryDecision).isEqualTo("ADMIT")
      assertThat(result.transcript).isNotEmpty()
      assertThat(result.vlmText).isNotEmpty()
      assertThat(result.retrievalText).isNotEmpty()
      assertThat(result.finalResponse).isNotEmpty()
      assertThat(result.stageBackends["vlm.fastvlm.primary"]).isEqualTo("npu")
      assertThat(result.stageBackends["retrieval.embedder.primary"]).isEqualTo(expectedRetrievalBackend)
    }
  }

  @Test
  fun workflowAConcurrentRequests_emitQueueObservability() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator = GraphPilotCoordinator(context)
    coordinator.use {
      val first = async { coordinator.runWorkflowAVoiceOnly(stateId = "cool") }
      delay(100)
      val second = async { coordinator.runWorkflowAVoiceOnly(stateId = "cool") }

      val firstResult = first.await()
      val secondResult = second.await()

      assertThat(firstResult.queueDepthAtAdmission).isEqualTo(0)
      assertThat(secondResult.queueDepthAtAdmission).isAtLeast(1)
      assertThat(secondResult.queueWaitMs).isGreaterThan(0L)
      assertThat(secondResult.requestId).isNotEqualTo(firstResult.requestId)
    }
  }

  @Test
  fun workflowATightMemoryBudget_degradesExplicitly() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator =
      GraphPilotCoordinator(
        context,
        memoryBudget = GraphPilotMemoryBudget(totalBytes = mebibytes(108), marginBytes = mebibytes(4)),
      )
    coordinator.use {
      val result = coordinator.runWorkflowAVoiceOnly(stateId = "cool")
      assertThat(result.memoryDecision).isEqualTo("DEGRADE")
      assertThat(result.memoryAppliedActions).contains("reduce_responder_max_tokens")
      assertThat(result.effectiveResponderMaxTokens).isEqualTo(24)
    }
  }

  @Test
  fun workflowAConcurrentRequests_tightMemoryBudgetRejectsSecond() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val coordinator =
      GraphPilotCoordinator(
        context,
        memoryBudget = GraphPilotMemoryBudget(totalBytes = mebibytes(108), marginBytes = mebibytes(4)),
      )
    coordinator.use {
      val first = async { coordinator.runWorkflowAVoiceOnly(stateId = "cool") }
      delay(100)
      val secondError =
        async {
          runCatching { coordinator.runWorkflowAVoiceOnly(stateId = "cool") }.exceptionOrNull()
        }.await()

      assertThat(secondError).isInstanceOf(GraphPilotMemoryAdmissionException::class.java)
      val typed = secondError as GraphPilotMemoryAdmissionException
      assertThat(typed.observation.decision.type.name).isEqualTo("REJECT")
      assertThat(typed.observation.decision.reason).contains("memory budget")
      first.await()
      Unit
    }
  }
}
