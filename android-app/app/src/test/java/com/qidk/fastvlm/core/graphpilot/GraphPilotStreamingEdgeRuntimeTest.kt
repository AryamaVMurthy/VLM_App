package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import org.junit.Test

class GraphPilotStreamingEdgeRuntimeTest {
  @Test
  fun splitMono16kPcmIntoChunksUsesExplicitChunkDuration() {
    val audio = FloatArray(40_000) { it.toFloat() }

    val chunks = splitMono16kPcmIntoChunks(audio, chunkSizeMs = 1000)

    assertThat(chunks).hasSize(3)
    assertThat(chunks[0].size).isEqualTo(16_000)
    assertThat(chunks[1].size).isEqualTo(16_000)
    assertThat(chunks[2].size).isEqualTo(8_000)
  }

  @Test
  fun runAsrPlannerPipelineRunsSinglePassWhenChunkingDisabled() = runTest {
    var nowMs = 1_000L

    val result =
      runAsrPlannerPipeline(
        audioData = FloatArray(16_000) { 0.25f },
        chunkSizeMs = null,
        totalStartMs = 1_000L,
        clockMs = { nowMs },
        transcribe = {
          nowMs += 40L
          "hello world"
        },
        plan = {
          nowMs += 10L
          "single plan"
        },
      )

    assertThat(result.transcript).isEqualTo("hello world")
    assertThat(result.plannerText).isEqualTo("single plan")
    assertThat(result.asrElapsedMs).isEqualTo(40L)
    assertThat(result.plannerElapsedMs).isEqualTo(10L)
    assertThat(result.plannerFirstPartialOffsetMs).isNull()
    assertThat(result.asrChunkCount).isEqualTo(1)
    assertThat(result.plannerInvocationCount).isEqualTo(1)
  }

  @Test
  fun runAsrPlannerPipelineInvokesPlannerForEachTranscriptChunk() = runTest {
    var nowMs = 1_000L
    val plannerInputs = mutableListOf<String>()
    val transcripts = ArrayDeque(listOf("hello", "world", "again"))

    val result =
      runAsrPlannerPipeline(
        audioData = FloatArray(40_000) { 0.5f },
        chunkSizeMs = 1000,
        totalStartMs = 1_000L,
        clockMs = { nowMs },
        transcribe = {
          nowMs += 5L
          transcripts.removeFirst()
        },
        plan = { transcript ->
          nowMs += 7L
          plannerInputs += transcript
          "plan for $transcript"
        },
      )

    assertThat(plannerInputs)
      .containsExactly(
        "hello",
        "hello world",
        "hello world again",
      )
      .inOrder()
    assertThat(result.transcript).isEqualTo("hello world again")
    assertThat(result.plannerText).isEqualTo("plan for hello world again")
    assertThat(result.asrElapsedMs).isEqualTo(15L)
    assertThat(result.plannerElapsedMs).isEqualTo(21L)
    assertThat(result.plannerFirstPartialOffsetMs).isEqualTo(12L)
    assertThat(result.asrChunkCount).isEqualTo(3)
    assertThat(result.plannerInvocationCount).isEqualTo(3)
  }
}
