package com.qidk.fastvlm.text

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.text.LiteRtLmTextStageAdapter
import com.qidk.fastvlm.core.text.TextStageModelSpec
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LiteRtLmTextStageInstrumentedTest {
  @Test
  fun plannerCpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.planner(Backend.CPU),
      prompt = "Summarize the next action in five words: inspect the image and answer the user.",
      maxOutputTokens = 24,
    )
  }

  @Test
  fun plannerGpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.planner(Backend.GPU),
      prompt = "Summarize the next action in five words: inspect the image and answer the user.",
      maxOutputTokens = 24,
    )
  }

  @Test
  fun plannerNpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.planner(Backend.NPU),
      prompt = "Summarize the next action in five words: inspect the image and answer the user.",
      maxOutputTokens = 24,
    )
  }

  @Test
  fun responderCpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.responder(Backend.CPU),
      prompt = "Answer in one sentence: what is an edge AI runtime?",
      maxOutputTokens = 32,
    )
  }

  @Test
  fun responderGpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.responder(Backend.GPU),
      prompt = "Answer in one sentence: what is an edge AI runtime?",
      maxOutputTokens = 32,
    )
  }

  @Test
  fun responderNpuPromptCompletes() = runBlocking {
    runStageSmoke(
      spec = TextStageModelSpec.responder(Backend.NPU),
      prompt = "Answer in one sentence: what is an edge AI runtime?",
      maxOutputTokens = 32,
    )
  }

  @Test
  fun responderCpuPromptStreamsNonEmptyDeltas() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val adapter = LiteRtLmTextStageAdapter(context)
    val deltas = mutableListOf<String>()
    try {
      withTimeout(180_000L) {
        adapter.ensureInitialized(TextStageModelSpec.responder(Backend.CPU))
        val result =
          adapter.generateStreaming(
            prompt = "Answer directly in one short sentence: what is an edge AI runtime?",
            maxOutputTokens = 32,
          ) { delta ->
            deltas += delta
          }
        assertThat(deltas.joinToString(separator = "")).isNotEmpty()
        assertThat(result.text).isNotEmpty()
        assertThat(result.firstTokenAtMs).isNotNull()
      }
    } finally {
      adapter.close()
    }
  }

  private suspend fun runStageSmoke(
    spec: TextStageModelSpec,
    prompt: String,
    maxOutputTokens: Int,
  ) {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val adapter = LiteRtLmTextStageAdapter(context)
    try {
      withTimeout(180_000L) {
        val initStartMs = System.currentTimeMillis()
        val modelPath = adapter.ensureInitialized(spec)
        val initElapsedMs = System.currentTimeMillis() - initStartMs
        val generateStartMs = System.currentTimeMillis()
        val output = adapter.generate(prompt, maxOutputTokens)
        val generateElapsedMs = System.currentTimeMillis() - generateStartMs
        Log.i(
          "LiteRtLmTextStageInstrumentedTest",
          "TEXT_STAGE role=${spec.role} backend=${spec.backend} model=$modelPath init_elapsed_ms=$initElapsedMs generate_elapsed_ms=$generateElapsedMs output='${output.take(120)}'",
        )
        assertThat(output).isNotEmpty()
      }
    } finally {
      adapter.close()
    }
  }
}
