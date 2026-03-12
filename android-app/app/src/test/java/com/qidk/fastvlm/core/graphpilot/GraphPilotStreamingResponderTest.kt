package com.qidk.fastvlm.core.graphpilot

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.runBlocking
import org.junit.Test

class GraphPilotStreamingResponderTest {
  @Test
  fun generateAndStreamForwardsNonBlankDeltasAndFinishesSpeech() = runBlocking {
    val clockValues = ArrayDeque(listOf(1_000L, 1_025L, 1_050L))
    val generator =
      object : GraphPilotStreamingTextGenerator {
        override suspend fun generateStreaming(
          prompt: String,
          maxOutputTokens: Int,
          onDelta: suspend (String) -> Unit,
        ): GraphPilotTextGenerationResult {
          assertThat(prompt).contains("Respond directly")
          assertThat(maxOutputTokens).isEqualTo(48)
          onDelta("Hello")
          onDelta(" ")
          onDelta("world.")
          return GraphPilotTextGenerationResult(text = "Hello world.", firstTokenAtMs = 1_025L)
        }
      }
    val streamer = FakeSpeechStreamer()

    val result =
      GraphPilotStreamingResponder(clockMs = { clockValues.removeFirst() })
        .generateAndStream(
          prompt = "Respond directly",
          maxOutputTokens = 48,
          generator = generator,
          speechStreamer = streamer,
        )

    assertThat(streamer.startedSessions).isEqualTo(1)
    assertThat(streamer.appendedDeltas).containsExactly("Hello", "world.").inOrder()
    assertThat(streamer.finishCalls).isEqualTo(1)
    assertThat(streamer.abortCalls).isEqualTo(0)
    assertThat(result.text).isEqualTo("Hello world.")
    assertThat(result.firstTokenOffsetMs).isEqualTo(25L)
  }

  @Test
  fun generateAndStreamAbortsSpeechOnGeneratorFailure() = runBlocking {
    val generator =
      object : GraphPilotStreamingTextGenerator {
        override suspend fun generateStreaming(
          prompt: String,
          maxOutputTokens: Int,
          onDelta: suspend (String) -> Unit,
        ): GraphPilotTextGenerationResult {
          onDelta("partial")
          error("stream failed")
        }
      }
    val streamer = FakeSpeechStreamer()

    val error =
      runCatching {
        GraphPilotStreamingResponder()
          .generateAndStream(
            prompt = "Respond directly",
            maxOutputTokens = 48,
            generator = generator,
            speechStreamer = streamer,
          )
      }.exceptionOrNull()

    assertThat(error).isNotNull()
    assertThat(error).hasMessageThat().contains("stream failed")
    assertThat(streamer.startedSessions).isEqualTo(1)
    assertThat(streamer.abortCalls).isEqualTo(1)
    assertThat(streamer.finishCalls).isEqualTo(0)
  }

  private class FakeSpeechStreamer : GraphPilotSpeechStreamer {
    var startedSessions = 0
    var finishCalls = 0
    var abortCalls = 0
    val appendedDeltas = mutableListOf<String>()

    override fun startSession() {
      startedSessions += 1
    }

    override fun appendDelta(delta: String) {
      appendedDeltas += delta
    }

    override fun finishStreaming() {
      finishCalls += 1
    }

    override fun abortStreaming() {
      abortCalls += 1
    }
  }
}
