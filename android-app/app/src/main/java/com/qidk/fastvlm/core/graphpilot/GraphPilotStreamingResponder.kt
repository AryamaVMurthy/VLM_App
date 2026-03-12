package com.qidk.fastvlm.core.graphpilot

data class GraphPilotTextGenerationResult(
  val text: String,
  val firstTokenAtMs: Long?,
)

interface GraphPilotStreamingTextGenerator {
  suspend fun generateStreaming(
    prompt: String,
    maxOutputTokens: Int,
    onDelta: suspend (String) -> Unit,
  ): GraphPilotTextGenerationResult
}

interface GraphPilotSpeechStreamer {
  fun startSession()

  fun appendDelta(delta: String)

  fun finishStreaming()

  fun abortStreaming()
}

data class GraphPilotStreamingResponse(
  val text: String,
  val firstTokenOffsetMs: Long?,
)

class GraphPilotStreamingResponder(
  private val clockMs: () -> Long = { System.currentTimeMillis() },
) {
  suspend fun generateAndStream(
    prompt: String,
    maxOutputTokens: Int,
    generator: GraphPilotStreamingTextGenerator,
    speechStreamer: GraphPilotSpeechStreamer,
  ): GraphPilotStreamingResponse {
    val startedAtMs = clockMs()
    speechStreamer.startSession()
    return try {
      val result =
        generator.generateStreaming(prompt, maxOutputTokens) { delta ->
          val trimmed = delta.trim()
          if (trimmed.isNotEmpty()) {
            speechStreamer.appendDelta(trimmed)
          }
        }
      check(result.text.isNotBlank()) {
        "Streaming responder produced an empty response. Remediation: inspect the text-stage stream and model output."
      }
      speechStreamer.finishStreaming()
      GraphPilotStreamingResponse(
        text = result.text,
        firstTokenOffsetMs = result.firstTokenAtMs?.minus(startedAtMs),
      )
    } catch (t: Throwable) {
      speechStreamer.abortStreaming()
      throw t
    }
  }
}
