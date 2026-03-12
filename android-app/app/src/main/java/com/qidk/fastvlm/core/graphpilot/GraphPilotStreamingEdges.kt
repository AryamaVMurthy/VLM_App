package com.qidk.fastvlm.core.graphpilot

private const val GRAPH_PILOT_AUDIO_SAMPLE_RATE_HZ = 16_000

data class GraphPilotAsrPlannerPipelineResult(
  val transcript: String,
  val plannerText: String,
  val asrElapsedMs: Long,
  val plannerElapsedMs: Long,
  val plannerFirstPartialOffsetMs: Long?,
  val asrChunkCount: Int,
  val plannerInvocationCount: Int,
)

internal fun splitMono16kPcmIntoChunks(
  audioData: FloatArray,
  chunkSizeMs: Int,
  sampleRateHz: Int = GRAPH_PILOT_AUDIO_SAMPLE_RATE_HZ,
): List<FloatArray> {
  require(chunkSizeMs > 0) { "chunkSizeMs must be > 0" }
  require(sampleRateHz > 0) { "sampleRateHz must be > 0" }
  if (audioData.isEmpty()) {
    return emptyList()
  }
  val samplesPerChunk = maxOf(1, chunkSizeMs * sampleRateHz / 1000)
  val chunks = mutableListOf<FloatArray>()
  var offset = 0
  while (offset < audioData.size) {
    val endExclusive = minOf(audioData.size, offset + samplesPerChunk)
    chunks += audioData.copyOfRange(offset, endExclusive)
    offset = endExclusive
  }
  return chunks
}

internal suspend fun runAsrPlannerPipeline(
  audioData: FloatArray,
  chunkSizeMs: Int?,
  totalStartMs: Long,
  clockMs: () -> Long = { System.currentTimeMillis() },
  transcribe: suspend (FloatArray) -> String,
  plan: suspend (String) -> String,
): GraphPilotAsrPlannerPipelineResult {
  require(audioData.isNotEmpty()) { "ASR audioData cannot be empty." }
  if (chunkSizeMs == null) {
    val asrStartedAtMs = clockMs()
    val transcript = transcribe(audioData).normalizeWhitespace()
    val asrElapsedMs = clockMs() - asrStartedAtMs
    check(transcript.isNotBlank()) { "ASR produced an empty transcript." }

    val plannerStartedAtMs = clockMs()
    val plannerText = plan(transcript).normalizeWhitespace()
    val plannerElapsedMs = clockMs() - plannerStartedAtMs
    check(plannerText.isNotBlank()) { "Planner produced an empty plan." }

    return GraphPilotAsrPlannerPipelineResult(
      transcript = transcript,
      plannerText = plannerText,
      asrElapsedMs = asrElapsedMs,
      plannerElapsedMs = plannerElapsedMs,
      plannerFirstPartialOffsetMs = null,
      asrChunkCount = 1,
      plannerInvocationCount = 1,
    )
  }

  val chunks = splitMono16kPcmIntoChunks(audioData, chunkSizeMs)
  check(chunks.isNotEmpty()) { "Chunked ASR requires at least one audio chunk." }

  var asrElapsedMs = 0L
  var plannerElapsedMs = 0L
  var plannerFirstPartialOffsetMs: Long? = null
  var plannerInvocationCount = 0
  val transcriptParts = mutableListOf<String>()
  var latestPlannerText = ""

  for (chunk in chunks) {
    val asrStartedAtMs = clockMs()
    val chunkTranscript = transcribe(chunk).normalizeWhitespace()
    asrElapsedMs += clockMs() - asrStartedAtMs
    if (chunkTranscript.isBlank()) {
      continue
    }

    transcriptParts += chunkTranscript
    val mergedTranscript = transcriptParts.joinToString(separator = " ").normalizeWhitespace()

    val plannerStartedAtMs = clockMs()
    val partialPlannerText = plan(mergedTranscript).normalizeWhitespace()
    plannerElapsedMs += clockMs() - plannerStartedAtMs
    check(partialPlannerText.isNotBlank()) {
      "Planner produced an empty partial plan for chunked ASR transcript."
    }
    plannerInvocationCount += 1
    latestPlannerText = partialPlannerText
    if (plannerFirstPartialOffsetMs == null) {
      plannerFirstPartialOffsetMs = clockMs() - totalStartMs
    }
  }

  val transcript = transcriptParts.joinToString(separator = " ").normalizeWhitespace()
  check(transcript.isNotBlank()) { "Chunked ASR produced an empty transcript." }
  check(latestPlannerText.isNotBlank()) { "Chunked ASR planner pipeline produced an empty plan." }

  return GraphPilotAsrPlannerPipelineResult(
    transcript = transcript,
    plannerText = latestPlannerText,
    asrElapsedMs = asrElapsedMs,
    plannerElapsedMs = plannerElapsedMs,
    plannerFirstPartialOffsetMs = plannerFirstPartialOffsetMs,
    asrChunkCount = chunks.size,
    plannerInvocationCount = plannerInvocationCount,
  )
}

private fun String.normalizeWhitespace(): String {
  return trim().replace(Regex("\\s+"), " ")
}
