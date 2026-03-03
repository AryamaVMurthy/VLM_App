package com.qidk.fastvlm.core.speech

import android.content.Context
import android.util.Log
import com.qidk.fastvlm.core.config.JsonCodec
import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.OkHttpClient

class WhisperSttEngine(
  private val context: Context,
  private val httpClient: OkHttpClient = OkHttpClient(),
) {
  companion object {
    private const val TAG = "WhisperSttEngine"
    private const val WARMUP_DURATION_MS = 500
    private const val SAMPLE_RATE_HZ = 16_000
  }

  private val lock = Mutex()
  private var whisperContext: WhisperContext? = null

  suspend fun ensureInitialized(configPath: String): String = lock.withLock {
    if (whisperContext != null) {
      return@withLock "ready"
    }

    val config = WhisperModelConfigLoader(JsonCodec.instance).fromFile(java.io.File(configPath))
    val modelPath = WhisperModelManager(context, httpClient).ensureModelReady(config).modelPath

    val ctx =
      runCatching {
        WhisperContext.createContextFromFile(modelPath)
      }.getOrElse { t ->
        throw IllegalStateException(
          "Whisper context initialization failed: ${t.message}. Remediation: verify model integrity and native whisper library packaging.",
          t,
        )
      }

    whisperContext = ctx
    return@withLock modelPath
  }

  suspend fun warmup(): Long = lock.withLock {
    val ctx =
      whisperContext
        ?: throw IllegalStateException(
          "Whisper STT is not initialized. Remediation: run speech initialization before warmup.",
        )

    val warmupSamples = FloatArray(SAMPLE_RATE_HZ * WARMUP_DURATION_MS / 1000)
    val startedAt = System.currentTimeMillis()
    runCatching {
      transcribeInternal(ctx, warmupSamples)
    }.onFailure { t ->
      throw IllegalStateException(
        "Whisper warmup failed: ${t.message}. Remediation: verify model integrity and retry.",
        t,
      )
    }
    val elapsed = System.currentTimeMillis() - startedAt
    Log.i(TAG, "warmup completed in ${elapsed}ms")
    elapsed
  }

  suspend fun transcribe(audioData: FloatArray): String = lock.withLock {
    val ctx =
      whisperContext
        ?: throw IllegalStateException(
          "Whisper STT is not initialized. Remediation: run speech initialization before transcription.",
        )

    val startedAt = System.currentTimeMillis()
    val normalized =
      runCatching {
        transcribeInternal(ctx, audioData)
      }.getOrElse { t ->
        throw IllegalStateException(
          "Whisper transcription failed: ${t.message}. Remediation: retry with a clean 16kHz microphone capture.",
          t,
        )
      }
    val elapsed = System.currentTimeMillis() - startedAt
    Log.i(
      TAG,
      "transcribe completed in ${elapsed}ms samples=${audioData.size} duration_ms=${audioData.size * 1000 / SAMPLE_RATE_HZ}",
    )
    normalized
  }

  suspend fun release() = lock.withLock {
    val ctx = whisperContext ?: return@withLock
    runCatching { ctx.release() }
    whisperContext = null
  }

  private suspend fun transcribeInternal(ctx: WhisperContext, audioData: FloatArray): String {
    val raw = ctx.transcribeData(audioData, printTimestamp = false)
    return raw.replace("\n", " ").replace(Regex("\\s+"), " ").trim()
  }
}
