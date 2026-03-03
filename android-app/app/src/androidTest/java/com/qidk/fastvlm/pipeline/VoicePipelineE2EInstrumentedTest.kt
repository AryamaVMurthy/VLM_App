package com.qidk.fastvlm.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.metrics.MetricsStore
import com.qidk.fastvlm.core.model.BackendTarget
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.model.VqaMetrics
import com.qidk.fastvlm.core.model.VqaRequest
import com.qidk.fastvlm.core.speech.AndroidTtsSpeaker
import com.qidk.fastvlm.core.speech.WhisperSttEngine
import com.qidk.fastvlm.core.bridge.FastVlmNativeBridge
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VoicePipelineE2EInstrumentedTest {
  @Test
  fun fullLocalPipeline_sttVlmTts_logsExactTimings() {
    runBlocking {
      val context = InstrumentationRegistry.getInstrumentation().targetContext
      val json = JsonCodec.instance

      val whisperConfigPath = stageConfig(context, "whisper_stt.json")
      val fastVlmConfigPath = stageConfig(context, "fastvlm_phase1.json")

      val stt = WhisperSttEngine(context)
      val bridge = FastVlmNativeBridge(context, MetricsStore(context, json))
      val tts = AndroidTtsSpeaker(context)

      try {
        val sttInitStartMs = System.currentTimeMillis()
        stt.ensureInitialized(whisperConfigPath)
        val sttInitMs = System.currentTimeMillis() - sttInitStartMs

        val sttWarmupMs = stt.warmup()
        val wav = resolveWavFixture()
        val pcm = readWavPcm16Mono16k(wav)
        val clip = pcm.copyOfRange(0, minOf(pcm.size, 24_000))

        val sttStartMs = System.currentTimeMillis()
        val transcript = stt.transcribe(clip).trim()
        val sttMs = System.currentTimeMillis() - sttStartMs
        assertThat(transcript.isNotEmpty()).isTrue()

        val vlmInitStartMs = System.currentTimeMillis()
        val initResult = bridge.nativeInit(fastVlmConfigPath)
        val vlmInitMs = System.currentTimeMillis() - vlmInitStartMs
        assertThat(initResult.ok).isTrue()

        val imageFile = createSyntheticImage(context)
        val done = CompletableDeferred<Unit>()
        val answer = StringBuilder()
        var firstTokenAtMs: Long? = null
        var metrics: VqaMetrics? = null

        val request =
          VqaRequest(
            question = transcript,
            image_path = imageFile.absolutePath,
            preferred_backend = BackendTarget.CPU,
            allow_fallback = false,
            frame_capture_ms = 0L,
            preprocess_ms = 0L,
          )

        val requestStartMs = System.currentTimeMillis()
        bridge.nativeRunVqa(
          json.encodeToString(VqaRequest.serializer(), request),
        ) { raw ->
          val event = json.decodeFromString(VqaEvent.serializer(), raw)
          when (event.type) {
            VqaEventType.TOKEN -> {
              if (firstTokenAtMs == null) {
                firstTokenAtMs = System.currentTimeMillis()
              }
              event.token?.let { answer.append(it) }
            }
            VqaEventType.METRICS -> {
              metrics = event.metrics
            }
            VqaEventType.DONE -> {
              done.complete(Unit)
            }
            VqaEventType.ERROR -> {
              val message = event.error?.message ?: "unknown_error"
              if (!done.isCompleted) {
                done.completeExceptionally(IllegalStateException("VLM failed: $message"))
              }
            }
            else -> Unit
          }
        }

        withTimeout(30_000L) {
          done.await()
        }
        val vlmWallMs = System.currentTimeMillis() - requestStartMs
        val responseText = answer.toString().trim()
        assertThat(responseText.isNotEmpty()).isTrue()

        val ttsInitStartMs = System.currentTimeMillis()
        tts.ensureInitialized()
        val ttsInitMs = System.currentTimeMillis() - ttsInitStartMs

        val ttsSpeakStartMs = System.currentTimeMillis()
        tts.speak(responseText.take(180))
        val ttsSpeakMs = System.currentTimeMillis() - ttsSpeakStartMs

        val metricsResult = metrics ?: error("Missing VLM metrics event.")
        val firstTokenMs = firstTokenAtMs?.minus(requestStartMs) ?: -1L

        Log.i(
          "VoicePipelineE2E",
          "PIPELINE_TIMINGS stt_init_ms=$sttInitMs stt_warmup_ms=$sttWarmupMs stt_transcribe_ms=$sttMs " +
            "vlm_init_ms=$vlmInitMs vlm_ttft_ms=$firstTokenMs vlm_prefill_ms=${metricsResult.stage_timings.prefill_ms} " +
            "vlm_decode_ms=${metricsResult.stage_timings.decode_ms} vlm_total_ms=${metricsResult.stage_timings.total_ms} " +
            "vlm_wall_ms=$vlmWallMs tts_init_ms=$ttsInitMs tts_speak_ms=$ttsSpeakMs " +
            "decode_tps=${metricsResult.token_stats.decode_toks_per_sec} output_tokens=${metricsResult.token_stats.output_tokens} " +
            "transcript='${transcript.take(80)}' answer='${responseText.take(120)}'",
        )
      } finally {
        runCatching { tts.release() }
        runCatching { bridge.close() }
        runCatching { stt.release() }
      }
    }
  }

  private fun stageConfig(context: Context, assetName: String): String {
    val configDir = File(context.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      error("Failed to create config dir '${configDir.absolutePath}'")
    }
    val target = File(configDir, assetName)
    context.assets.open(assetName).use { input ->
      target.outputStream().use { output -> input.copyTo(output) }
    }
    return target.absolutePath
  }

  private fun resolveWavFixture(): File {
    val candidates =
      listOf(
        File("/data/local/tmp/vlm_phase1/stt_sample.wav"),
        File("/sdcard/Download/stt_sample.wav"),
      )
    return candidates.firstOrNull { it.exists() }
      ?: error(
        "Missing wav fixture. Remediation: adb push a 16kHz mono PCM wav to /data/local/tmp/vlm_phase1/stt_sample.wav",
      )
  }

  private fun createSyntheticImage(context: Context): File {
    val out = File(context.cacheDir, "pipeline_e2e_input.jpg")
    val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    canvas.drawColor(Color.rgb(242, 246, 252))

    val titlePaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(25, 35, 52)
        textSize = 42f
      }
    val subPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(35, 60, 100)
        textSize = 30f
      }
    canvas.drawText("FASTVLM CPU", 56f, 180f, titlePaint)
    canvas.drawText("Instrumentation image input", 56f, 240f, subPaint)

    FileOutputStream(out).use { fos ->
      if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 92, fos)) {
        error("Failed to encode synthetic JPEG at '${out.absolutePath}'")
      }
    }
    bitmap.recycle()
    return out
  }

  private fun readWavPcm16Mono16k(file: File): FloatArray {
    val bytes = file.readBytes()
    require(bytes.size >= 44) { "Invalid wav: header too short (${bytes.size} bytes)" }
    require(String(bytes, 0, 4) == "RIFF") { "Invalid wav: missing RIFF header" }
    require(String(bytes, 8, 4) == "WAVE") { "Invalid wav: missing WAVE header" }

    var offset = 12
    var dataOffset = -1
    var dataSize = -1
    var numChannels = -1
    var sampleRate = -1
    var bitsPerSample = -1

    while (offset + 8 <= bytes.size) {
      val id = String(bytes, offset, 4)
      val size = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
      val payloadOffset = offset + 8
      if (id == "fmt " && size >= 16 && payloadOffset + size <= bytes.size) {
        val fmt = ByteBuffer.wrap(bytes, payloadOffset, size).order(ByteOrder.LITTLE_ENDIAN)
        val audioFormat = fmt.short.toInt() and 0xFFFF
        numChannels = fmt.short.toInt() and 0xFFFF
        sampleRate = fmt.int
        fmt.int
        fmt.short
        bitsPerSample = fmt.short.toInt() and 0xFFFF
        require(audioFormat == 1) { "Unsupported wav format=$audioFormat; require PCM(1)" }
      } else if (id == "data" && payloadOffset + size <= bytes.size) {
        dataOffset = payloadOffset
        dataSize = size
        break
      }
      offset = payloadOffset + size + (size and 1)
    }

    require(dataOffset >= 0 && dataSize > 0) { "Invalid wav: data chunk not found" }
    require(numChannels == 1) { "Unsupported channel count=$numChannels; require mono" }
    require(sampleRate == 16_000) { "Unsupported sample_rate=$sampleRate; require 16000Hz" }
    require(bitsPerSample == 16) { "Unsupported bit depth=$bitsPerSample; require 16-bit PCM" }

    val sampleCount = dataSize / 2
    val pcm = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
    val out = FloatArray(sampleCount)
    for (i in 0 until sampleCount) {
      out[i] = (pcm.short / 32767.0f).coerceIn(-1f, 1f)
    }
    return out
  }
}
