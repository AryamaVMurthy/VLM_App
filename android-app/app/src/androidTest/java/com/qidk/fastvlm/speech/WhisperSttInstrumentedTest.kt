package com.qidk.fastvlm.speech

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.speech.WhisperSttEngine
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WhisperSttInstrumentedTest {
  companion object {
    private const val MAX_ACCEPTABLE_STT_MS = 15_000L
  }

  @Test
  fun transcribeTwoSecondWavClipWithinTenSeconds() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val configPath = stageSpeechConfig(context)
    val engine = WhisperSttEngine(context)
    val modelPath = engine.ensureInitialized(configPath)

    val wavFile = resolveWavFixture()
    val audio = readWavPcm16Mono16k(wavFile)
    val clipSamples = audio.copyOfRange(0, minOf(audio.size, 24_000))
    val firstStartedAt = System.currentTimeMillis()
    val firstTranscript = engine.transcribe(clipSamples)
    val firstElapsed = System.currentTimeMillis() - firstStartedAt

    val secondStartedAt = System.currentTimeMillis()
    val secondTranscript = engine.transcribe(clipSamples)
    val secondElapsed = System.currentTimeMillis() - secondStartedAt

    Log.i(
      "WhisperSttInstrumentedTest",
      "model=$modelPath fixture=${wavFile.absolutePath} clip_samples=${clipSamples.size} first_elapsed_ms=$firstElapsed second_elapsed_ms=$secondElapsed first='${firstTranscript.take(80)}' second='${secondTranscript.take(80)}'",
    )

    assertThat(firstElapsed).isLessThan(MAX_ACCEPTABLE_STT_MS)
    assertThat(secondElapsed).isLessThan(MAX_ACCEPTABLE_STT_MS)
    assertThat(firstTranscript).isNotNull()
    assertThat(firstTranscript.trim().isEmpty()).isFalse()
  }

  private fun stageSpeechConfig(context: Context): String {
    val configDir = File(context.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      error("Failed to create speech config dir '${configDir.absolutePath}'")
    }
    val target = File(configDir, "whisper_stt.json")
    context.assets.open("whisper_stt.json").use { input ->
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
      val size =
        ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
      val payloadOffset = offset + 8
      if (id == "fmt " && size >= 16 && payloadOffset + size <= bytes.size) {
        val fmt = ByteBuffer.wrap(bytes, payloadOffset, size).order(ByteOrder.LITTLE_ENDIAN)
        val audioFormat = fmt.short.toInt() and 0xFFFF
        numChannels = fmt.short.toInt() and 0xFFFF
        sampleRate = fmt.int
        fmt.int // byte rate
        fmt.short // block align
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
