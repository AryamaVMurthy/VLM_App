package com.qidk.fastvlm.core.speech

import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class WhisperModelConfig(
  val model_id: String,
  val artifact: String,
  val hf_revision: String,
  val download_url: String,
  val sha256: String,
  val size_bytes: Long,
  val sample_rate_hz: Int,
  val language: String,
)

class WhisperModelConfigException(message: String) : IllegalStateException(message)

class WhisperModelConfigLoader(private val json: Json) {
  fun fromFile(file: File): WhisperModelConfig {
    if (!file.exists()) {
      throw WhisperModelConfigException(
        "Whisper config file not found at '${file.absolutePath}'. Remediation: ensure whisper_stt.json is copied into app-private storage before STT init.",
      )
    }
    return fromString(file.readText())
  }

  fun fromString(rawJson: String): WhisperModelConfig {
    val config = json.decodeFromString<WhisperModelConfig>(rawJson)
    if (!config.artifact.endsWith(".bin")) {
      throw WhisperModelConfigException(
        "Invalid Whisper artifact '${config.artifact}'. Expected a .bin whisper.cpp model file.",
      )
    }
    if (config.sample_rate_hz != 16_000) {
      throw WhisperModelConfigException(
        "Unsupported Whisper sample rate ${config.sample_rate_hz}. This app requires 16000 Hz mono input.",
      )
    }
    if (!config.language.equals("en", ignoreCase = true)) {
      throw WhisperModelConfigException(
        "Unsupported Whisper language '${config.language}'. This release is locked to English for latency and quality stability.",
      )
    }
    return config
  }
}
