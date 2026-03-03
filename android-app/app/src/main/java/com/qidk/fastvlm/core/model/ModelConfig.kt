package com.qidk.fastvlm.core.model

import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class FastVlmModelConfig(
  val model_id: String,
  val artifact: String,
  val hf_revision: String,
  val download_url: String,
  val sha256: String,
  val size_bytes: Long,
  val qairt_version: String,
  val supported_soc: String,
  val supported_sdk_int: Int,
)

class ModelConfigException(message: String) : IllegalStateException(message)

class ModelConfigLoader(private val json: Json) {
  fun fromFile(file: File): FastVlmModelConfig {
    if (!file.exists()) {
      throw ModelConfigException(
        "Model config file not found at '${file.absolutePath}'. Remediation: ensure fastvlm_phase1.json is copied into app-private storage before init.",
      )
    }
    return fromString(file.readText())
  }

  fun fromString(rawJson: String): FastVlmModelConfig {
    val config = json.decodeFromString<FastVlmModelConfig>(rawJson)
    if (config.artifact != "FastVLM-0.5B.litertlm") {
      throw ModelConfigException(
        "Invalid artifact '${config.artifact}'. CPU release is locked to FastVLM-0.5B.litertlm only.",
      )
    }
    return config
  }
}
