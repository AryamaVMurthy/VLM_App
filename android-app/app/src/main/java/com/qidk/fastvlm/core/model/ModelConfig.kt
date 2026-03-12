package com.qidk.fastvlm.core.model

import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class FastVlmModelConfig(
  val model_id: String,
  val artifact: String,
  val hf_revision: String,
  val download_url: String? = null,
  val sha256: String,
  val size_bytes: Long,
  val qairt_version: String,
  val supported_soc: String,
  val supported_sdk_int: Int,
  val preferred_backend: BackendTarget = BackendTarget.CPU,
  val local_mirror_path: String? = null,
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
    val supportedArtifacts =
      setOf(
        "FastVLM-0.5B.litertlm",
        "FastVLM-0.5B.qualcomm.sm8750.litertlm",
        "FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm",
      )
    if (config.artifact !in supportedArtifacts) {
      throw ModelConfigException(
        "Invalid artifact '${config.artifact}'. Supported artifacts: ${supportedArtifacts.sorted().joinToString()}.",
      )
    }
    if (config.preferred_backend == BackendTarget.NPU && config.local_mirror_path.isNullOrBlank()) {
      throw ModelConfigException(
        "NPU config for artifact '${config.artifact}' must declare local_mirror_path. Remediation: stage the real SM8750 NPU artifact on device and point local_mirror_path to it.",
      )
    }
    if (config.download_url.isNullOrBlank() && config.local_mirror_path.isNullOrBlank()) {
      throw ModelConfigException(
        "Model config for artifact '${config.artifact}' is missing both download_url and local_mirror_path. Remediation: provide a pinned download URL or an explicit local mirror path.",
      )
    }
    return config
  }
}
