package com.qidk.fastvlm.core.model

import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.config.JsonCodec
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class ModelConfigLoaderTest {

  private val loader = ModelConfigLoader(JsonCodec.instance)

  @Test
  fun parsesPinnedSm8750Config() {
    val config =
      loader.fromString(
        """
        {
          "model_id": "litert-community/FastVLM-0.5B",
          "artifact": "FastVLM-0.5B.litertlm",
          "hf_revision": "rev",
          "download_url": "https://example.com/model.litertlm",
          "sha256": "abc",
          "size_bytes": 123,
          "qairt_version": "2.43.0",
          "supported_soc": "SM8750P",
          "supported_sdk_int": 35
        }
        """.trimIndent(),
      )

    assertThat(config.artifact).isEqualTo("FastVLM-0.5B.litertlm")
    assertThat(config.supported_soc).isEqualTo("SM8750P")
    assertThat(config.preferred_backend).isEqualTo(BackendTarget.CPU)
  }

  @Test
  fun parsesPinnedNpuConfigWithMirror() {
    val config =
      loader.fromString(
        """
        {
          "model_id": "local/FastVLM-0.5B-sm8750-runtime",
          "artifact": "FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm",
          "hf_revision": "local-sm8750-runtime",
          "sha256": "abc",
          "size_bytes": 123,
          "qairt_version": "2.43.0",
          "supported_soc": "SM8750P",
          "supported_sdk_int": 35,
          "preferred_backend": "NPU",
          "local_mirror_path": "/data/local/tmp/vlm_phase1/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm"
        }
        """.trimIndent(),
      )

    assertThat(config.preferred_backend).isEqualTo(BackendTarget.NPU)
    assertThat(config.local_mirror_path).contains("auxmaskrope_runtime")
  }

  @Test(expected = ModelConfigException::class)
  fun rejectsUnsupportedArtifact() {
    loader.fromString(
      """
      {
        "model_id": "litert-community/FastVLM-0.5B",
        "artifact": "FastVLM-0.5B.invalid.litertlm",
        "hf_revision": "rev",
        "download_url": "https://example.com/model.litertlm",
        "sha256": "abc",
        "size_bytes": 123,
        "qairt_version": "2.43.0",
        "supported_soc": "SM8750P",
        "supported_sdk_int": 35
      }
      """.trimIndent(),
    )
  }

  @Test(expected = ModelConfigException::class)
  fun rejectsNpuConfigWithoutMirror() {
    loader.fromString(
      """
      {
        "model_id": "litert-community/FastVLM-0.5B",
        "artifact": "FastVLM-0.5B.qualcomm.sm8750.litertlm",
        "hf_revision": "rev",
        "download_url": "https://example.com/model.litertlm",
        "sha256": "abc",
        "size_bytes": 123,
        "qairt_version": "2.43.0",
        "supported_soc": "SM8750P",
        "supported_sdk_int": 35,
        "preferred_backend": "NPU"
      }
      """.trimIndent(),
    )
  }
}
