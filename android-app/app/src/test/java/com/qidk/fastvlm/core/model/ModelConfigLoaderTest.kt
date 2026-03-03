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
  }

  @Test(expected = ModelConfigException::class)
  fun rejectsNonSm8750Artifact() {
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
        "supported_sdk_int": 35
      }
      """.trimIndent(),
    )
  }
}
