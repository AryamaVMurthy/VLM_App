package com.qidk.fastvlm.core.speech

import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.config.JsonCodec
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class WhisperModelConfigLoaderTest {

  private val loader = WhisperModelConfigLoader(JsonCodec.instance)

  @Test
  fun parsesPinnedWhisperConfig() {
    val config =
      loader.fromString(
        """
        {
          "model_id": "ggerganov/whisper.cpp",
          "artifact": "ggml-base.en-q5_1.bin",
          "hf_revision": "rev",
          "download_url": "https://example.com/model.bin",
          "sha256": "abc",
          "size_bytes": 123,
          "sample_rate_hz": 16000,
          "language": "en"
        }
        """.trimIndent(),
      )

    assertThat(config.artifact).isEqualTo("ggml-base.en-q5_1.bin")
    assertThat(config.sample_rate_hz).isEqualTo(16000)
    assertThat(config.language).isEqualTo("en")
  }

  @Test(expected = WhisperModelConfigException::class)
  fun rejectsNonEnglishLanguage() {
    loader.fromString(
      """
      {
        "model_id": "ggerganov/whisper.cpp",
        "artifact": "ggml-base.en-q5_1.bin",
        "hf_revision": "rev",
        "download_url": "https://example.com/model.bin",
        "sha256": "abc",
        "size_bytes": 123,
        "sample_rate_hz": 16000,
        "language": "hi"
      }
      """.trimIndent(),
    )
  }
}
