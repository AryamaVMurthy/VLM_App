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
          "local_mirror_path": "/data/local/tmp/graphpilot_edge/whisper/ggml-base.en-q5_1.bin",
          "download_url": "https://example.com/model.bin",
          "sha256": "abc",
          "size_bytes": 123,
          "sample_rate_hz": 16000,
          "language": "en"
        }
        """.trimIndent(),
      )

    assertThat(config.artifact).isEqualTo("ggml-base.en-q5_1.bin")
    assertThat(config.local_mirror_path).contains("ggml-base.en-q5_1.bin")
    assertThat(config.sample_rate_hz).isEqualTo(16000)
    assertThat(config.language).isEqualTo("en")
  }

  @Test
  fun acceptsMirrorOnlyWhisperConfig() {
    val config =
      loader.fromString(
        """
        {
          "model_id": "ggerganov/whisper.cpp",
          "artifact": "ggml-base.en-q5_1.bin",
          "hf_revision": "rev",
          "local_mirror_path": "/data/local/tmp/graphpilot_edge/whisper/ggml-base.en-q5_1.bin",
          "sha256": "abc",
          "size_bytes": 123,
          "sample_rate_hz": 16000,
          "language": "en"
        }
        """.trimIndent(),
      )

    assertThat(config.download_url).isNull()
    assertThat(config.local_mirror_path).contains("ggml-base.en-q5_1.bin")
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
