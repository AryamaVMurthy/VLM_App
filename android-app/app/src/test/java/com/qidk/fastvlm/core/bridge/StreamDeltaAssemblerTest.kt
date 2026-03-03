package com.qidk.fastvlm.core.bridge

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class StreamDeltaAssemblerTest {

  @Test
  fun chunkMode_streamAppendsWithoutCorruption() {
    val chunks = listOf("This ", "is ", "a ", "valid ", "response.")
    val assembled = assemble(chunks)

    assertThat(assembled).isEqualTo("This is a valid response.")
  }

  @Test
  fun cumulativeMode_emitsOnlyNewSuffix() {
    val chunks = listOf("H", "He", "Hel", "Hell", "Hello")
    val assembled = assemble(chunks)

    assertThat(assembled).isEqualTo("Hello")
  }

  @Test
  fun mixedMode_handlesChunkAndSnapshotTogether() {
    val chunks = listOf("The ", "answer ", "The answer ", "is ", "The answer is ", "clear.")
    val assembled = assemble(chunks)

    assertThat(assembled).isEqualTo("The answer is clear.")
  }

  private fun assemble(chunks: List<String>): String {
    val out = StringBuilder()
    for (chunk in chunks) {
      val delta = computeStreamingDeltaFromChunk(out.toString(), chunk)
      if (delta.isNotEmpty()) {
        out.append(delta)
      }
    }
    return out.toString()
  }
}
