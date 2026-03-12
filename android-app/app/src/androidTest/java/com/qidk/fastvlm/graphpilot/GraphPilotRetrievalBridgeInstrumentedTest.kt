package com.qidk.fastvlm.graphpilot

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litertlm.Backend
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.graphpilot.GraphPilotRetrievalBridge
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GraphPilotRetrievalBridgeInstrumentedTest {
  @Test
  fun cpuQueryReturnsJfkHit() = runBlocking {
    runQuerySmoke(Backend.CPU)
  }

  @Test
  fun gpuQueryReturnsJfkHit() = runBlocking {
    runQuerySmoke(Backend.GPU)
  }

  @Test
  fun npuQueryReturnsJfkHit() = runBlocking {
    runQuerySmoke(Backend.NPU)
  }

  private suspend fun runQuerySmoke(backend: Backend) {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val bridge = GraphPilotRetrievalBridge(context)
    val result =
      bridge.retrieve(
        "ask not what your country can do for you",
        topK = 2,
        backend = backend,
      )
    assertThat(result.accelerator.lowercase()).isEqualTo(backend.name.lowercase())
    assertThat(result.hits).isNotEmpty()
    assertThat(result.hits.first().doc_id).contains("jfk")
  }
}
