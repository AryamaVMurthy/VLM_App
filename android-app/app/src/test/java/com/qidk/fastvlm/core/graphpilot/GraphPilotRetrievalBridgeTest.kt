package com.qidk.fastvlm.core.graphpilot

import com.google.ai.edge.litertlm.Backend
import com.google.common.truth.Truth.assertThat
import java.util.Base64
import org.junit.Test

class GraphPilotRetrievalBridgeTest {
  @Test
  fun buildRetrievalDaemonRequest_gpuEncodesQueryAsBase64() {
    val request =
      buildRetrievalDaemonRequest(
        query = "ask not what your country can do for you",
        topK = 2,
        backend = Backend.GPU,
        requestId = 42L,
      )

    assertThat(request[0]).isEqualTo("GRAPHPILOT_RETRIEVAL_V1")
    assertThat(request[1]).isEqualTo("42")
    assertThat(request[3]).isEqualTo("2")
    val decoded = String(Base64.getDecoder().decode(request[2]), Charsets.UTF_8)
    assertThat(decoded).isEqualTo("ask not what your country can do for you")
  }

  @Test
  fun resolveLibraryDirs_gpuUsesPreparedGpuRuntime() {
    val result =
      resolveRetrievalLibraryDirs(
        backend = Backend.GPU,
        gpuRuntimeLibraryDir = "/tmp/gpu_runtime",
        npuRuntimeLibraryDir = "/tmp/npu_runtime",
      )

    assertThat(result.runtimeLibraryDir).isEqualTo("/tmp/gpu_runtime")
    assertThat(result.dispatchLibraryDir).isEmpty()
  }

  @Test
  fun resolveLibraryDirs_npuUsesDispatchDir() {
    val result =
      resolveRetrievalLibraryDirs(
        backend = Backend.NPU,
        gpuRuntimeLibraryDir = "/tmp/gpu_runtime",
        npuRuntimeLibraryDir = "/tmp/npu_runtime",
      )

    assertThat(result.runtimeLibraryDir).isEqualTo("/tmp/npu_runtime")
    assertThat(result.dispatchLibraryDir).isEqualTo("/tmp/npu_runtime")
  }

  @Test
  fun resolveLibraryDirs_cpuUsesNoRuntimeDir() {
    val result =
      resolveRetrievalLibraryDirs(
        backend = Backend.CPU,
        gpuRuntimeLibraryDir = "/tmp/gpu_runtime",
        npuRuntimeLibraryDir = "/tmp/npu_runtime",
      )

    assertThat(result.runtimeLibraryDir).isEmpty()
    assertThat(result.dispatchLibraryDir).isEmpty()
  }
}
