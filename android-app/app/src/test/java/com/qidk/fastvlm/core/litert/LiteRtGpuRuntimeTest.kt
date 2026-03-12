package com.qidk.fastvlm.core.litert

import com.google.common.truth.Truth.assertThat
import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Test

class LiteRtGpuRuntimeTest {
  @Test
  fun stageRuntimeLibrariesCopiesRequiredLibrariesAndPrunesStaleFiles() {
    val root = createTempDirectory("litert-gpu-runtime-test").toFile()
    val source = File(root, "source").apply { mkdirs() }
    val staged = File(root, "staged").apply { mkdirs() }
    File(source, "libLiteRt.so").writeText("runtime")
    File(source, "libLiteRtGpuAccelerator.so").writeText("gpu")
    File(source, "libLiteRtOpenClAccelerator.so").writeText("opencl")
    File(source, "libLiteRtTopKOpenClSampler.so").writeText("sampler")
    File(source, "libGemmaModelConstraintProvider.so").writeText("constraint")
    File(staged, "stale.so").writeText("stale")

    val result = LiteRtGpuRuntime.stageRuntimeLibrariesForTest(staged, source)

    assertThat(result).isEqualTo(staged)
    assertThat(File(staged, "libLiteRt.so").readText()).isEqualTo("runtime")
    assertThat(File(staged, "libLiteRtGpuAccelerator.so").readText()).isEqualTo("gpu")
    assertThat(File(staged, "stale.so").exists()).isFalse()
    assertThat(File(staged, ".fingerprint").exists()).isTrue()
  }
}
