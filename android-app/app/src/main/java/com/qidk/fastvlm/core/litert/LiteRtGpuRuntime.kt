package com.qidk.fastvlm.core.litert

import android.content.Context
import android.util.Log
import java.io.File

private const val GPU_TAG = "LiteRtGpuRuntime"

object LiteRtGpuRuntime {
  private val REQUIRED_GPU_RUNTIME_LIBS =
    listOf(
      "libLiteRt.so",
      "libLiteRtGpuAccelerator.so",
      "libLiteRtOpenClAccelerator.so",
      "libLiteRtTopKOpenClSampler.so",
      "libGemmaModelConstraintProvider.so",
    )

  @Volatile private var preparedDir: String? = null

  @Synchronized
  fun ensurePrepared(context: Context): String {
    preparedDir?.let { cached ->
      if (File(cached).exists()) {
        return cached
      }
      preparedDir = null
    }

    val nativeDirFile = File(context.applicationInfo.nativeLibraryDir)
    check(nativeDirFile.exists()) {
      "App native library directory is missing at '${nativeDirFile.absolutePath}'. Remediation: reinstall the APK and retry."
    }
    REQUIRED_GPU_RUNTIME_LIBS.forEach { name ->
      val candidate = File(nativeDirFile, name)
      check(candidate.exists()) {
        "Missing required GPU runtime library '${candidate.absolutePath}'. Remediation: package the LiteRT GPU accelerator bundle into app jniLibs and reinstall."
      }
    }

    val stagedDir = File(context.filesDir, "gpu_runtime_libs")
    val prepared = stageRuntimeLibrariesForTest(stagedDir, nativeDirFile)
    preparedDir = prepared.absolutePath
    Log.i(GPU_TAG, "Prepared LiteRT GPU runtime at ${prepared.absolutePath}")
    return prepared.absolutePath
  }

  internal fun stageRuntimeLibrariesForTest(stagedDir: File, sourceDir: File): File {
    if (!stagedDir.exists() && !stagedDir.mkdirs()) {
      error(
        "Failed to create staged GPU runtime directory '${stagedDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val sourceLibs = REQUIRED_GPU_RUNTIME_LIBS.map { name -> File(sourceDir, name) }
    sourceLibs.forEach { file ->
      check(file.exists()) {
        "Missing required GPU runtime library '${file.absolutePath}'. Remediation: install the matching LiteRT GPU runtime bundle before using the GPU backend."
      }
    }

    val fingerprint =
      buildString {
        sourceLibs.forEach { file ->
          append(file.name)
          append(':')
          append(file.length())
          append(':')
          append(file.lastModified())
          append('\n')
        }
      }
    val marker = File(stagedDir, ".fingerprint")
    if (marker.exists() && marker.readText() == fingerprint) {
      return stagedDir
    }

    sourceLibs.forEach { source ->
      val destination = File(stagedDir, source.name)
      source.copyTo(destination, overwrite = true)
      check(destination.setReadable(true, false)) {
        "Failed to mark staged GPU runtime library '${destination.absolutePath}' as readable."
      }
      check(destination.setExecutable(true, false)) {
        "Failed to mark staged GPU runtime library '${destination.absolutePath}' as executable."
      }
    }

    stagedDir
      .listFiles { file -> file.isFile && file.name.endsWith(".so") }
      ?.filter { staged -> sourceLibs.none { it.name == staged.name } }
      ?.forEach { stale -> stale.delete() }

    marker.writeText(fingerprint)
    return stagedDir
  }
}
