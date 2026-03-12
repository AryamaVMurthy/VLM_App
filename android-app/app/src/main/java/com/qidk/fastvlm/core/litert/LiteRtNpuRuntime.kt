package com.qidk.fastvlm.core.litert

import android.content.Context
import android.system.ErrnoException
import android.system.Os
import android.util.Log
import java.io.File

private const val TAG = "LiteRtNpuRuntime"
private const val EXTERNAL_NPU_DISPATCH_DIR = "/data/local/tmp/vlm_phase1/dispatch_libs"
private const val EXTERNAL_NPU_HEXAGON_DIR = "/data/local/tmp/vlm_phase1/hexagon-v79"

object LiteRtNpuRuntime {
  @Volatile private var preparedDir: String? = null

  @Synchronized
  fun ensurePrepared(context: Context): String {
    preparedDir?.let { cached ->
      if (File(cached).exists()) {
        return cached
      }
      preparedDir = null
    }

    val nativeDir = context.applicationInfo.nativeLibraryDir
    val nativeDirFile = File(nativeDir)
    check(nativeDirFile.exists()) {
      "App native library directory is missing at '${nativeDirFile.absolutePath}'. Remediation: reinstall the APK and retry."
    }

    val litertLmJniLib = File(nativeDirFile, "liblitertlm_jni.so")
    check(litertLmJniLib.exists()) {
      "Missing LiteRT-LM JNI bridge library at '${litertLmJniLib.absolutePath}'. Remediation: build and package //kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni into app jniLibs."
    }
    val litertRuntimeLib = File(nativeDirFile, "libLiteRt.so")
    check(litertRuntimeLib.exists()) {
      "Missing LiteRT runtime library at '${litertRuntimeLib.absolutePath}'. Remediation: build and package @litert//litert/c:litert_runtime_c_api_so into app jniLibs."
    }

    val dispatchSourceDir = File(EXTERNAL_NPU_DISPATCH_DIR)
    check(dispatchSourceDir.exists()) {
      "Missing external Qualcomm runtime directory at '${dispatchSourceDir.absolutePath}'. Remediation: push the real SM8750 dispatch runtime bundle to ${EXTERNAL_NPU_DISPATCH_DIR}."
    }
    val hexagonSourceDir = File(EXTERNAL_NPU_HEXAGON_DIR)
    check(hexagonSourceDir.exists()) {
      "Missing external Hexagon runtime directory at '${hexagonSourceDir.absolutePath}'. Remediation: push the real SM8750 hexagon-v79 bundle to ${EXTERNAL_NPU_HEXAGON_DIR}."
    }

    val stagedRuntimeDir = stageRuntimeLibraries(context, listOf(nativeDirFile, dispatchSourceDir, hexagonSourceDir))
    val requiredRuntimeLibs =
      listOf(
        "libLiteRtDispatch_Qualcomm.so",
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnHtpPrepare.so",
        "libQnnHexagonSkel_dspApp.so",
        "libQnnHtpV79.so",
        "libQnnHtpV79Skel.so",
        "libQnnNetRunDirectV79Skel.so",
      )
    requiredRuntimeLibs.forEach { name ->
      val candidate = File(stagedRuntimeDir, name)
      check(candidate.exists()) {
        "Missing required staged NPU runtime library '${candidate.absolutePath}'. Remediation: repush the exact SM8750 runtime bundle to ${EXTERNAL_NPU_DISPATCH_DIR} and retry."
      }
    }

    val adspPath =
      "${stagedRuntimeDir.absolutePath};/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
    try {
      Os.setenv("ADSP_LIBRARY_PATH", adspPath, true)
      Os.setenv("LITERT_QNN_SIGNED_PD_SUPPORT", "1", true)
      Os.setenv("LITERT_QNN_DISABLE_DEVICE_PLATFORM_INFO", "1", true)
    } catch (e: ErrnoException) {
      throw IllegalStateException(
        "Failed to set the NPU runtime environment: ${e.message}. Remediation: verify app process permission and retry.",
        e,
      )
    }

    preparedDir = stagedRuntimeDir.absolutePath
    Log.i(TAG, "Prepared LiteRT NPU runtime at ${stagedRuntimeDir.absolutePath}")
    return stagedRuntimeDir.absolutePath
  }

  private fun stageRuntimeLibraries(context: Context, sourceDirs: List<File>): File {
    val stagedDir = File(context.filesDir, "npu_dispatch_libs")
    if (!stagedDir.exists() && !stagedDir.mkdirs()) {
      error(
        "Failed to create staged NPU runtime directory '${stagedDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val sourceLibsByName = linkedMapOf<String, File>()
    sourceDirs.forEach { sourceDir ->
      if (!sourceDir.exists()) {
        return@forEach
      }
      sourceDir
        .listFiles { file -> file.isFile && file.name.endsWith(".so") }
        ?.sortedBy { it.name }
        ?.forEach { file ->
          sourceLibsByName[file.name] = file
        }
    }
    val sourceLibs = sourceLibsByName.values.toList()
    check(sourceLibs.isNotEmpty()) {
      "No shared libraries found in the configured NPU runtime sources. Remediation: provision ${EXTERNAL_NPU_DISPATCH_DIR} and ${EXTERNAL_NPU_HEXAGON_DIR} before initializing the NPU path."
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
        "Failed to mark staged runtime library '${destination.absolutePath}' as readable."
      }
      check(destination.setExecutable(true, false)) {
        "Failed to mark staged runtime library '${destination.absolutePath}' as executable."
      }
    }

    stagedDir
      .listFiles { file -> file.isFile && file.name.endsWith(".so") }
      ?.filter { staged -> sourceLibs.none { source -> source.name == staged.name } }
      ?.forEach { stale -> stale.delete() }

    marker.writeText(fingerprint)
    return stagedDir
  }
}
