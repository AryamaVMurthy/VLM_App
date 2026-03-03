package com.qidk.fastvlm.core.device

import android.content.Context
import android.os.Build
import androidx.core.content.pm.PackageInfoCompat
import com.qidk.fastvlm.core.model.DeviceInfo

private const val REQUIRED_SOC = "SM8750P"
private const val REQUIRED_SDK_INT = 35
private const val REQUIRED_ABI = "arm64-v8a"

data class DeviceCompatibilityResult(
  val isCompatible: Boolean,
  val deviceInfo: DeviceInfo,
  val failureMessage: String? = null,
  val remediation: String? = null,
)

class DeviceCompatibilityChecker(private val context: Context) {

  fun check(): DeviceCompatibilityResult {
    val deviceInfo = buildDeviceInfo(context)

    val problems = mutableListOf<String>()
    if (!deviceInfo.soc.equals(REQUIRED_SOC, ignoreCase = true)) {
      problems += "Unsupported SoC '${deviceInfo.soc}'. Required '${REQUIRED_SOC}'."
    }
    if (deviceInfo.sdkInt != REQUIRED_SDK_INT) {
      problems += "Unsupported SDK '${deviceInfo.sdkInt}'. Required '${REQUIRED_SDK_INT}' (Android 15)."
    }
    if (!deviceInfo.abi.contains(REQUIRED_ABI, ignoreCase = true)) {
      problems += "Unsupported ABI '${deviceInfo.abi}'. Required '${REQUIRED_ABI}'."
    }

    return if (problems.isEmpty()) {
      DeviceCompatibilityResult(
        isCompatible = true,
        deviceInfo = deviceInfo,
      )
    } else {
      DeviceCompatibilityResult(
        isCompatible = false,
        deviceInfo = deviceInfo,
        failureMessage = problems.joinToString(separator = " "),
        remediation =
          "Run this app only on a QIDK device with SoC ${REQUIRED_SOC}, Android 15 (SDK ${REQUIRED_SDK_INT}), and ${REQUIRED_ABI} userspace.",
      )
    }
  }

  companion object {
    fun buildDeviceInfo(context: Context): DeviceInfo {
      val packageInfo =
        context.packageManager.getPackageInfo(context.packageName, 0)
      val versionName = packageInfo.versionName ?: "unknown"
      val versionCode = PackageInfoCompat.getLongVersionCode(packageInfo)

      val abi = Build.SUPPORTED_ABIS.joinToString(",")
      return DeviceInfo(
        soc = Build.SOC_MODEL ?: "unknown",
        device = Build.DEVICE ?: "unknown",
        model = Build.MODEL ?: "unknown",
        fingerprint = Build.FINGERPRINT ?: "unknown",
        sdkInt = Build.VERSION.SDK_INT,
        abi = abi,
        appVersion = "$versionName($versionCode)",
      )
    }
  }
}
