package com.qidk.fastvlm.bridge

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.bridge.FastVlmNativeBridge
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.metrics.MetricsStore
import com.qidk.fastvlm.core.model.BackendTarget
import com.qidk.fastvlm.core.model.FastVlmModelConfig
import com.qidk.fastvlm.core.model.ModelConfigLoader
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.model.VqaRequest
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FastVlmBridgeInstrumentedTest {
  private val json = JsonCodec.instance

  @Test
  fun cpuRequestCompletesWithDoneOrError() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val configPath = stageRuntimeConfig(context)
    val config = ModelConfigLoader(json).fromFile(File(configPath))
    provisionModelFromLocalMirror(context, config)
    val imagePath = stageImageForTest(context)

    val bridge = FastVlmNativeBridge(context, MetricsStore(context, json))
    val init = bridge.nativeInit(configPath)
    assertThat(init.ok).isTrue()
    assertThat(init.preferred_backend).isEqualTo(BackendTarget.CPU)

    val request =
      VqaRequest(
        question = "Describe this image briefly.",
        image_path = imagePath,
        preferred_backend = BackendTarget.CPU,
        allow_fallback = false,
      )

    val done = CountDownLatch(1)
    var terminalType: VqaEventType? = null
    val requestId =
      bridge.nativeRunVqa(json.encodeToString(VqaRequest.serializer(), request)) { raw ->
        val event = json.decodeFromString(VqaEvent.serializer(), raw)
        when (event.type) {
          VqaEventType.DONE, VqaEventType.ERROR -> {
            terminalType = event.type
            done.countDown()
          }

          else -> Unit
        }
      }

    val completed = done.await(90, TimeUnit.SECONDS)
    assertThat(completed).isTrue()
    assertThat(terminalType).isAnyOf(VqaEventType.DONE, VqaEventType.ERROR)

    val metrics = bridge.nativeGetLastMetrics(requestId)
    Log.i("FastVlmBridgeInstrumentedTest", "metrics=$metrics")
    assertThat(metrics.contains("metrics_not_found")).isFalse()
    assertThat(metrics).contains("\"backend_status\"")
    assertThat(metrics).contains("\"backend_config_actual\":\"CPU\"")

    bridge.close()
  }

  private fun stageRuntimeConfig(context: Context): String {
    val configDir = File(context.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      error("Failed to create config dir '${configDir.absolutePath}'")
    }
    val target = File(configDir, "fastvlm_phase1.json")
    context.assets.open("fastvlm_phase1.json").use { input ->
      target.outputStream().use { output -> input.copyTo(output) }
    }
    return target.absolutePath
  }

  private fun provisionModelFromLocalMirror(context: Context, config: FastVlmModelConfig) {
    val source = File("/data/local/tmp/vlm_phase1/${config.artifact}")
    if (!source.exists()) {
      return
    }
    val targetDir = File(context.filesDir, "models/${config.hf_revision}")
    if (!targetDir.exists()) {
      targetDir.mkdirs()
    }
    val target = File(targetDir, config.artifact)
    if (!target.exists()) {
      source.copyTo(target, overwrite = false)
    }
  }

  private fun stageImageForTest(context: Context): String {
    val source = File("/data/local/tmp/vlm_phase1/image.jpg")
    assertThat(source.exists()).isTrue()
    val target = File(context.cacheDir, "instrumentation_image.jpg")
    source.copyTo(target, overwrite = true)
    return target.absolutePath
  }
}
