package com.qidk.fastvlm.core.orchestrator

import android.content.Context
import android.util.Log
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import com.qidk.fastvlm.core.bridge.FastVlmNativeBridge
import com.qidk.fastvlm.core.camera.CameraController
import com.qidk.fastvlm.core.camera.ImagePreprocessor
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.metrics.BenchmarkRunSummary
import com.qidk.fastvlm.core.metrics.MetricsStore
import com.qidk.fastvlm.core.model.InitResult
import com.qidk.fastvlm.core.model.TransitionTimings
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.model.VqaMetrics
import com.qidk.fastvlm.core.model.VqaRequest
import com.qidk.fastvlm.core.model.BackendTarget
import java.io.File
import java.util.UUID
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "VqaOrchestrator"
private const val CAMERA_ROTATION_COMPENSATION_DEGREES = 180
private const val PREVIEW_CACHE_MAX_AGE_MS = 10_000L
private const val PREVIEW_TARGET_LONGEST_EDGE = 512
private const val DEFAULT_MAX_OUTPUT_TOKENS = 160
private const val DEFAULT_LATENCY_MODE = "voice_low_latency"

private data class PreviewCacheEntry(
  val preprocessedFile: File,
  val updatedAtMs: Long,
)

private data class PreparedRequestImage(
  val imageFile: File,
  val imageSource: String,
  val imageAgeMs: Long,
  val frameCaptureMs: Long,
  val preprocessMs: Long,
  val frameCaptureDoneMs: Long,
  val preprocessDoneMs: Long,
)

data class VoicePrefillHandle(
  val voiceSessionId: String,
  val prefillStartedMs: Long,
  val prefillDoneMs: Long,
  val imageSource: String,
  val imageAgeMs: Long,
  val frameCaptureMs: Long,
  val preprocessMs: Long,
  val frameCaptureDoneMs: Long,
  val preprocessDoneMs: Long,
)

private data class VoicePrefillContext(
  val handle: VoicePrefillHandle,
  val preparedImage: PreparedRequestImage,
)

class VqaOrchestrator(private val context: Context) {
  private val json = JsonCodec.instance
  private val metricsStore = MetricsStore(context, json)
  private val bridge = FastVlmNativeBridge(context, metricsStore)
  private val cameraController = CameraController(context)
  private val imagePreprocessor =
    ImagePreprocessor(
      extraRotationDegrees = CAMERA_ROTATION_COMPENSATION_DEGREES,
      targetLongestEdge = PREVIEW_TARGET_LONGEST_EDGE,
    )
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val previewCacheLock = Any()
  private val voicePrefillLock = Any()
  @Volatile private var previewCacheEntry: PreviewCacheEntry? = null
  private val voicePrefillContexts = mutableMapOf<String, VoicePrefillContext>()

  private val captureRoot by lazy { File(context.cacheDir, "captures") }
  private val preprocessRoot by lazy { File(context.cacheDir, "preprocessed") }

  suspend fun initialize(): InitResult {
    val configPath = ensureRuntimeConfigFile().absolutePath
    Log.i(TAG, "initialize: configPath=$configPath")
    return bridge.nativeInit(configPath)
  }

  fun bindCamera(
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    onReadyStateChanged: (ready: Boolean, reason: String?) -> Unit,
  ) {
    cameraController.bind(lifecycleOwner, previewView) { ready, reason ->
      if (ready) {
        scope.launch {
          runCatching {
            refreshPreviewCache("capture_fallback_prime_cache")
          }.onFailure { t ->
            Log.w(TAG, "preview cache prime failed: ${t.message}")
          }
        }
      }
      onReadyStateChanged(ready, reason)
    }
  }

  fun isCameraReady(): Boolean = cameraController.isReady()

  fun cameraNotReadyReason(): String? = cameraController.readinessFailureReason()

  suspend fun runSingleQuestion(
    question: String,
    sttFinalDoneMs: Long = 0L,
    voicePrefillSessionId: String? = null,
    onEvent: (VqaEvent) -> Unit,
  ): Long {
    val prefillContext =
      voicePrefillSessionId?.let { sessionId ->
        synchronized(voicePrefillLock) {
          voicePrefillContexts.remove(sessionId)
        }
      }
    if (voicePrefillSessionId != null && prefillContext == null) {
      throw IllegalStateException(
        "Voice prefill context '$voicePrefillSessionId' is missing. Remediation: restart hold-to-speak and release again.",
      )
    }
    val preparedImage = prefillContext?.preparedImage ?: resolveImageForRequest()
    val voicePrefillHandle = prefillContext?.handle
    val requestDispatchedMs = System.currentTimeMillis()

    val request =
      VqaRequest(
        question = question,
        image_path = preparedImage.imageFile.absolutePath,
        preferred_backend = BackendTarget.CPU,
        allow_fallback = false,
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS,
        image_source = preparedImage.imageSource,
        image_age_ms = preparedImage.imageAgeMs,
        latency_mode = DEFAULT_LATENCY_MODE,
        stt_final_done_ms = sttFinalDoneMs,
        vlm_request_dispatched_ms = requestDispatchedMs,
        frame_capture_done_ms = preparedImage.frameCaptureDoneMs,
        preprocess_done_ms = preparedImage.preprocessDoneMs,
        voice_session_id = voicePrefillHandle?.voiceSessionId,
        voice_prefill_started_ms = voicePrefillHandle?.prefillStartedMs ?: 0L,
        voice_prefill_done_ms = voicePrefillHandle?.prefillDoneMs ?: 0L,
        frame_capture_ms = preparedImage.frameCaptureMs,
        preprocess_ms = preparedImage.preprocessMs,
      )

    val requestId =
      submitRequest(request) { event ->
        onEvent(event)
        if (event.type == VqaEventType.DONE || event.type == VqaEventType.ERROR) {
          scope.launch {
            runCatching {
              refreshPreviewCache("capture_fallback_post_request_refresh")
            }.onFailure { t ->
              Log.w(TAG, "post-request preview cache refresh failed: ${t.message}")
            }
          }
        }
      }
    Log.i(
      TAG,
      "runSingleQuestion: requestId=$requestId image_source=${preparedImage.imageSource} image_age_ms=${preparedImage.imageAgeMs} frame_capture_ms=${preparedImage.frameCaptureMs} preprocess_ms=${preparedImage.preprocessMs} dispatched_ms=$requestDispatchedMs image=${preparedImage.imageFile.absolutePath}",
    )
    return requestId
  }

  suspend fun beginVoicePrefill(sessionId: String): VoicePrefillHandle {
    if (sessionId.isBlank()) {
      throw IllegalArgumentException(
        "Voice prefill session id cannot be blank. Remediation: pass a non-empty session id.",
      )
    }
    val preparedImage = refreshPreviewCache("press_hold_frame")
    val prefill =
      bridge.beginVoicePrefill(
        sessionId = sessionId,
        imagePath = preparedImage.imageFile.absolutePath,
        backend = BackendTarget.CPU,
        maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS,
      )
    val handle =
      VoicePrefillHandle(
        voiceSessionId = prefill.sessionId,
        prefillStartedMs = prefill.startedAtMs,
        prefillDoneMs = prefill.completedAtMs,
        imageSource = preparedImage.imageSource,
        imageAgeMs = preparedImage.imageAgeMs,
        frameCaptureMs = preparedImage.frameCaptureMs,
        preprocessMs = preparedImage.preprocessMs,
        frameCaptureDoneMs = preparedImage.frameCaptureDoneMs,
        preprocessDoneMs = preparedImage.preprocessDoneMs,
      )
    synchronized(voicePrefillLock) {
      voicePrefillContexts[sessionId] =
        VoicePrefillContext(
          handle = handle,
          preparedImage = preparedImage,
        )
    }
    return handle
  }

  fun cancelVoicePrefill(sessionId: String?): Boolean {
    val removed =
      if (sessionId.isNullOrBlank()) {
        false
      } else {
        synchronized(voicePrefillLock) {
          voicePrefillContexts.remove(sessionId) != null
        }
      }
    val bridgeCancelled = bridge.cancelVoiceSession(sessionId)
    return removed || bridgeCancelled
  }

  fun cancel(requestId: Long): Boolean = bridge.nativeCancel(requestId)

  suspend fun updateRequestTransitionTimings(
    requestId: Long,
    update: (TransitionTimings) -> TransitionTimings,
  ): VqaMetrics {
    return withContext(Dispatchers.IO) {
      metricsStore.updateRequestTransitionTimings(requestId, update)
    }
  }

  suspend fun primePreviewFrameCache() {
    withContext(Dispatchers.IO) {
      refreshPreviewCache("capture_fallback_manual_prime")
    }
  }

  suspend fun runTenRunBenchmark(
    question: String,
    onProgress: (Int, Int) -> Unit,
    onEvent: (VqaEvent) -> Unit,
  ): BenchmarkRunSummary {
    val frameStart = System.currentTimeMillis()
    val captured = cameraController.captureFrameToFile(captureRoot)
    val frameCaptureMs = System.currentTimeMillis() - frameStart

    val preprocessStart = System.currentTimeMillis()
    val preprocessed = imagePreprocessor.preprocessForFastVlm(captured, preprocessRoot)
    val preprocessMs = System.currentTimeMillis() - preprocessStart

    val requestIds = mutableListOf<Long>()
    val metrics = mutableListOf<VqaMetrics>()

    for (i in 1..10) {
      onProgress(i, 10)
      val request =
        VqaRequest(
          question = question,
          image_path = preprocessed.absolutePath,
          preferred_backend = BackendTarget.CPU,
          allow_fallback = false,
          frame_capture_ms = frameCaptureMs,
          preprocess_ms = preprocessMs,
        )

      val done = CompletableDeferred<Unit>()
      val requestId =
        submitRequest(request) { event ->
          onEvent(event)
          if (event.type == VqaEventType.DONE || event.type == VqaEventType.ERROR) {
            done.complete(Unit)
          }
        }

      requestIds += requestId
      Log.i(TAG, "benchmark run: ${i}/10 requestId=$requestId")
      done.await()

      val rawMetrics = bridge.nativeGetLastMetrics(requestId)
      if (!rawMetrics.contains("metrics_not_found")) {
        runCatching {
          json.decodeFromString(VqaMetrics.serializer(), rawMetrics)
        }.onSuccess { metrics += it }
          .onFailure { Log.w(TAG, "Failed to parse metrics for benchmark run $requestId: ${it.message}") }
      }
    }

    val summary =
      BenchmarkRunSummary(
        benchmark_id = "benchmark_${UUID.randomUUID()}",
        created_at_ms = System.currentTimeMillis(),
        question = question,
        run_count = metrics.size,
        avg_ttft_ms = if (metrics.isEmpty()) 0.0 else metrics.map { it.token_stats.ttft_ms }.average(),
        avg_decode_toks_per_sec =
          if (metrics.isEmpty()) 0.0 else metrics.map { it.token_stats.decode_toks_per_sec }.average(),
        avg_total_ms = if (metrics.isEmpty()) 0.0 else metrics.map { it.stage_timings.total_ms }.average(),
        requests = requestIds,
      )

    metricsStore.persistBenchmarkSummary(summary)
    return summary
  }

  fun shutdown() {
    scope.cancel()
    synchronized(voicePrefillLock) {
      voicePrefillContexts.keys.toList().forEach { sessionId ->
        runCatching { bridge.cancelVoiceSession(sessionId) }
      }
      voicePrefillContexts.clear()
    }
    cameraController.shutdown()
    bridge.close()
  }

  fun exportLatestBenchmarkReport(): Pair<File, File> = metricsStore.exportLatestBenchmarkReport()

  private suspend fun submitRequest(request: VqaRequest, onEvent: (VqaEvent) -> Unit): Long =
    withContext(Dispatchers.IO) {
      val requestJson = json.encodeToString(VqaRequest.serializer(), request)
      bridge.nativeRunVqa(requestJson) { rawEvent ->
        runCatching {
          json.decodeFromString(VqaEvent.serializer(), rawEvent)
        }.onSuccess(onEvent)
          .onFailure {
            Log.e(TAG, "Failed to decode VqaEvent: ${it.message}; raw=$rawEvent")
          }
      }
    }

  private fun ensureRuntimeConfigFile(): File {
    val configDir = File(context.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create config dir '${configDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val configFile = File(configDir, "fastvlm_phase1.json")
    context.assets.open("fastvlm_phase1.json").use { input ->
      configFile.outputStream().use { output ->
        input.copyTo(output)
      }
    }
    return configFile
  }

  private suspend fun resolveImageForRequest(): PreparedRequestImage {
    val now = System.currentTimeMillis()
    val cacheSnapshot =
      synchronized(previewCacheLock) {
        previewCacheEntry
      }
    if (cacheSnapshot != null && cacheSnapshot.preprocessedFile.exists()) {
      val ageMs = now - cacheSnapshot.updatedAtMs
      if (ageMs <= PREVIEW_CACHE_MAX_AGE_MS) {
        return PreparedRequestImage(
          imageFile = cacheSnapshot.preprocessedFile,
          imageSource = "preview_cache",
          imageAgeMs = ageMs,
          frameCaptureMs = 0L,
          preprocessMs = 0L,
          frameCaptureDoneMs = cacheSnapshot.updatedAtMs,
          preprocessDoneMs = cacheSnapshot.updatedAtMs,
        )
      }
      Log.i(TAG, "resolveImageForRequest: preview cache stale age_ms=$ageMs; refreshing")
      return refreshPreviewCache("capture_fallback_stale_cache")
    }
    return refreshPreviewCache("capture_fallback_no_cache")
  }

  private suspend fun refreshPreviewCache(imageSource: String): PreparedRequestImage {
    val frameStartMs = System.currentTimeMillis()
    val (captured, effectiveImageSource) =
      runCatching {
        cameraController.capturePreviewBitmapToFile(captureRoot)
      }.map { file ->
        file to imageSource
      }.getOrElse { previewError ->
        Log.w(TAG, "preview bitmap capture failed; using image capture fallback: ${previewError.message}")
        val fallbackSource = "${imageSource}_preview_bitmap_unavailable"
        cameraController.captureFrameToFile(captureRoot) to fallbackSource
      }
    val frameCaptureDoneMs = System.currentTimeMillis()
    val frameCaptureMs = frameCaptureDoneMs - frameStartMs

    val preprocessStartMs = System.currentTimeMillis()
    val preprocessed = imagePreprocessor.preprocessForFastVlm(captured, preprocessRoot)
    val preprocessDoneMs = System.currentTimeMillis()
    val preprocessMs = preprocessDoneMs - preprocessStartMs
    synchronized(previewCacheLock) {
      previewCacheEntry = PreviewCacheEntry(preprocessedFile = preprocessed, updatedAtMs = preprocessDoneMs)
    }
    return PreparedRequestImage(
      imageFile = preprocessed,
      imageSource = effectiveImageSource,
      imageAgeMs = 0L,
      frameCaptureMs = frameCaptureMs,
      preprocessMs = preprocessMs,
      frameCaptureDoneMs = frameCaptureDoneMs,
      preprocessDoneMs = preprocessDoneMs,
    )
  }
}
