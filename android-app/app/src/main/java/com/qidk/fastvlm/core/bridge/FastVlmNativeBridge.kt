package com.qidk.fastvlm.core.bridge

import android.content.Context
import android.system.ErrnoException
import android.system.Os
import android.util.Base64
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.LogSeverity
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageSendOptions
import com.google.ai.edge.litertlm.SamplerConfig
import com.qidk.fastvlm.BuildConfig
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.device.DeviceCompatibilityChecker
import com.qidk.fastvlm.core.metrics.MetricsStore
import com.qidk.fastvlm.core.model.BackendStatus
import com.qidk.fastvlm.core.model.BackendTarget
import com.qidk.fastvlm.core.model.FallbackEvent
import com.qidk.fastvlm.core.model.FastVlmModelConfig
import com.qidk.fastvlm.core.model.InitResult
import com.qidk.fastvlm.core.model.ModelConfigLoader
import com.qidk.fastvlm.core.model.ModelManager
import com.qidk.fastvlm.core.model.PipelineStage
import com.qidk.fastvlm.core.model.StageTimings
import com.qidk.fastvlm.core.model.TokenStats
import com.qidk.fastvlm.core.model.TransitionTimings
import com.qidk.fastvlm.core.model.VqaError
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.model.VqaMetrics
import com.qidk.fastvlm.core.model.VqaRequest
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToLong
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import org.json.JSONObject

private const val TAG = "FastVlmNativeBridge"
private const val REQUEST_TIMEOUT_MS = 20_000L
private const val ROOT_DAEMON_HOST = "127.0.0.1"
private const val ROOT_DAEMON_PORT = 21909
private const val ROOT_DAEMON_CONNECT_TIMEOUT_MS = 1_500
private const val ROOT_DAEMON_REQUEST_PROTOCOL = "VLM_PHASE1"
private const val ROOT_DAEMON_DEVICE_DIR = "/data/local/tmp/vlm_phase1"
private const val ROOT_DAEMON_SETUP_TIMEOUT_MS = 15_000L
private const val ROOT_DAEMON_ASSET_HANDLER = "daemon/vlm_daemon_handler.sh"
private const val ROOT_DAEMON_ASSET_BINARY = "daemon/litert_lm_advanced_main"
private const val ENGINE_MAX_NUM_TOKENS = 768
private const val VLM_SYSTEM_INSTRUCTION =
  "Answer only from the provided image. If information is not visible, say so. " +
    "Keep the response brief and factual."
private const val VLM_SAMPLER_TOP_K = 16
private const val VLM_SAMPLER_TOP_P = 0.9
private const val VLM_SAMPLER_TEMPERATURE = 0.2
private const val VLM_STREAM_DELIVERY_WARN_MS = 250L
private const val ENABLE_RUNTIME_BENCHMARK = true
private const val DEV_DAEMON_OPT_IN_FLAG = "dev_enable_root_daemon_bridge.flag"
private const val VOICE_PREFILL_TIMEOUT_MS = 8_000L
private const val VOICE_PREFILL_WARMUP_PROMPT = "Observe this image context. Reply with OK."
private const val VOICE_PREFILL_WARMUP_MAX_TOKENS = 1

private data class RunningRequest(
  val jobId: Long,
  val cancel: () -> Unit,
)

private data class BridgeState(
  val config: FastVlmModelConfig,
  val modelPath: String,
)

private data class PrewarmedConversation(
  val backend: BackendTarget,
  val conversation: Conversation,
)

private data class ShellExecResult(
  val exitCode: Int,
  val stdout: String,
  val stderr: String,
)

private data class DaemonRunResult(
  val text: String,
  val done: Boolean,
  val prefillMs: Long,
  val decodeMs: Long,
  val ttftMs: Long?,
  val decodeTps: Double?,
  val firstTokenAt: Long?,
  val firstCallbackMessageMs: Long?,
  val firstDeltaEmittedMs: Long?,
  val sendMessageCalledMs: Long?,
  val doneCallbackMs: Long?,
)

private data class VoicePrefillSession(
  val sessionId: String,
  val backend: BackendTarget,
  val conversation: Conversation,
)

data class VoicePrefillStartResult(
  val sessionId: String,
  val startedAtMs: Long,
  val completedAtMs: Long,
)

class FastVlmNativeBridge(
  private val context: Context,
  private val metricsStore: MetricsStore,
  private val httpClient: OkHttpClient = OkHttpClient(),
  private val json: Json = JsonCodec.instance,
) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val requestCounter = AtomicLong(0)
  private val runMutex = Mutex()
  private val engineMutex = Mutex()
  private val daemonMutex = Mutex()
  private val conversationPrewarmMutex = Mutex()

  private val engines = mutableMapOf<BackendTarget, Engine>()
  private val runningRequests = ConcurrentHashMap<Long, RunningRequest>()
  private val lastMetrics = ConcurrentHashMap<Long, VqaMetrics>()
  private val cancelledRequests = ConcurrentHashMap.newKeySet<Long>()
  private val voicePrefillSessions = ConcurrentHashMap<String, VoicePrefillSession>()
  @Volatile private var prewarmedConversation: PrewarmedConversation? = null
  @Volatile private var npuLibrariesDir: String? = null

  @Volatile private var state: BridgeState? = null

  suspend fun nativeInit(configPath: String): InitResult {
    Log.i(TAG, "nativeInit: start configPath=$configPath")
    val compatibility = DeviceCompatibilityChecker(context).check()
    if (!compatibility.isCompatible) {
      Log.e(TAG, "nativeInit: compatibility failed: ${compatibility.failureMessage}")
      return InitResult(
        ok = false,
        message = compatibility.failureMessage ?: "Unsupported device",
        device_info = compatibility.deviceInfo,
      )
    }

    val configFile = File(configPath)
    val configLoader = ModelConfigLoader(json)
    val config =
      try {
        configLoader.fromFile(configFile)
      } catch (t: Throwable) {
        Log.e(TAG, "nativeInit: config parse failed", t)
        return InitResult(
          ok = false,
          message = t.message ?: "Failed to parse config",
          device_info = compatibility.deviceInfo,
        )
      }

    if (!compatibility.deviceInfo.soc.equals(config.supported_soc, ignoreCase = true)) {
      return InitResult(
        ok = false,
        message =
          "Config/device mismatch. Config supports '${config.supported_soc}' but device is '${compatibility.deviceInfo.soc}'.",
        device_info = compatibility.deviceInfo,
      )
    }
    if (compatibility.deviceInfo.sdkInt != config.supported_sdk_int) {
      return InitResult(
        ok = false,
        message =
          "Config/device mismatch. Config SDK is '${config.supported_sdk_int}' but device SDK is '${compatibility.deviceInfo.sdkInt}'.",
        device_info = compatibility.deviceInfo,
      )
    }

    val modelPath =
      try {
        val modelManager = ModelManager(context, httpClient, json)
        modelManager.ensureModelReady(config).modelPath
      } catch (t: Throwable) {
        Log.e(TAG, "nativeInit: model provisioning failed", t)
        return InitResult(
          ok = false,
          message = t.message ?: "Model provisioning failed",
          device_info = compatibility.deviceInfo,
        )
      }

    state = BridgeState(config = config, modelPath = modelPath)

    Engine.setNativeMinLogSeverity(LogSeverity.WARNING)
    val cpuSelfCheckFailure =
      runCatching {
        getOrCreateEngine(BackendTarget.CPU, modelPath)
      }.exceptionOrNull()
    if (cpuSelfCheckFailure != null) {
      state = null
      engines.clear()
      val failureMessage =
        "CPU self-check failed during LiteRT engine initialization: ${cpuSelfCheckFailure.message}. " +
          "Remediation: verify model compatibility and LiteRT runtime packaging."
      Log.e(TAG, "nativeInit: CPU self-check failed: $failureMessage", cpuSelfCheckFailure)
      return InitResult(
        ok = false,
        message = failureMessage,
        device_info = compatibility.deviceInfo,
      )
    }

    return InitResult(
      ok = true,
      model_path = modelPath,
      preferred_backend = BackendTarget.CPU,
      device_info = compatibility.deviceInfo,
      message =
        "Model is provisioned and CPU runtime self-check passed. CPU backend is the only enabled execution path in this build.",
    )
      .also {
        scheduleConversationPrewarm(modelPath = modelPath, backend = BackendTarget.CPU)
      }
  }

  fun nativeRunVqa(requestJson: String, callback: (String) -> Unit): Long {
    val currentState =
      state
        ?: throw IllegalStateException(
          "Bridge not initialized. Remediation: call nativeInit(...) and ensure it returns ok=true before running inference.",
        )

    val request =
      try {
        json.decodeFromString(VqaRequest.serializer(), requestJson)
      } catch (se: SerializationException) {
        throw IllegalArgumentException(
          "Invalid VqaRequest JSON: ${se.message}. Remediation: provide a valid request payload.",
          se,
        )
      }
    if (request.preferred_backend != BackendTarget.CPU) {
      throw IllegalArgumentException(
        "CPU-only build received preferred_backend='${request.preferred_backend}'. Remediation: set preferred_backend=CPU and allow_fallback=false.",
      )
    }

    val requestId = requestCounter.incrementAndGet()
    cancelledRequests.remove(requestId)
    Log.i(
      TAG,
      "nativeRunVqa: requestId=$requestId preferred_backend=${request.preferred_backend} allow_fallback=${request.allow_fallback}",
    )

    val job =
      scope.launch {
        runMutex.withLock {
          executeRequest(
            requestId = requestId,
            request = request,
            modelPath = currentState.modelPath,
            callback = callback,
          )
        }
      }

    runningRequests[requestId] = RunningRequest(jobId = requestId) {
      job.cancel()
    }

    return requestId
  }

  fun nativeCancel(requestId: Long): Boolean {
    val running = runningRequests[requestId] ?: return false
    cancelledRequests += requestId
    running.cancel.invoke()
    return true
  }

  fun nativeGetLastMetrics(requestId: Long): String {
    val metrics =
      lastMetrics[requestId]
        ?: return "{\"error\":\"metrics_not_found\",\"request_id\":${requestId}}"
    return json.encodeToString(VqaMetrics.serializer(), metrics)
  }

  @OptIn(ExperimentalApi::class)
  suspend fun beginVoicePrefill(
    sessionId: String,
    imagePath: String,
    backend: BackendTarget = BackendTarget.CPU,
    maxOutputTokens: Int = 160,
  ): VoicePrefillStartResult {
    if (sessionId.isBlank()) {
      throw IllegalArgumentException(
        "Voice prefill session id cannot be blank. Remediation: pass a non-empty session id from the hold-to-speak flow.",
      )
    }
    val currentState =
      state
        ?: throw IllegalStateException(
          "Bridge not initialized. Remediation: call nativeInit(...) and ensure it returns ok=true before starting voice prefill.",
        )
    if (backend != BackendTarget.CPU) {
      throw IllegalArgumentException(
        "CPU-only build received beginVoicePrefill backend='${backend.name}'. Remediation: pass backend=CPU.",
      )
    }
    val imageFile = File(imagePath)
    if (!imageFile.exists()) {
      throw IllegalStateException(
        "Voice prefill image is missing at '${imageFile.absolutePath}'. Remediation: capture preview frame before beginVoicePrefill.",
      )
    }

    cancelVoiceSession(sessionId)

    val engine = getOrCreateEngine(backend, currentState.modelPath)
    val conversation = acquireConversation(engine = engine, backend = backend)
    val warmupMaxTokens =
      maxOutputTokens
        .coerceAtLeast(1)
        .coerceAtMost(VOICE_PREFILL_WARMUP_MAX_TOKENS)
    val prefillMessage =
      Message.user(
        Contents.of(
          Content.ImageFile(imagePath),
          Content.Text(VOICE_PREFILL_WARMUP_PROMPT),
        )
      )
    val startedAtMs = System.currentTimeMillis()
    Log.i(
      TAG,
      "beginVoicePrefill: session_id=$sessionId backend=$backend image_path=$imagePath warmup_max_output_tokens=$warmupMaxTokens",
    )
    try {
      withTimeout(VOICE_PREFILL_TIMEOUT_MS) {
        conversation
          .sendMessageAsync(
            prefillMessage,
            MessageSendOptions(
              hasPendingMessage = false,
              maxOutputTokens = warmupMaxTokens,
            ),
          )
          .collect {
            // Warmup turn is intentionally ignored; it only primes multimodal state.
          }
      }
    } catch (cancelled: CancellationException) {
      runCatching { conversation.cancelProcess() }
      runCatching { conversation.close() }
      throw cancelled
    } catch (t: Throwable) {
      runCatching { conversation.cancelProcess() }
      runCatching { conversation.close() }
      throw IllegalStateException(
        "Voice prefill warmup failed for session '$sessionId': ${t.message}. Remediation: retry hold-to-speak and verify camera frame capture.",
        t,
      )
    }
    val completedAtMs = System.currentTimeMillis()
    Log.i(
      TAG,
      "beginVoicePrefill: ready session_id=$sessionId duration_ms=${completedAtMs - startedAtMs}",
    )

    voicePrefillSessions[sessionId] =
      VoicePrefillSession(
        sessionId = sessionId,
        backend = backend,
        conversation = conversation,
      )
    return VoicePrefillStartResult(
      sessionId = sessionId,
      startedAtMs = startedAtMs,
      completedAtMs = completedAtMs,
    )
  }

  fun cancelVoiceSession(sessionId: String?): Boolean {
    if (sessionId.isNullOrBlank()) {
      return false
    }
    val session = voicePrefillSessions.remove(sessionId) ?: return false
    runCatching { session.conversation.cancelProcess() }
    runCatching { session.conversation.close() }
    return true
  }

  fun close() {
    scope.cancel()
    voicePrefillSessions.values.forEach { session ->
      runCatching { session.conversation.cancelProcess() }
      runCatching { session.conversation.close() }
    }
    voicePrefillSessions.clear()
    runCatching { prewarmedConversation?.conversation?.close() }
    prewarmedConversation = null
    engines.values.forEach {
      runCatching { it.close() }
    }
    engines.clear()
  }

  private suspend fun executeRequest(
    requestId: Long,
    request: VqaRequest,
    modelPath: String,
    callback: (String) -> Unit,
  ) {
    val fallbackEvents = mutableListOf<FallbackEvent>()
    val requestStartMs = System.currentTimeMillis()

    emitEvent(
      callback,
      VqaEvent(
        type = VqaEventType.START,
        request_id = requestId,
        backend_status =
          BackendStatus(
            backend_config_requested = request.preferred_backend,
            backend_config_actual = request.preferred_backend,
          ),
      ),
    )

    try {
      val primaryMetrics =
        runAttempt(
          requestId = requestId,
          request = request,
          actualBackend = request.preferred_backend,
          modelPath = modelPath,
          fallbackEvents = fallbackEvents,
          callback = callback,
        )
      metricsStore.persistRequestMetrics(primaryMetrics)
      lastMetrics[requestId] = primaryMetrics
      emitEvent(callback, VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = primaryMetrics))
      emitEvent(callback, VqaEvent(type = VqaEventType.DONE, request_id = requestId))
      return
    } catch (firstFailure: Throwable) {
      if (isCancelledRequest(requestId, firstFailure)) {
        val cancelledError = cancelledError()
        val cancelledMetrics =
          buildFailureMetrics(
            requestId = requestId,
            request = request,
            actualBackend = request.preferred_backend,
            fallbackEvents = fallbackEvents,
            attemptStartMs = requestStartMs,
            error = cancelledError,
          )
        metricsStore.persistRequestMetrics(cancelledMetrics)
        lastMetrics[requestId] = cancelledMetrics
        emitEvent(callback, VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = cancelledMetrics))
        emitEvent(callback, VqaEvent(type = VqaEventType.ERROR, request_id = requestId, error = cancelledError))
        return
      }

      val primaryError =
        VqaError(
          stage = PipelineStage.DECODE,
          code = "primary_backend_failed",
          message = firstFailure.message ?: "Primary backend failed",
          remediation = "Enable allow_fallback=true or fix the preferred backend runtime on device.",
        )
      val primaryFailureMetrics =
        buildFailureMetrics(
          requestId = requestId,
          request = request,
          actualBackend = request.preferred_backend,
          fallbackEvents = fallbackEvents,
          attemptStartMs = requestStartMs,
          error = primaryError,
        )
      metricsStore.persistRequestMetrics(primaryFailureMetrics, attemptTag = "primary_failed")

      if (!request.allow_fallback || request.preferred_backend == BackendTarget.CPU) {
        metricsStore.persistRequestMetrics(primaryFailureMetrics)
        lastMetrics[requestId] = primaryFailureMetrics
        emitEvent(
          callback,
          VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = primaryFailureMetrics),
        )
        emitEvent(callback, VqaEvent(type = VqaEventType.ERROR, request_id = requestId, error = primaryError))
        return
      }

      val fallbackEvent =
        FallbackEvent(
          stage = PipelineStage.INIT,
          from_backend = request.preferred_backend,
          to_backend = BackendTarget.CPU,
          fallback_reason = firstFailure.message ?: "Preferred backend failed",
          timestamp_ms = System.currentTimeMillis(),
        )
      fallbackEvents += fallbackEvent
      Log.w(TAG, "executeRequest: fallback requestId=$requestId reason=${fallbackEvent.fallback_reason}")
      emitEvent(
        callback,
        VqaEvent(
          type = VqaEventType.FALLBACK,
          request_id = requestId,
          fallback = fallbackEvent,
          backend_status =
            BackendStatus(
              backend_config_requested = request.preferred_backend,
              backend_config_actual = BackendTarget.CPU,
            ),
        ),
      )

      val fallbackStartMs = System.currentTimeMillis()
      try {
        val fallbackMetrics =
          runAttempt(
            requestId = requestId,
            request = request,
            actualBackend = BackendTarget.CPU,
            modelPath = modelPath,
            fallbackEvents = fallbackEvents,
            callback = callback,
          )
        metricsStore.persistRequestMetrics(fallbackMetrics, attemptTag = "fallback_success")
        metricsStore.persistRequestMetrics(fallbackMetrics)
        lastMetrics[requestId] = fallbackMetrics
        emitEvent(callback, VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = fallbackMetrics))
        emitEvent(callback, VqaEvent(type = VqaEventType.DONE, request_id = requestId))
      } catch (secondFailure: Throwable) {
        if (isCancelledRequest(requestId, secondFailure)) {
          val cancelledError = cancelledError()
          val cancelledMetrics =
            buildFailureMetrics(
              requestId = requestId,
              request = request,
              actualBackend = BackendTarget.CPU,
              fallbackEvents = fallbackEvents,
              attemptStartMs = fallbackStartMs,
              error = cancelledError,
            )
          metricsStore.persistRequestMetrics(cancelledMetrics)
          lastMetrics[requestId] = cancelledMetrics
          emitEvent(
            callback,
            VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = cancelledMetrics),
          )
          emitEvent(callback, VqaEvent(type = VqaEventType.ERROR, request_id = requestId, error = cancelledError))
          return
        }

        val fallbackError =
          VqaError(
            stage = PipelineStage.DECODE,
            code = "fallback_backend_failed",
            message = secondFailure.message ?: "CPU fallback failed",
            remediation =
              "Verify model integrity and runtime libraries. If persistent, reinstall app and clear model cache.",
          )
        val fallbackFailureMetrics =
          buildFailureMetrics(
            requestId = requestId,
            request = request,
            actualBackend = BackendTarget.CPU,
            fallbackEvents = fallbackEvents,
            attemptStartMs = fallbackStartMs,
            error = fallbackError,
          )
        metricsStore.persistRequestMetrics(fallbackFailureMetrics, attemptTag = "fallback_failed")
        metricsStore.persistRequestMetrics(fallbackFailureMetrics)
        lastMetrics[requestId] = fallbackFailureMetrics
        emitEvent(
          callback,
          VqaEvent(type = VqaEventType.METRICS, request_id = requestId, metrics = fallbackFailureMetrics),
        )
        emitEvent(callback, VqaEvent(type = VqaEventType.ERROR, request_id = requestId, error = fallbackError))
      }
    } finally {
      runningRequests.remove(requestId)
      cancelledRequests.remove(requestId)
    }
  }

  private fun isCancelledRequest(requestId: Long, throwable: Throwable): Boolean {
    if (cancelledRequests.contains(requestId)) {
      return true
    }
    var current: Throwable? = throwable
    while (current != null) {
      if (current is CancellationException) {
        return true
      }
      current = current.cause
    }
    return false
  }

  private fun cancelledError(): VqaError {
    return VqaError(
      stage = PipelineStage.DECODE,
      code = "request_cancelled",
      message = "Request cancelled",
      remediation = "Retry with a new request when ready.",
    )
  }

  private fun buildFailureMetrics(
    requestId: Long,
    request: VqaRequest,
    actualBackend: BackendTarget,
    fallbackEvents: List<FallbackEvent>,
    attemptStartMs: Long,
    error: VqaError,
  ): VqaMetrics {
    val doneAtMs = System.currentTimeMillis()
    val totalMs = maxOf(1L, doneAtMs - attemptStartMs)
    return VqaMetrics(
      request_id = requestId,
      device_info = DeviceCompatibilityChecker.buildDeviceInfo(context),
      backend_status =
        BackendStatus(
          backend_config_requested = request.preferred_backend,
          backend_config_actual = actualBackend,
        ),
      stage_timings =
        StageTimings(
          frame_capture_ms = request.frame_capture_ms,
          preprocess_ms = request.preprocess_ms,
          prefill_ms = 0L,
          decode_ms = totalMs,
          total_ms = totalMs,
        ),
      token_stats =
        TokenStats(
          ttft_ms = totalMs,
          decode_toks_per_sec = 0.0,
          output_tokens = 0,
        ),
      fallback_events = fallbackEvents,
      transition_timings =
        buildTransitionTimings(
          request = request,
          vqaStartMs = attemptStartMs,
          conversationCreateStartMs = null,
          conversationCreateDoneMs = null,
          sendMessageCalledMs = null,
          firstCallbackMessageMs = null,
          firstDeltaEmittedMs = null,
          doneCallbackMs = doneAtMs,
        ),
      error = error,
    )
  }

  private fun isDevRootDaemonBridgeEnabled(): Boolean {
    if (!BuildConfig.DEBUG || !BuildConfig.ENABLE_DEV_ROOT_DAEMON_BRIDGE) {
      return false
    }
    val optInFile = File(context.filesDir, DEV_DAEMON_OPT_IN_FLAG)
    val enabled = optInFile.exists()
    if (!enabled) {
      Log.i(
        TAG,
        "Root daemon bridge disabled: create '${optInFile.absolutePath}' to opt-in for debug fallback testing.",
      )
    }
    return enabled
  }

  @OptIn(ExperimentalApi::class)
  private suspend fun runAttempt(
    requestId: Long,
    request: VqaRequest,
    actualBackend: BackendTarget,
    modelPath: String,
    fallbackEvents: List<FallbackEvent>,
    callback: (String) -> Unit,
  ): VqaMetrics {
    Log.i(
      TAG,
      "runAttempt: requestId=$requestId backend=$actualBackend " +
        "systemInstruction='${VLM_SYSTEM_INSTRUCTION.replace("\n", " ")}' " +
        "sampler_top_k=$VLM_SAMPLER_TOP_K top_p=$VLM_SAMPLER_TOP_P temperature=$VLM_SAMPLER_TEMPERATURE",
    )
    val attemptStart = System.currentTimeMillis()
    if (actualBackend == BackendTarget.NPU && isDevRootDaemonBridgeEnabled()) {
      return runViaRootDaemon(
        requestId = requestId,
        request = request,
        modelPath = modelPath,
        daemonBackend = actualBackend,
        fallbackEvents = fallbackEvents,
        callback = callback,
        attemptStart = attemptStart,
      )
    }
    val voiceSessionId = request.voice_session_id?.trim().orEmpty().ifBlank { null }
    val voicePrefillSession =
      voiceSessionId?.let { sessionId ->
        val session = voicePrefillSessions.remove(sessionId)
        if (session == null) {
          throw IllegalStateException(
            "Voice prefill session '$sessionId' is unavailable or stale. Remediation: restart hold-to-speak and release again.",
          )
        }
        if (session.backend != actualBackend) {
          runCatching { session.conversation.cancelProcess() }
          runCatching { session.conversation.close() }
          throw IllegalStateException(
            "Voice prefill session '$sessionId' backend mismatch. session_backend=${session.backend} request_backend=$actualBackend. Remediation: restart hold-to-speak on CPU path.",
          )
        }
        session
      }

    val (conversation, conversationCreateStartMs, conversationCreateDoneMs) =
      if (voicePrefillSession != null) {
        Triple(
          voicePrefillSession.conversation,
          request.voice_prefill_started_ms.takeIf { it > 0L } ?: attemptStart,
          request.voice_prefill_done_ms.takeIf { it > 0L } ?: attemptStart,
        )
      } else {
        val engine = getOrCreateEngine(actualBackend, modelPath)
        val conversationCreateStart = System.currentTimeMillis()
        val freshConversation = acquireConversation(engine = engine, backend = actualBackend)
        val conversationCreateDone = System.currentTimeMillis()
        Triple(freshConversation, conversationCreateStart, conversationCreateDone)
      }
    val text = StringBuilder()
    var firstTokenAt: Long? = null
    var firstCallbackMessageMs: Long? = null
    var firstDeltaEmittedMs: Long? = null
    var streamChunkCount = 0

    val message =
      if (voicePrefillSession != null) {
        Message.user(
          Contents.of(
            Content.Text(request.question),
          )
        )
      } else {
        Message.user(
          Contents.of(
            Content.ImageFile(request.image_path),
            Content.Text(request.question),
          )
        )
      }
    val sendOptions =
      MessageSendOptions(
        hasPendingMessage = false,
        maxOutputTokens = request.max_output_tokens.coerceAtLeast(1),
      )

    val existingCancel = runningRequests[requestId]?.cancel
    runningRequests[requestId] = RunningRequest(jobId = requestId) {
      runCatching { conversation.cancelProcess() }
      runCatching { existingCancel?.invoke() }
    }
    val sendMessageCalledMs = System.currentTimeMillis()

    try {
      withTimeout(REQUEST_TIMEOUT_MS) {
        conversation.sendMessageAsync(message, sendOptions).collect { partialMessage ->
          val callbackAtMs = System.currentTimeMillis()
          if (firstCallbackMessageMs == null) {
            firstCallbackMessageMs = callbackAtMs
          }

          val incoming = partialMessage.toString()
          val delta =
            computeStreamingDeltaFromChunk(
              emittedText = text.toString(),
              incoming = incoming,
            )
          if (delta.isEmpty()) {
            return@collect
          }

          if (firstTokenAt == null) {
            firstTokenAt = callbackAtMs
          }
          if (firstDeltaEmittedMs == null) {
            firstDeltaEmittedMs = callbackAtMs
            val streamDeliveryLagMs = durationBetween(firstCallbackMessageMs, firstDeltaEmittedMs)
            if (streamDeliveryLagMs != null && streamDeliveryLagMs > VLM_STREAM_DELIVERY_WARN_MS) {
              Log.w(
                TAG,
                "stream_delivery_lag_ms=$streamDeliveryLagMs request_id=$requestId backend=$actualBackend",
              )
            }
          }

          text.append(delta)
          streamChunkCount += 1
          if (streamChunkCount <= 12) {
            Log.v(
              TAG,
              "stream_chunk request_id=$requestId chunk_idx=$streamChunkCount incoming_len=${incoming.length} delta_len=${delta.length} delta_preview='${delta.take(80)}'",
            )
          }
          emitEvent(
            callback,
            VqaEvent(
              type = VqaEventType.TOKEN,
              request_id = requestId,
              token = delta,
              backend_status =
                BackendStatus(
                  backend_config_requested = request.preferred_backend,
                  backend_config_actual = actualBackend,
                ),
            ),
          )
        }
      }
    } catch (timeout: TimeoutCancellationException) {
      runCatching { conversation.cancelProcess() }
      runCatching { conversation.close() }
      scheduleConversationPrewarm(modelPath = modelPath, backend = actualBackend)
      throw IllegalStateException(
        "Inference timed out after ${REQUEST_TIMEOUT_MS}ms on backend=$actualBackend. Remediation: reduce prompt/image load or switch backend configuration.",
      )
    } catch (cancelled: CancellationException) {
      runCatching { conversation.cancelProcess() }
      runCatching { conversation.close() }
      scheduleConversationPrewarm(modelPath = modelPath, backend = actualBackend)
      throw cancelled
    } catch (t: Throwable) {
      runCatching { conversation.cancelProcess() }
      runCatching { conversation.close() }
      scheduleConversationPrewarm(modelPath = modelPath, backend = actualBackend)
      throw t
    }

    val now = System.currentTimeMillis()
    val totalMs = now - attemptStart
    val doneCallbackMs = now

    val benchmark =
      runCatching {
        conversation.getBenchmarkInfo()
      }.onFailure {
        Log.w(TAG, "Benchmark info unavailable for request=$requestId: ${it.message}")
      }.getOrNull()
    val benchmarkMarks = parseNativeBenchmarkMarks(benchmark?.markDurationsJson)

    runCatching { conversation.close() }
    scheduleConversationPrewarm(modelPath = modelPath, backend = actualBackend)

    val outputTokens = estimateTokenCount(text.toString())

    val prefillMs =
      benchmark
        ?.takeIf { it.lastPrefillTokensPerSecond > 0.0 }
        ?.let { ((it.lastPrefillTokenCount.toDouble() / it.lastPrefillTokensPerSecond) * 1000.0).roundToLong() }
        ?: 0L

    val decodeMs =
      benchmark
        ?.takeIf { it.lastDecodeTokensPerSecond > 0.0 }
        ?.let { ((it.lastDecodeTokenCount.toDouble() / it.lastDecodeTokensPerSecond) * 1000.0).roundToLong() }
        ?: maxOf(1L, totalMs - prefillMs)

    val ttftMs =
      benchmark
        ?.takeIf { it.timeToFirstTokenInSecond > 0.0 }
        ?.let { (it.timeToFirstTokenInSecond * 1000.0).roundToLong() }
        ?: (firstTokenAt?.minus(attemptStart) ?: totalMs)

    val decodeTps =
      benchmark?.lastDecodeTokensPerSecond
        ?.takeIf { it > 0.0 }
        ?: if (decodeMs > 0) outputTokens * 1000.0 / decodeMs else 0.0

    val deviceInfo = DeviceCompatibilityChecker.buildDeviceInfo(context)
    val transitionTimings =
      buildTransitionTimings(
        request = request,
        vqaStartMs = attemptStart,
        conversationCreateStartMs = conversationCreateStartMs,
        conversationCreateDoneMs = conversationCreateDoneMs,
        sendMessageCalledMs = sendMessageCalledMs,
        firstCallbackMessageMs = firstCallbackMessageMs,
        firstDeltaEmittedMs = firstDeltaEmittedMs,
        doneCallbackMs = doneCallbackMs,
        nativeMessageParseMs = benchmarkMarks["jni_message_parse"],
        nativeRenderPromptMs = benchmarkMarks["conversation_render_prompt"],
        nativeToInputDataMs = benchmarkMarks["conversation_to_input_data"],
        nativePreprocessContentsMs = benchmarkMarks["session_preprocess_contents"],
        nativeVisionEncodeMs = benchmarkMarks["vision_executor"],
        nativePrefillWaitMs = benchmarkMarks["conversation_prefill_wait"],
        nativeDecodeToFirstChunkMs = benchmarkMarks["conversation_decode_to_first_chunk"],
      )
    val metrics =
      VqaMetrics(
        request_id = requestId,
        device_info = deviceInfo,
        backend_status =
          BackendStatus(
            backend_config_requested = request.preferred_backend,
            backend_config_actual = actualBackend,
          ),
        stage_timings =
          StageTimings(
            frame_capture_ms = request.frame_capture_ms,
            preprocess_ms = request.preprocess_ms,
            prefill_ms = prefillMs,
            decode_ms = decodeMs,
            total_ms = totalMs,
          ),
        token_stats =
          TokenStats(
            ttft_ms = ttftMs,
            decode_toks_per_sec = decodeTps,
            output_tokens = outputTokens,
          ),
        fallback_events = fallbackEvents,
        transition_timings = transitionTimings,
      )

    return metrics
  }

  private suspend fun runViaRootDaemon(
    requestId: Long,
    request: VqaRequest,
    modelPath: String,
    daemonBackend: BackendTarget,
    fallbackEvents: List<FallbackEvent>,
    callback: (String) -> Unit,
    attemptStart: Long,
  ): VqaMetrics {
    val daemonResult =
      runCatching {
        runViaRootDaemonOnce(
          requestId = requestId,
          request = request,
          modelPath = modelPath,
          daemonBackend = daemonBackend,
          callback = callback,
        )
      }.getOrElse { firstFailure ->
        val shouldForceRestart = isDaemonRestartRequiredFailure(firstFailure)
        if (!isDaemonConnectivityFailure(firstFailure) && !shouldForceRestart) {
          throw firstFailure
        }
        Log.w(
          TAG,
          "runViaRootDaemon: daemon recovery attempt for requestId=$requestId backend=$daemonBackend forceRestart=$shouldForceRestart reason=${firstFailure.message}",
        )
        ensureRootDaemonRunning(forceRestart = shouldForceRestart)
        try {
          runViaRootDaemonOnce(
            requestId = requestId,
            request = request,
            modelPath = modelPath,
            daemonBackend = daemonBackend,
            callback = callback,
          )
        } catch (retryFailure: Throwable) {
          val daemonLogs = readRootDaemonLogs()
          throw IllegalStateException(
            "Root daemon bridge remained unavailable at ${ROOT_DAEMON_HOST}:${ROOT_DAEMON_PORT} after auto-start attempt. " +
              "First error='${firstFailure.message}'. Retry error='${retryFailure.message}'. " +
              "Daemon logs:\n$daemonLogs\n" +
              "Remediation: verify root access is available to the app process or provision daemon using scripts/start_root_vlm_daemon_adb.sh.",
            retryFailure,
          )
        }
      }

    if (!daemonResult.done) {
      throw IllegalStateException(
        "Root daemon stream ended before DONE for request ${requestId}. Remediation: inspect daemon logs and restart the daemon.",
      )
    }

    val now = System.currentTimeMillis()
    val totalMs = now - attemptStart
    val outputTokens = estimateTokenCount(daemonResult.text)
    val prefillMs = daemonResult.prefillMs
    val decodeMs = if (daemonResult.decodeMs > 0L) daemonResult.decodeMs else maxOf(1L, totalMs - prefillMs)
    val ttftMs = daemonResult.ttftMs ?: (daemonResult.firstTokenAt?.minus(attemptStart) ?: totalMs)
    val decodeTps =
      daemonResult.decodeTps ?: if (decodeMs > 0) outputTokens * 1000.0 / decodeMs else 0.0
    val transitionTimings =
      buildTransitionTimings(
        request = request,
        vqaStartMs = attemptStart,
        conversationCreateStartMs = attemptStart,
        conversationCreateDoneMs = attemptStart,
        sendMessageCalledMs = daemonResult.sendMessageCalledMs ?: attemptStart,
        firstCallbackMessageMs = daemonResult.firstCallbackMessageMs,
        firstDeltaEmittedMs = daemonResult.firstDeltaEmittedMs,
        doneCallbackMs = daemonResult.doneCallbackMs ?: now,
      )

    return VqaMetrics(
      request_id = requestId,
      device_info = DeviceCompatibilityChecker.buildDeviceInfo(context),
      backend_status =
        BackendStatus(
          backend_config_requested = request.preferred_backend,
          backend_config_actual = daemonBackend,
        ),
      stage_timings =
        StageTimings(
          frame_capture_ms = request.frame_capture_ms,
          preprocess_ms = request.preprocess_ms,
          prefill_ms = prefillMs,
          decode_ms = decodeMs,
          total_ms = totalMs,
        ),
      token_stats =
        TokenStats(
          ttft_ms = ttftMs,
          decode_toks_per_sec = decodeTps,
          output_tokens = outputTokens,
        ),
      fallback_events = fallbackEvents,
      transition_timings = transitionTimings,
    )
  }

  private fun runViaRootDaemonOnce(
    requestId: Long,
    request: VqaRequest,
    modelPath: String,
    daemonBackend: BackendTarget,
    callback: (String) -> Unit,
  ): DaemonRunResult {
    val socket = Socket()
    val text = StringBuilder()
    var done = false
    var firstTokenAt: Long? = null
    var firstCallbackMessageMs: Long? = null
    var firstDeltaEmittedMs: Long? = null
    var prefillMsFromBench = 0L
    var decodeMsFromBench = 0L
    var ttftMsFromBench: Long? = null
    var decodeTpsFromBench: Double? = null
    var sendMessageCalledMs: Long? = null
    var doneCallbackMs: Long? = null

    try {
      socket.connect(InetSocketAddress(ROOT_DAEMON_HOST, ROOT_DAEMON_PORT), ROOT_DAEMON_CONNECT_TIMEOUT_MS)
      socket.soTimeout = REQUEST_TIMEOUT_MS.toInt()
      val existingCancel = runningRequests[requestId]?.cancel
      runningRequests[requestId] = RunningRequest(jobId = requestId) {
        runCatching { socket.close() }
        runCatching { existingCancel?.invoke() }
      }

      val writer = BufferedWriter(OutputStreamWriter(socket.getOutputStream(), Charsets.UTF_8))
      val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))

      val questionBase64 =
        Base64.encodeToString(request.question.toByteArray(Charsets.UTF_8), Base64.NO_WRAP)
      writer.write("${ROOT_DAEMON_REQUEST_PROTOCOL}\n")
      writer.write("${requestId}\n")
      writer.write("${questionBase64}\n")
      writer.write("${request.image_path}\n")
      writer.write("${modelPath}\n")
      val requestedMaxTokens = request.max_output_tokens.coerceAtLeast(1)
      writer.write("${requestedMaxTokens}\n")
      writer.write("${daemonBackend.name.lowercase()}\n")
      sendMessageCalledMs = System.currentTimeMillis()
      writer.flush()

      while (true) {
        val line = reader.readLine() ?: break
        if (!line.startsWith("VLM_EVENT ")) {
          Log.v(TAG, "runViaRootDaemonOnce: ignored non-event line='$line'")
          continue
        }
        val payload = line.removePrefix("VLM_EVENT ").trim()
        if (payload.isEmpty()) {
          continue
        }
        val event =
          try {
            JSONObject(payload)
          } catch (t: Throwable) {
            throw IllegalStateException(
              "Root daemon returned malformed event JSON: '$payload'. Remediation: inspect daemon handler and runtime logs.",
              t,
            )
          }
        when (event.optString("type")) {
          "TOKEN" -> {
            if (firstCallbackMessageMs == null) {
              firstCallbackMessageMs = System.currentTimeMillis()
            }
            val chunk = event.optString("text")
            if (chunk.isNotEmpty()) {
              if (firstTokenAt == null) {
                firstTokenAt = System.currentTimeMillis()
              }
              if (firstDeltaEmittedMs == null) {
                firstDeltaEmittedMs = System.currentTimeMillis()
              }
              text.append(chunk)
              emitEvent(
                callback,
                VqaEvent(
                  type = VqaEventType.TOKEN,
                  request_id = requestId,
                  token = chunk,
                  backend_status =
                    BackendStatus(
                      backend_config_requested = request.preferred_backend,
                      backend_config_actual = daemonBackend,
                    ),
                ),
              )
            }
          }

          "BENCHMARK" -> {
            val ttftSec = event.optDouble("ttft_sec", Double.NaN)
            if (!ttftSec.isNaN() && ttftSec > 0.0) {
              ttftMsFromBench = (ttftSec * 1000.0).roundToLong()
            }
            val turns = event.optJSONArray("turns")
            if (turns != null && turns.length() > 0) {
              val first = turns.optJSONObject(0)
              if (first != null) {
                val prefill = first.optDouble("prefill_duration_ms", Double.NaN)
                if (!prefill.isNaN() && prefill >= 0.0) {
                  prefillMsFromBench = prefill.roundToLong()
                }
                val decode = first.optDouble("decode_duration_ms", Double.NaN)
                if (!decode.isNaN() && decode >= 0.0) {
                  decodeMsFromBench = decode.roundToLong()
                }
                val decodeTps = first.optDouble("decode_tokens_per_sec", Double.NaN)
                if (!decodeTps.isNaN() && decodeTps > 0.0) {
                  decodeTpsFromBench = decodeTps
                }
              }
            }
          }

          "ERROR" -> {
            val message = event.optString("message", "Root daemon returned ERROR event.")
            throw IllegalStateException(
              "Root daemon inference failed on backend=${daemonBackend.name}: ${message}. " +
                "Remediation: inspect daemon logs and runtime deployment.",
            )
          }

          "DONE" -> {
            done = true
            doneCallbackMs = System.currentTimeMillis()
          }
        }
      }
    } finally {
      runCatching { socket.close() }
    }

    return DaemonRunResult(
      text = text.toString(),
      done = done,
      prefillMs = prefillMsFromBench,
      decodeMs = decodeMsFromBench,
      ttftMs = ttftMsFromBench,
      decodeTps = decodeTpsFromBench,
      firstTokenAt = firstTokenAt,
      firstCallbackMessageMs = firstCallbackMessageMs,
      firstDeltaEmittedMs = firstDeltaEmittedMs,
      sendMessageCalledMs = sendMessageCalledMs,
      doneCallbackMs = doneCallbackMs,
    )
  }

  private suspend fun ensureRootDaemonRunning() {
    ensureRootDaemonRunning(forceRestart = false)
  }

  private suspend fun ensureRootDaemonRunning(forceRestart: Boolean) {
    if (!forceRestart && isRootDaemonReachable()) {
      return
    }
    daemonMutex.withLock {
      if (!forceRestart && isRootDaemonReachable()) {
        return
      }
      val stagedDir = File(context.filesDir, "daemon_stage")
      val stagedHandler = stageDaemonAsset(ROOT_DAEMON_ASSET_HANDLER, File(stagedDir, "vlm_daemon_handler.sh"), executable = true)
      val stagedBinary = stageDaemonAsset(ROOT_DAEMON_ASSET_BINARY, File(stagedDir, "litert_lm_advanced_main"), executable = true)

      val startScript =
        buildRootDaemonStartScript(
          stagedHandlerPath = stagedHandler.absolutePath,
          stagedBinaryPath = stagedBinary.absolutePath,
          nativeLibDir = context.applicationInfo.nativeLibraryDir,
          forceRestart = forceRestart,
        )

      val startResult = runRootCommand(startScript, ROOT_DAEMON_SETUP_TIMEOUT_MS)
      if (startResult.exitCode != 0) {
        throw IllegalStateException(
          "Failed to start root daemon from app process. exit=${startResult.exitCode}, stdout='${startResult.stdout}', stderr='${startResult.stderr}'. " +
            "Remediation: ensure the device root binary is accessible to app domain or start daemon manually via adb script.",
        )
      }

      if (!isRootDaemonReachable()) {
        val daemonLogs = readRootDaemonLogs()
        throw IllegalStateException(
          "Root daemon start command succeeded but daemon port ${ROOT_DAEMON_PORT} is still unreachable. " +
            "Daemon logs:\n$daemonLogs\n" +
            "Remediation: verify toybox nc supports -L on this build and inspect SELinux denials.",
        )
      }

      Log.i(
        TAG,
        "ensureRootDaemonRunning: daemon is reachable on ${ROOT_DAEMON_HOST}:${ROOT_DAEMON_PORT} forceRestart=$forceRestart",
      )
    }
  }

  private fun isDaemonConnectivityFailure(throwable: Throwable): Boolean {
    return throwable is IOException ||
      (throwable is IllegalStateException &&
        throwable.message?.contains("bridge unavailable", ignoreCase = true) == true)
  }

  private fun isDaemonRestartRequiredFailure(throwable: Throwable): Boolean {
    if (throwable !is IllegalStateException) {
      return false
    }
    val message = throwable.message ?: return false
    return message.contains("missing native libs directory", ignoreCase = true)
  }

  private fun isRootDaemonReachable(): Boolean {
    val probe = Socket()
    return try {
      probe.connect(InetSocketAddress(ROOT_DAEMON_HOST, ROOT_DAEMON_PORT), 250)
      true
    } catch (_: IOException) {
      false
    } finally {
      runCatching { probe.close() }
    }
  }

  private fun stageDaemonAsset(assetPath: String, destination: File, executable: Boolean): File {
    val parent = destination.parentFile
    if (parent != null && !parent.exists() && !parent.mkdirs()) {
      throw IllegalStateException(
        "Failed to create daemon staging directory '${parent.absolutePath}'. Remediation: free app storage and retry.",
      )
    }
    try {
      context.assets.open(assetPath).use { input ->
        FileOutputStream(destination).use { output ->
          input.copyTo(output)
        }
      }
    } catch (io: IOException) {
      throw IllegalStateException(
        "Failed to stage daemon asset '$assetPath' to '${destination.absolutePath}': ${io.message}. " +
          "Remediation: reinstall APK with daemon assets bundled.",
        io,
      )
    }
    if (executable && !destination.setExecutable(true, false)) {
      throw IllegalStateException(
        "Failed to mark '${destination.absolutePath}' as executable. Remediation: verify app filesystem permissions and retry.",
      )
    }
    if (!destination.setReadable(true, false)) {
      throw IllegalStateException(
        "Failed to mark '${destination.absolutePath}' as world-readable for root daemon provisioning. Remediation: verify app filesystem permissions.",
      )
    }
    return destination
  }

  private fun buildRootDaemonStartScript(
    stagedHandlerPath: String,
    stagedBinaryPath: String,
    nativeLibDir: String,
    forceRestart: Boolean,
  ): String {
    val deviceHandlerPath = "${ROOT_DAEMON_DEVICE_DIR}/vlm_daemon_handler.sh"
    val deviceBinaryPath = "${ROOT_DAEMON_DEVICE_DIR}/litert_lm_advanced_main"
    val stdoutPath = "${ROOT_DAEMON_DEVICE_DIR}/vlm_daemon_stdout.log"
    val stderrPath = "${ROOT_DAEMON_DEVICE_DIR}/vlm_daemon_stderr.log"
    val forceRestartSnippet =
      if (forceRestart) {
        """
      pkill -f ${shQuote("toybox nc -s 127.0.0.1 -p ${ROOT_DAEMON_PORT} -L")} 2>/dev/null || true
      sleep 1
        """.trimIndent()
      } else {
        ""
      }
    return """
      set -eu
      DEVICE_DIR=${shQuote(ROOT_DAEMON_DEVICE_DIR)}
      mkdir -p "${'$'}DEVICE_DIR"
      cp ${shQuote(stagedHandlerPath)} ${shQuote(deviceHandlerPath)}
      cp ${shQuote(stagedBinaryPath)} ${shQuote(deviceBinaryPath)}
      chmod 0755 ${shQuote(deviceHandlerPath)} ${shQuote(deviceBinaryPath)}
      $forceRestartSnippet
      if toybox netstat -tlpn 2>/dev/null | grep -q ':${ROOT_DAEMON_PORT} '; then
        exit 0
      fi
      nohup env VLM_NATIVE_LIB_DIR=${shQuote(nativeLibDir)} toybox nc -s 127.0.0.1 -p ${ROOT_DAEMON_PORT} -L ${shQuote(deviceHandlerPath)} >${shQuote(stdoutPath)} 2>${shQuote(stderrPath)} </dev/null &
      sleep 1
      toybox netstat -tlpn 2>/dev/null | grep -q ':${ROOT_DAEMON_PORT} '
    """.trimIndent()
  }

  private fun runRootCommand(script: String, timeoutMs: Long): ShellExecResult {
    val candidates = listOf("/system/bin/su", "/system/xbin/su", "su")
    val failures = mutableListOf<String>()
    for (candidate in candidates) {
      val attempt = runCatching { runProcess(listOf(candidate, "-c", script), timeoutMs) }
      if (attempt.isFailure) {
        failures += "$candidate failed to start: ${attempt.exceptionOrNull()?.message}"
        continue
      }
      val result = attempt.getOrThrow()
      if (result.exitCode == 0) {
        return result
      }
      failures +=
        "$candidate exit=${result.exitCode} stdout='${result.stdout.take(256)}' stderr='${result.stderr.take(256)}'"
    }
    throw IllegalStateException(
      "Unable to execute root command from app process. Attempts:\n${failures.joinToString(separator = "\n")}\n" +
        "Remediation: provision daemon with adb root script (scripts/start_root_vlm_daemon_adb.sh).",
    )
  }

  private fun runProcess(command: List<String>, timeoutMs: Long): ShellExecResult {
    val commandName = command.firstOrNull() ?: "<unknown>"
    val process =
      try {
        ProcessBuilder(command)
          .redirectErrorStream(false)
          .start()
      } catch (io: IOException) {
        throw IllegalStateException(
          "Failed to start command '${commandName}': ${io.message}",
          io,
        )
      }

    if (!process.waitFor(timeoutMs, TimeUnit.MILLISECONDS)) {
      process.destroyForcibly()
      throw IllegalStateException(
        "Command '${commandName}' timed out after ${timeoutMs}ms.",
      )
    }

    val stdout =
      process.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText().trim() }
    val stderr =
      process.errorStream.bufferedReader(Charsets.UTF_8).use { it.readText().trim() }
    return ShellExecResult(
      exitCode = process.exitValue(),
      stdout = stdout,
      stderr = stderr,
    )
  }

  private fun readRootDaemonLogs(): String {
    val logScript =
      """
      set +e
      tail -n 120 ${shQuote("${ROOT_DAEMON_DEVICE_DIR}/vlm_daemon_stderr.log")} 2>/dev/null
      tail -n 120 ${shQuote("${ROOT_DAEMON_DEVICE_DIR}/vlm_daemon_stdout.log")} 2>/dev/null
      exit 0
      """.trimIndent()
    return runCatching {
      val result = runRootCommand(logScript, 3_000L)
      listOf(result.stdout, result.stderr).filter { it.isNotBlank() }.joinToString(separator = "\n")
    }.getOrElse {
      "unavailable (${it.message})"
    }.ifBlank { "empty" }
  }

  private fun shQuote(raw: String): String {
    return "'" + raw.replace("'", "'\"'\"'") + "'"
  }

  private suspend fun getOrCreateEngine(backend: BackendTarget, modelPath: String): Engine {
    engineMutex.withLock {
      engines[backend]?.let { return it }

      @OptIn(ExperimentalApi::class)
      ExperimentalFlags.enableBenchmark = ENABLE_RUNTIME_BENCHMARK
      if (backend == BackendTarget.NPU) {
        val dispatchDir =
          npuLibrariesDir
            ?: throw IllegalStateException(
              "NPU runtime libraries directory is not initialized. Remediation: call nativeInit(...) and verify QAIRT runtime assets are packaged.",
            )
        @OptIn(ExperimentalApi::class)
        ExperimentalFlags.npuLibrariesDir = dispatchDir
      } else {
        @OptIn(ExperimentalApi::class)
        ExperimentalFlags.npuLibrariesDir = ""
      }

      val cacheDir = File(context.cacheDir, "litert_cache")
      if (!cacheDir.exists() && !cacheDir.mkdirs()) {
        throw IllegalStateException(
          "Failed to create LiteRT cache dir '${cacheDir.absolutePath}'. Remediation: free app storage and retry.",
        )
      }

      val litertBackend = backend.toLiteRtBackend()
      val visionBackend =
        if (backend == BackendTarget.NPU) {
          Backend.NPU
        } else {
          Backend.CPU
        }
      val engineConfig =
        EngineConfig(
          modelPath = modelPath,
          backend = litertBackend,
          visionBackend = visionBackend,
          maxNumTokens = ENGINE_MAX_NUM_TOKENS,
          cacheDir = cacheDir.absolutePath,
        )

      return try {
        Engine(engineConfig).also {
          it.initialize()
          engines[backend] = it
        }
      } catch (t: Throwable) {
        throw IllegalStateException(
          "Engine initialization failed for backend=$backend: ${t.message}. Remediation: confirm model compatibility and required runtime libraries on device.",
          t,
        )
      }
    }
  }

  private fun estimateTokenCount(text: String): Int {
    return text
      .trim()
      .split(Regex("\\s+"))
      .filter { it.isNotEmpty() }
      .size
  }

  @OptIn(ExperimentalApi::class)
  private fun buildConversationConfig(): ConversationConfig {
    return ConversationConfig(
      systemInstruction = Contents.of(VLM_SYSTEM_INSTRUCTION),
      samplerConfig =
        SamplerConfig(
          topK = VLM_SAMPLER_TOP_K,
          topP = VLM_SAMPLER_TOP_P,
          temperature = VLM_SAMPLER_TEMPERATURE,
        ),
      automaticToolCalling = false,
    )
  }

  @OptIn(ExperimentalApi::class)
  private suspend fun acquireConversation(engine: Engine, backend: BackendTarget): Conversation {
    conversationPrewarmMutex.withLock {
      val cached = prewarmedConversation
      if (cached != null) {
        if (cached.backend == backend) {
          prewarmedConversation = null
          return cached.conversation
        }
        runCatching { cached.conversation.close() }
        prewarmedConversation = null
      }
    }
    return engine.createConversation(conversationConfig = buildConversationConfig())
  }

  private fun scheduleConversationPrewarm(modelPath: String, backend: BackendTarget) {
    scope.launch {
      runCatching {
        val engine = getOrCreateEngine(backend, modelPath)
        val freshConversation = engine.createConversation(conversationConfig = buildConversationConfig())
        conversationPrewarmMutex.withLock {
          prewarmedConversation?.let { stale ->
            runCatching { stale.conversation.close() }
          }
          prewarmedConversation = PrewarmedConversation(backend = backend, conversation = freshConversation)
        }
      }.onFailure { t ->
        Log.w(TAG, "conversation prewarm failed backend=$backend: ${t.message}")
      }
    }
  }

  private fun buildTransitionTimings(
    request: VqaRequest,
    vqaStartMs: Long,
    conversationCreateStartMs: Long?,
    conversationCreateDoneMs: Long?,
    sendMessageCalledMs: Long?,
    firstCallbackMessageMs: Long?,
    firstDeltaEmittedMs: Long?,
    doneCallbackMs: Long?,
    nativeMessageParseMs: Long? = null,
    nativeRenderPromptMs: Long? = null,
    nativeToInputDataMs: Long? = null,
    nativePreprocessContentsMs: Long? = null,
    nativeVisionEncodeMs: Long? = null,
    nativePrefillWaitMs: Long? = null,
    nativeDecodeToFirstChunkMs: Long? = null,
  ): TransitionTimings {
    val sttFinalDoneMs = request.stt_final_done_ms.takeIf { it > 0L }
    val dispatchMs = request.vlm_request_dispatched_ms.takeIf { it > 0L }
    val frameCaptureDoneMs = request.frame_capture_done_ms.takeIf { it > 0L }
    val preprocessDoneMs = request.preprocess_done_ms.takeIf { it > 0L }
    val voicePrefillStartedMs = request.voice_prefill_started_ms.takeIf { it > 0L }
    val voicePrefillDoneMs = request.voice_prefill_done_ms.takeIf { it > 0L }
    val streamDeliveryLagMs = durationBetween(firstCallbackMessageMs, firstDeltaEmittedMs)
    return TransitionTimings(
      stt_final_done_ms = sttFinalDoneMs,
      vlm_request_dispatched_ms = dispatchMs,
      frame_capture_done_ms = frameCaptureDoneMs,
      preprocess_done_ms = preprocessDoneMs,
      voice_session_id = request.voice_session_id,
      voice_prefill_started_ms = voicePrefillStartedMs,
      voice_prefill_done_ms = voicePrefillDoneMs,
      conversation_create_start_ms = conversationCreateStartMs,
      conversation_create_done_ms = conversationCreateDoneMs,
      send_message_called_ms = sendMessageCalledMs,
      first_callback_message_ms = firstCallbackMessageMs,
      first_delta_emitted_ms = firstDeltaEmittedMs,
      done_callback_ms = doneCallbackMs,
      native_message_parse_ms = nativeMessageParseMs,
      native_render_prompt_ms = nativeRenderPromptMs,
      native_to_input_data_ms = nativeToInputDataMs,
      native_preprocess_contents_ms = nativePreprocessContentsMs,
      native_vision_encode_ms = nativeVisionEncodeMs,
      native_prefill_wait_ms = nativePrefillWaitMs,
      native_decode_to_first_chunk_ms = nativeDecodeToFirstChunkMs,
      stream_delivery_lag_ms =
        if (streamDeliveryLagMs != null && streamDeliveryLagMs > VLM_STREAM_DELIVERY_WARN_MS) {
          streamDeliveryLagMs
        } else {
          null
        },
      stt_to_dispatch_ms = durationBetween(sttFinalDoneMs, dispatchMs),
      dispatch_to_vqa_start_ms = durationBetween(dispatchMs, vqaStartMs),
      vqa_start_to_first_callback_ms = durationBetween(vqaStartMs, firstCallbackMessageMs),
      first_callback_to_first_delta_ms = durationBetween(firstCallbackMessageMs, firstDeltaEmittedMs),
      dispatch_to_first_delta_ms = durationBetween(dispatchMs, firstDeltaEmittedMs),
    )
  }

  private fun durationBetween(startMs: Long?, endMs: Long?): Long? {
    if (startMs == null || endMs == null || startMs <= 0L || endMs <= 0L || endMs < startMs) {
      return null
    }
    return endMs - startMs
  }

  private fun parseNativeBenchmarkMarks(rawJson: String?): Map<String, Long> {
    if (rawJson.isNullOrBlank()) {
      return emptyMap()
    }
    return runCatching {
      val obj = JSONObject(rawJson)
      buildMap {
        val keys = obj.keys()
        while (keys.hasNext()) {
          val key = keys.next()
          val valueMs = obj.optDouble(key, Double.NaN)
          if (!valueMs.isNaN() && valueMs >= 0.0) {
            put(key, valueMs.roundToLong())
          }
        }
      }
    }.onFailure { t ->
      Log.w(TAG, "Failed to parse benchmark mark durations JSON: ${t.message}")
    }.getOrDefault(emptyMap())
  }

  private fun emitEvent(callback: (String) -> Unit, event: VqaEvent) {
    val raw = json.encodeToString(VqaEvent.serializer(), event)
    callback(raw)
  }

  private fun prepareNpuRuntimeEnvironment() {
    val nativeDir = context.applicationInfo.nativeLibraryDir
    val nativeDirFile = File(nativeDir)
    if (!nativeDirFile.exists()) {
      throw IllegalStateException(
        "App native library directory is missing at '${nativeDirFile.absolutePath}'. Remediation: reinstall the APK and retry.",
      )
    }

    val dispatchLib = File(nativeDirFile, "libLiteRtDispatch_Qualcomm.so")
    if (!dispatchLib.exists()) {
      throw IllegalStateException(
        "Missing Qualcomm dispatch library at '${dispatchLib.absolutePath}'. Remediation: rebuild APK with Phase 1 QNN runtime packaging enabled.",
      )
    }
    val litertLmJniLib = File(nativeDirFile, "liblitertlm_jni.so")
    if (!litertLmJniLib.exists()) {
      throw IllegalStateException(
        "Missing LiteRT-LM JNI bridge library at '${litertLmJniLib.absolutePath}'. Remediation: build and package //kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni into app jniLibs.",
      )
    }
    val litertRuntimeLib = File(nativeDirFile, "libLiteRt.so")
    if (!litertRuntimeLib.exists()) {
      throw IllegalStateException(
        "Missing LiteRT runtime library at '${litertRuntimeLib.absolutePath}'. Remediation: build and package @litert//litert/c:litert_runtime_c_api_so into app jniLibs.",
      )
    }

    val qnnLib = File(nativeDirFile, "libQnnHtp.so")
    if (!qnnLib.exists()) {
      throw IllegalStateException(
        "Missing QNN runtime library at '${qnnLib.absolutePath}'. Remediation: package QAIRT arm64 libraries into app jniLibs and reinstall.",
      )
    }

    val requiredSkelLibs =
      listOf(
        "libQnnHexagonSkel_dspApp.so",
        "libQnnHtpV79.so",
        "libQnnHtpV79Skel.so",
        "libQnnNetRunDirectV79Skel.so",
      )
    requiredSkelLibs.forEach { name ->
      val candidate = File(nativeDirFile, name)
      if (!candidate.exists()) {
        throw IllegalStateException(
          "Missing required Hexagon v79 skel library at '${candidate.absolutePath}'. Remediation: package QAIRT hexagon-v79 unsigned skel libs into app jniLibs and reinstall.",
        )
      }
    }

    val stagedRuntimeDir = stageNpuRuntimeLibraries(nativeDirFile)

    // Keep library presence checks strict, but let LiteRT dispatch own dynamic
    // loading of QNN/LiteRT runtime libs to avoid namespace collisions.
    Log.i(TAG, "Skipping explicit System.load preloads; runtime loading is delegated to LiteRT.")

    val adspPath =
      "${stagedRuntimeDir.absolutePath};/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
    try {
      Os.setenv("ADSP_LIBRARY_PATH", adspPath, true)
    } catch (e: ErrnoException) {
      throw IllegalStateException(
        "Failed to set ADSP_LIBRARY_PATH for QNN runtime: ${e.message}. Remediation: verify app process permission and retry.",
        e,
      )
    }
    try {
      Os.setenv("LITERT_QNN_SIGNED_PD_SUPPORT", "1", true)
      Log.i(TAG, "Set LITERT_QNN_SIGNED_PD_SUPPORT=1 for QNN runtime")
    } catch (e: ErrnoException) {
      throw IllegalStateException(
        "Failed to set LITERT_QNN_SIGNED_PD_SUPPORT: ${e.message}. Remediation: verify app process permission and retry.",
        e,
      )
    }
    try {
      Os.setenv("LITERT_QNN_DISABLE_DEVICE_PLATFORM_INFO", "1", true)
      Log.i(TAG, "Set LITERT_QNN_DISABLE_DEVICE_PLATFORM_INFO=1 for QNN runtime")
    } catch (e: ErrnoException) {
      throw IllegalStateException(
        "Failed to set LITERT_QNN_DISABLE_DEVICE_PLATFORM_INFO: ${e.message}. Remediation: verify app process permission and retry.",
        e,
      )
    }

    npuLibrariesDir = stagedRuntimeDir.absolutePath
    Log.i(
      TAG,
      "NPU runtime prepared: dispatch_dir=${stagedRuntimeDir.absolutePath}, adsp_path=$adspPath",
    )
  }

  private fun stageNpuRuntimeLibraries(nativeDirFile: File): File {
    val stagedDir = File(context.filesDir, "npu_dispatch_libs")
    if (!stagedDir.exists() && !stagedDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create staged NPU runtime directory '${stagedDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val sourceLibs =
      nativeDirFile
        .listFiles { file -> file.isFile && file.name.endsWith(".so") }
        ?.sortedBy { it.name }
        ?: throw IllegalStateException(
          "No shared libraries found in '${nativeDirFile.absolutePath}'. Remediation: reinstall APK with runtime libraries packaged.",
        )

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
      if (!destination.setReadable(true, false)) {
        throw IllegalStateException(
          "Failed to mark staged runtime library '${destination.absolutePath}' as readable.",
        )
      }
      if (!destination.setExecutable(true, false)) {
        throw IllegalStateException(
          "Failed to mark staged runtime library '${destination.absolutePath}' as executable.",
        )
      }
    }

    stagedDir
      .listFiles { file -> file.isFile && file.name.endsWith(".so") }
      ?.filter { staged -> sourceLibs.none { source -> source.name == staged.name } }
      ?.forEach { stale -> stale.delete() }

    marker.writeText(fingerprint)
    return stagedDir
  }

  private fun BackendTarget.toLiteRtBackend(): Backend {
    return when (this) {
      BackendTarget.NPU -> Backend.NPU
      BackendTarget.CPU -> Backend.CPU
    }
  }
}

internal fun computeStreamingDeltaFromChunk(
  emittedText: String,
  incoming: String,
): String {
  if (incoming.isEmpty()) {
    return ""
  }
  if (emittedText.isEmpty()) {
    return incoming
  }
  if (incoming.startsWith(emittedText)) {
    return incoming.substring(emittedText.length)
  }
  return incoming
}
