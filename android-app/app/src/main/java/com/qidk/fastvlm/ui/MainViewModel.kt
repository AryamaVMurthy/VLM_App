package com.qidk.fastvlm.ui

import android.app.Application
import android.Manifest
import android.util.Log
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.model.TransitionTimings
import com.qidk.fastvlm.core.model.VqaEvent
import com.qidk.fastvlm.core.model.VqaEventType
import com.qidk.fastvlm.core.orchestrator.VqaOrchestrator
import com.qidk.fastvlm.core.orchestrator.VoicePrefillHandle
import com.qidk.fastvlm.core.speech.AndroidTtsSpeaker
import com.qidk.fastvlm.core.speech.PcmVoiceRecorder
import com.qidk.fastvlm.core.speech.StreamingTtsCoordinator
import com.qidk.fastvlm.core.speech.WhisperSttEngine
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.absoluteValue
import kotlin.math.max
import kotlin.math.sqrt
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout

class MainViewModel(application: Application) : AndroidViewModel(application) {
  companion object {
    private const val TAG = "MainViewModel"
    private const val STT_CHUNK_MS = 300
    private const val STT_SAMPLE_RATE_HZ = 16_000
    private const val STT_INTERIM_TRANSCRIPT_INTERVAL_MS = 500L
    private const val STT_INTERIM_WINDOW_MS = 1_400L
    private const val STT_CHUNK_CHANNEL_CAPACITY = 96
    private const val STT_COLLECTOR_JOIN_TIMEOUT_MS = 15_000L
    private const val STT_MAX_TRANSCRIPTION_LATENCY_MS = 10_000L
    private const val STT_FINAL_TRIM_THRESHOLD_RATIO = 0.06f
    private const val STT_FINAL_TRIM_ABS_MIN = 0.0015f
    private const val STT_FINAL_TRIM_ABS_MAX = 0.0060f
    private const val STT_FINAL_TRIM_MIN_KEEP_RATIO = 0.55f
    private const val STT_MIN_SPEECH_PEAK = 0.030f
    private const val STT_MIN_SPEECH_RMS = 0.0080f
    private const val STT_MIN_SPEECH_MS = 120L
    private const val STT_MIN_SPEECH_PCT = 0.008f
    private const val STT_MIN_SPEECH_FALLBACK_MS = 220L
    private const val VOICE_PREFILL_JOIN_TIMEOUT_MS = 900L
  }

  private val appContext = application.applicationContext
  private val orchestrator = VqaOrchestrator(appContext)
  private val whisperSttEngine = WhisperSttEngine(appContext)
  private val voiceRecorder = PcmVoiceRecorder()
  private val ttsSpeaker = AndroidTtsSpeaker(appContext)
  private val json = JsonCodec.instance

  private val _uiState = MutableStateFlow(AppUiState())
  val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()
  private data class VoiceCaptureTelemetry(
    val startedAtMs: Long = 0L,
    val sampleCount: Long = 0L,
    val speechSampleCount: Long = 0L,
    val speechRmsSum: Float = 0f,
    val speechChunkCount: Long = 0L,
    val maxAbsAmplitude: Float = 0f,
    val interimTranscript: String = "",
    val firstChunkMs: Long = -1L,
  )

  private val sttTelemetryLock = Any()
  private var sttTelemetry: VoiceCaptureTelemetry = VoiceCaptureTelemetry()
  private var voiceChunkChannel: Channel<FloatArray>? = null
  private var voiceChunkCollectorJob: Job? = null
  private var activeVoicePrefillJob: Job? = null
  private val voiceSessionCounter = AtomicLong(0L)
  @Volatile private var activeVoiceSessionId: Long = 0L

  private var initialized = false
  @Volatile private var isStoppingVoiceCapture = false
  @Volatile private var stopVoiceFlowInProgress = false
  @Volatile private var autoSpeakOnDone = false
  @Volatile private var sttFinalDoneMs = 0L
  @Volatile private var ttsFirstChunkQueuedMs = 0L
  @Volatile private var ttsAudioStartMs = 0L
  @Volatile private var requestStartMs = 0L
  @Volatile private var vlmFirstTokenMs = 0L
  @Volatile private var vlmFirstTokenAbsoluteMs = 0L
  @Volatile private var activeRequestId = -1L
  @Volatile private var ttsTimingRequestId = -1L
  @Volatile private var activeVoicePrefillSessionId: String? = null
  @Volatile private var activeVoicePrefillHandle: VoicePrefillHandle? = null
  @Volatile private var activeVoicePrefillFailure: String? = null
  private val streamingTtsCoordinator by lazy {
    StreamingTtsCoordinator(
      ttsSpeaker = ttsSpeaker,
      scope = viewModelScope,
      onFirstChunkQueued = { queuedAtMs -> handleTtsFirstChunkQueued(queuedAtMs) },
      onFirstAudioStart = { startedAtMs -> handleTtsAudioStart(startedAtMs) },
      onSpeechActive = {
        _uiState.value =
          _uiState.value.copy(
            isSpeaking = true,
            speechStatus = "speaking_live",
            transitionState = "speaking_live",
          )
      },
      onSpeechIdle = {
        _uiState.value =
          _uiState.value.copy(
            isSpeaking = false,
            speechStatus = "ready",
            transitionState = if (_uiState.value.isRunning) "answering_stream" else "idle",
          )
      },
    )
  }

  private fun withTelemetry(update: VoiceCaptureTelemetry.() -> VoiceCaptureTelemetry) {
    synchronized(sttTelemetryLock) {
      sttTelemetry = sttTelemetry.update()
    }
  }

  private fun currentTelemetry(): VoiceCaptureTelemetry {
    synchronized(sttTelemetryLock) {
      return sttTelemetry
    }
  }

  private fun handleTtsFirstChunkQueued(queuedAtMs: Long) {
    if (ttsFirstChunkQueuedMs > 0L) {
      return
    }
    ttsFirstChunkQueuedMs = queuedAtMs
    val requestId = if (activeRequestId > 0L) activeRequestId else ttsTimingRequestId
    if (requestId > 0L && !_uiState.value.isRunning) {
      updateTransitionTimingsForRequest(requestId) { existing ->
        existing.copy(
          tts_first_chunk_queued_ms = queuedAtMs,
        )
      }
    }
  }

  private fun handleTtsAudioStart(startedAtMs: Long) {
    if (ttsAudioStartMs > 0L) {
      return
    }
    ttsAudioStartMs = startedAtMs
    val firstAudioLatencyMs =
      if (vlmFirstTokenAbsoluteMs > 0L && startedAtMs >= vlmFirstTokenAbsoluteMs) {
        (startedAtMs - vlmFirstTokenAbsoluteMs).coerceAtLeast(0L)
      } else {
        null
      }
    _uiState.value =
      _uiState.value.copy(
        firstAudioLatencyMs = firstAudioLatencyMs,
        transitionState = "speaking_live",
        isSpeaking = true,
        speechStatus = "speaking_live",
      )
    val requestId = if (activeRequestId > 0L) activeRequestId else ttsTimingRequestId
    if (requestId > 0L && !_uiState.value.isRunning) {
      updateTransitionTimingsForRequest(requestId) { existing ->
        existing.copy(
          tts_audio_start_ms = startedAtMs,
        )
      }
    }
  }

  private fun updateTransitionTimingsForRequest(
    requestId: Long,
    update: (TransitionTimings) -> TransitionTimings,
  ) {
    viewModelScope.launch(Dispatchers.IO) {
      runCatching {
        orchestrator.updateRequestTransitionTimings(requestId) { existing ->
          mergeDerivedTransitionTimings(update(existing))
        }
      }.onSuccess { updatedMetrics ->
        _uiState.value =
          _uiState.value.copy(
            metricsJson = json.encodeToString(com.qidk.fastvlm.core.model.VqaMetrics.serializer(), updatedMetrics),
            hiddenPrepareMs = computeHiddenPrepareMs(updatedMetrics.transition_timings),
          )
      }.onFailure { t ->
        Log.w(TAG, "Failed to update transition timings request_id=$requestId: ${t.message}")
      }
    }
  }

  private fun mergeDerivedTransitionTimings(existing: TransitionTimings): TransitionTimings {
    val firstDeltaToAudioMs =
      if (
        existing.first_delta_emitted_ms != null &&
          existing.tts_audio_start_ms != null &&
          existing.tts_audio_start_ms >= existing.first_delta_emitted_ms
      ) {
        existing.tts_audio_start_ms - existing.first_delta_emitted_ms
      } else {
        null
      }
    return existing.copy(first_delta_to_first_tts_audio_ms = firstDeltaToAudioMs)
  }

  private fun computeHiddenPrepareMs(transitionTimings: TransitionTimings?): Long? {
    if (transitionTimings == null) {
      return null
    }
    transitionTimings.native_prefill_wait_ms?.let { prefillWaitMs ->
      return prefillWaitMs
    }
    val components =
      listOf(
        transitionTimings.native_message_parse_ms,
        transitionTimings.native_render_prompt_ms,
        transitionTimings.native_to_input_data_ms,
        transitionTimings.native_preprocess_contents_ms,
        transitionTimings.native_vision_encode_ms,
      ).filterNotNull()
    if (components.isEmpty()) {
      return null
    }
    return components.sum()
  }

  fun initializeIfNeeded() {
    if (initialized) return
    initialized = true
    val hasMicFeature =
      appContext.packageManager.hasSystemFeature(android.content.pm.PackageManager.FEATURE_MICROPHONE)
    if (!hasMicFeature) {
      _uiState.value =
        _uiState.value.copy(
          isInitializing = false,
          speechReady = false,
          speechStatus = "mic_feature_missing",
          errorMessage =
            "Microphone hardware missing. Remediation: install on a device with microphone support.",
        )
      return
    }
    Log.i(TAG, "initializeIfNeeded: microphone_feature=$hasMicFeature")

    viewModelScope.launch {
      Log.i(TAG, "initializeIfNeeded: start")
      val result = orchestrator.initialize()
      if (!result.ok) {
        _uiState.value =
          _uiState.value.copy(
            isInitializing = false,
            initResult = result,
            backendStatus = "init_failed",
            errorMessage = result.message,
            speechReady = false,
            speechStatus = "init_failed",
          )
        return@launch
      }

      runCatching {
        val sttConfig = ensureSpeechRuntimeConfigFile().absolutePath
        val sttModelPath = whisperSttEngine.ensureInitialized(sttConfig)
        val warmupMs = whisperSttEngine.warmup()
        ttsSpeaker.ensureInitialized()
        Triple(sttConfig, sttModelPath, warmupMs)
      }.onSuccess { (_, modelPath, warmupMs) ->
        _uiState.value =
          _uiState.value.copy(
            isInitializing = false,
            initResult = result,
            backendStatus = "requested=CPU actual=${result.preferred_backend ?: "unknown"}",
            errorMessage = null,
            speechReady = true,
            speechStatus = "ready model=${File(modelPath).name} warmup_ms=$warmupMs",
          )
      }.onFailure { t ->
        Log.e(TAG, "Speech initialization failed", t)
        _uiState.value =
          _uiState.value.copy(
            isInitializing = false,
            initResult = result,
            backendStatus = "requested=CPU actual=${result.preferred_backend ?: "unknown"}",
            speechReady = false,
            speechStatus = "failed",
            errorMessage = t.message ?: "Speech initialization failed",
          )
      }
    }
  }

  fun bindCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
    orchestrator.bindCamera(lifecycleOwner, previewView) { ready, reason ->
      if (ready) {
        viewModelScope.launch(Dispatchers.IO) {
          runCatching { orchestrator.primePreviewFrameCache() }
            .onFailure { t ->
              Log.w(TAG, "primePreviewFrameCache failed: ${t.message}")
            }
        }
      }
      _uiState.value =
        _uiState.value.copy(
          cameraReady = ready,
          errorMessage = if (ready) null else reason,
        )
    }
  }

  fun onQuestionChanged(question: String) {
    _uiState.value = _uiState.value.copy(question = question)
  }

  fun ask() {
    startRequest(_uiState.value.question.trim(), shouldAutoSpeak = false, sttDoneTimestampMs = 0L)
  }

  fun startVoiceRecording() {
    streamingTtsCoordinator.abortStreaming()
    if (_uiState.value.isRunning) {
      val runningRequestId = _uiState.value.activeRequestId
      if (runningRequestId != null) {
        runCatching { orchestrator.cancel(runningRequestId) }
      }
      _uiState.value =
        _uiState.value.copy(
          isRunning = false,
          activeRequestId = null,
          transitionState = "idle",
        )
      requestStartMs = 0L
      vlmFirstTokenMs = 0L
      vlmFirstTokenAbsoluteMs = 0L
      activeRequestId = -1L
      ttsTimingRequestId = -1L
      autoSpeakOnDone = false
    }
    Log.i(
      TAG,
      "startVoiceRecording requested: isRecording=${_uiState.value.isRecordingVoice}, isRunning=${_uiState.value.isRunning}, isInitializing=${_uiState.value.isInitializing}, speechReady=${_uiState.value.speechReady}",
    )
    if (_uiState.value.isRecordingVoice || _uiState.value.isInitializing || _uiState.value.isTranscribingVoice) {
      Log.w(
        TAG,
        "startVoiceRecording ignored due state guard: isRecording=${_uiState.value.isRecordingVoice}, isInitializing=${_uiState.value.isInitializing}, isTranscribing=${_uiState.value.isTranscribingVoice}",
      )
      return
    }
    if (!_uiState.value.speechReady) {
      Log.w(TAG, "startVoiceRecording ignored: speech runtime not ready")
      _uiState.value =
        _uiState.value.copy(
          errorMessage =
            "Speech runtime is not ready. Remediation: wait for initialization to complete and retry.",
        )
      return
    }
    val hasMicPermission =
      appContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
        android.content.pm.PackageManager.PERMISSION_GRANTED
    if (!hasMicPermission) {
      Log.w(TAG, "startVoiceRecording aborted: RECORD_AUDIO permission not granted")
      _uiState.value =
        _uiState.value.copy(
          errorMessage = "Microphone permission not granted. Remediation: grant RECORD_AUDIO permission.",
          speechStatus = "permission_denied",
        )
      return
    }

    viewModelScope.launch(Dispatchers.IO) {
      val previousPrefillSessionId = activeVoicePrefillSessionId
      runCatching { activeVoicePrefillJob?.cancel() }
      runCatching { orchestrator.cancelVoicePrefill(previousPrefillSessionId) }
      activeVoicePrefillJob = null
      activeVoicePrefillSessionId = null
      activeVoicePrefillHandle = null
      activeVoicePrefillFailure = null

      val voiceSessionId = voiceSessionCounter.incrementAndGet()
      activeVoiceSessionId = voiceSessionId
      val prefillSessionId = "voice_${voiceSessionId}_${System.currentTimeMillis()}"
      activeVoicePrefillSessionId = prefillSessionId
      val chunkChannel = Channel<FloatArray>(capacity = STT_CHUNK_CHANNEL_CAPACITY)
      voiceChunkChannel = chunkChannel
      isStoppingVoiceCapture = false
      withTelemetry {
        copy(
          startedAtMs = System.currentTimeMillis(),
          sampleCount = 0L,
          speechSampleCount = 0L,
          speechRmsSum = 0f,
          speechChunkCount = 0L,
          maxAbsAmplitude = 0f,
          interimTranscript = "",
          firstChunkMs = -1L,
        )
      }

      val collectorJob =
        launch {
          val interimWindowSampleLimit =
            ((STT_SAMPLE_RATE_HZ.toLong() * STT_INTERIM_WINDOW_MS) / 1000L).toInt().coerceAtLeast(1)
          var lastInterimTranscriptFlushMs = 0L
          var interimWindowSamples = FloatArray(0)
          var interimTranscript = ""
          var speechChunkCountSeen = 0
          val collectorStartedAt = System.currentTimeMillis()
          for (firstChunk in chunkChannel) {
            if (voiceSessionId != activeVoiceSessionId) {
              Log.i(
                TAG,
                "Realtime STT collector stale session ignored collector=$voiceSessionId active=$activeVoiceSessionId",
              )
              break
            }
            var pendingChunk: FloatArray? = firstChunk
            var lastChunkPeak = 0f
            var lastChunkRms = 0f
            while (pendingChunk != null) {
              val chunk = pendingChunk
              val chunkPeak = chunkPeakAmplitude(chunk)
              val chunkRms = chunkRmsAmplitude(chunk)
              lastChunkPeak = chunkPeak
              lastChunkRms = chunkRms
              val isSpeechChunk = chunkPeak >= STT_MIN_SPEECH_PEAK && chunkRms >= STT_MIN_SPEECH_RMS
              if (isSpeechChunk) {
                speechChunkCountSeen += 1
              }
              withTelemetry {
                copy(
                  sampleCount = sampleCount + chunk.size.toLong(),
                  speechSampleCount = speechSampleCount + if (isSpeechChunk) chunk.size.toLong() else 0L,
                  speechRmsSum = speechRmsSum + if (isSpeechChunk) chunkRms else 0f,
                  speechChunkCount = speechChunkCount + if (isSpeechChunk) 1L else 0L,
                  maxAbsAmplitude = max(maxAbsAmplitude, chunkPeak),
                )
              }
              if (isSpeechChunk || interimTranscript.isNotBlank()) {
                interimWindowSamples =
                  appendInterimWindow(interimWindowSamples, chunk, interimWindowSampleLimit)
              }
              pendingChunk = chunkChannel.tryReceive().getOrNull()
            }
            val nowMs = System.currentTimeMillis()
            if (interimWindowSamples.isEmpty()) {
              continue
            }
            if (nowMs - lastInterimTranscriptFlushMs < STT_INTERIM_TRANSCRIPT_INTERVAL_MS) {
              continue
            }
            if (speechChunkCountSeen < 2) {
              continue
            }
            lastInterimTranscriptFlushMs = nowMs

            val transcript: String? =
              runCatching {
                Log.i(
                  TAG,
                  "Realtime STT transcribe start samples=${interimWindowSamples.size} peak=$lastChunkPeak rms=$lastChunkRms",
                )
                whisperSttEngine.transcribe(interimWindowSamples)
              }.getOrElse { t ->
                if (t is CancellationException) {
                  Log.w(TAG, "Realtime STT chunk processing cancelled")
                } else {
                  Log.w(TAG, "Realtime STT chunk failed: ${t.message}")
                }
                null
              }
            if (transcript.isNullOrBlank()) {
              continue
            }
            val hadFirstChunk = currentTelemetry().firstChunkMs >= 0L
            interimTranscript = mergeTranscripts(interimTranscript, transcript)
            val currentMs = System.currentTimeMillis() - collectorStartedAt
            withTelemetry {
              copy(
                interimTranscript = interimTranscript,
                firstChunkMs = if (firstChunkMs < 0) currentMs else firstChunkMs,
              )
            }
            if (!hadFirstChunk) {
              Log.i(
                TAG,
                "Realtime STT first chunk transcript latency_ms=$currentMs",
              )
            }
            Log.i(
              TAG,
              "Realtime STT updated in $currentMs ms: '${transcript.take(80)}'",
            )
            if (voiceSessionId == activeVoiceSessionId) {
              _uiState.value = _uiState.value.copy(question = interimTranscript)
            }
          }
          Log.i(
            TAG,
            "Realtime STT collector completed in ${System.currentTimeMillis() - collectorStartedAt} ms session=$voiceSessionId",
          )
        }
      voiceChunkCollectorJob = collectorJob

      runCatching {
        _uiState.value =
          _uiState.value.copy(
            isRecordingVoice = true,
            question = "",
            errorMessage = null,
            speechStatus = "preparing_image",
            transitionState = "preparing_image",
            firstTokenLatencyMs = null,
            firstAudioLatencyMs = null,
            hiddenPrepareMs = null,
          )
        stopVoiceFlowInProgress = false
        val startMs = System.currentTimeMillis()
        voiceRecorder.startRecording(
          chunkDurationMs = STT_CHUNK_MS,
          onChunk = { chunk ->
            if (!isStoppingVoiceCapture) {
              val sendResult = chunkChannel.trySend(chunk)
              if (!sendResult.isSuccess) {
                Log.w(
                  TAG,
                  "Realtime chunk queue send failed: isClosed=${sendResult.isClosed}, cause=${sendResult.exceptionOrNull()?.message}",
                )
              }
            }
          },
        )
        Log.i(TAG, "startVoiceRecording: recorder_started_ms=${System.currentTimeMillis() - startMs}")
      }.onSuccess {
        val prefillJob =
          launch {
            Log.i(TAG, "Voice prefill begin session_id=$prefillSessionId")
            runCatching {
              orchestrator.beginVoicePrefill(prefillSessionId)
            }.onSuccess { handle ->
              if (activeVoicePrefillSessionId != prefillSessionId) {
                runCatching { orchestrator.cancelVoicePrefill(prefillSessionId) }
                return@onSuccess
              }
              activeVoicePrefillHandle = handle
              activeVoicePrefillFailure = null
              Log.i(
                TAG,
                "Voice prefill ready session_id=${handle.voiceSessionId} duration_ms=${handle.prefillDoneMs - handle.prefillStartedMs} image_source=${handle.imageSource}",
              )
              if (_uiState.value.isRecordingVoice) {
                _uiState.value =
                  _uiState.value.copy(
                    speechStatus = "recording",
                    transitionState = "idle",
                  )
              }
            }.onFailure { t ->
              if (activeVoicePrefillSessionId != prefillSessionId) {
                return@onFailure
              }
              activeVoicePrefillHandle = null
              activeVoicePrefillFailure = t.message ?: "Voice prefill failed"
              Log.w(
                TAG,
                "Voice prefill failed session_id=$prefillSessionId reason=${activeVoicePrefillFailure}",
              )
              _uiState.value =
                _uiState.value.copy(
                  speechStatus = "prefill_failed",
                  transitionState = if (_uiState.value.isRecordingVoice) "idle" else _uiState.value.transitionState,
                  errorMessage =
                    "Voice prefill failed; continuing on standard path. reason=${activeVoicePrefillFailure}",
                )
            }
          }
        activeVoicePrefillJob = prefillJob
        _uiState.value =
          _uiState.value.copy(
            speechStatus =
              if (activeVoicePrefillFailure == null) {
                "preparing_image"
              } else {
                "prefill_failed"
              },
          )
      }.onFailure { t ->
        Log.w(TAG, "startVoiceRecording failed, resetting recording state", t)
        Log.e(TAG, "startVoiceRecording failed: ${t.message}", t)
        runCatching { activeVoicePrefillJob?.cancel() }
        runCatching { orchestrator.cancelVoicePrefill(prefillSessionId) }
        activeVoicePrefillJob = null
        activeVoicePrefillSessionId = null
        activeVoicePrefillHandle = null
        activeVoicePrefillFailure = null
        collectorJob.cancel()
        runCatching { chunkChannel.close() }
        voiceChunkChannel = null
        voiceChunkCollectorJob = null
        _uiState.value =
          _uiState.value.copy(
            isRecordingVoice = false,
            speechStatus = "record_failed",
            errorMessage = t.message ?: "Failed to start voice recording",
          )
      }
    }
  }

  fun stopVoiceRecordingAndAsk() {
    Log.i(
      TAG,
      "stopVoiceRecordingAndAsk requested: isRecording=${_uiState.value.isRecordingVoice}, isTranscribing=${_uiState.value.isTranscribingVoice}, hasActiveRecorder=${voiceRecorder.hasActiveRecording()}",
    )
    if (!_uiState.value.isRecordingVoice) {
      return
    }
    if (stopVoiceFlowInProgress) {
      Log.w(TAG, "stopVoiceRecordingAndAsk ignored: stop flow already in progress")
      return
    }
    stopVoiceFlowInProgress = true

    viewModelScope.launch(Dispatchers.IO) {
      val stopSessionId = activeVoiceSessionId
      val prefillSessionAtStop = activeVoicePrefillSessionId
      _uiState.value =
        _uiState.value.copy(
          isRecordingVoice = false,
          isTranscribingVoice = true,
          speechStatus = "transcribing",
          errorMessage = null,
        )

      val transcriptResult = runCatching {
        val chunkCollector = voiceChunkCollectorJob
        val chunkChannel = voiceChunkChannel
        voiceChunkCollectorJob = null
        voiceChunkChannel = null
        val stopStartMs = System.currentTimeMillis()
        isStoppingVoiceCapture = true
        val hadActiveRecorder = voiceRecorder.hasActiveRecording()
        val telemetryAtStop = currentTelemetry()
        Log.i(TAG, "stopVoiceRecordingAndAsk: stop flow start hasActiveRecorder=$hadActiveRecorder")
        Log.i(
          TAG,
          "stopVoiceRecordingAndAsk: speech telemetry samples=${telemetryAtStop.sampleCount} speech_samples=${telemetryAtStop.speechSampleCount} peak=${telemetryAtStop.maxAbsAmplitude} first_chunk_ms=${telemetryAtStop.firstChunkMs}",
        )

        runCatching { chunkChannel?.close() }
          .onFailure { t ->
            Log.w(TAG, "stopVoiceRecordingAndAsk: failed to close chunk channel", t)
          }
        val samples =
          if (hadActiveRecorder) {
            runCatching { voiceRecorder.stopRecording() }.getOrElse { throw it }
          } else {
            FloatArray(0)
          }
        isStoppingVoiceCapture = false
        Log.i(
          TAG,
          "stopVoiceRecordingAndAsk: recorder stop elapsed_ms=${System.currentTimeMillis() - stopStartMs}",
        )
        val collectorJoined =
          runCatching {
            Log.i(
              TAG,
              "stopVoiceRecordingAndAsk: awaiting collector completion with timeoutMs=$STT_COLLECTOR_JOIN_TIMEOUT_MS",
            )
            withTimeout(STT_COLLECTOR_JOIN_TIMEOUT_MS) {
              chunkCollector?.join()
            }
            true
          }.getOrElse { t ->
            Log.w(
              TAG,
              "stopVoiceRecordingAndAsk: chunk collector exceeded cap",
              t,
            )
            false
          }
        if (!collectorJoined) {
          if (activeVoiceSessionId == stopSessionId) {
            activeVoiceSessionId = voiceSessionCounter.incrementAndGet()
          }
          throw IllegalStateException(
            "Realtime transcription exceeded ${STT_COLLECTOR_JOIN_TIMEOUT_MS}ms cap. Remediation: retry with shorter utterance and verify device CPU headroom.",
          )
        }
        val telemetryAfterCollector = currentTelemetry()

        val minSpeechSamples =
          STT_SAMPLE_RATE_HZ.toLong() * STT_MIN_SPEECH_MS / 1000L
        val fallbackMinSpeechSamples =
          STT_SAMPLE_RATE_HZ.toLong() * STT_MIN_SPEECH_FALLBACK_MS / 1000L
        val avgSpeechRms =
          if (telemetryAfterCollector.speechChunkCount > 0L) {
            telemetryAfterCollector.speechRmsSum / telemetryAfterCollector.speechChunkCount
          } else {
            0f
          }
        val speechRatio =
          if (telemetryAfterCollector.sampleCount > 0L) {
            telemetryAfterCollector.speechSampleCount.toFloat() / telemetryAfterCollector.sampleCount.toFloat()
          } else {
            0f
          }
        val hasLikelySpeech =
          telemetryAfterCollector.maxAbsAmplitude >= STT_MIN_SPEECH_PEAK * 0.85f &&
            telemetryAfterCollector.sampleCount >= minSpeechSamples &&
            (telemetryAfterCollector.speechSampleCount >= maxOf(1L, fallbackMinSpeechSamples / 8) ||
              avgSpeechRms >= STT_MIN_SPEECH_RMS * 1.25f) &&
            speechRatio >= STT_MIN_SPEECH_PCT

        val interimTranscript = telemetryAfterCollector.interimTranscript.trim()
        if (!hadActiveRecorder) {
          throw IllegalStateException(
            "No active recorder. Remediation: restart voice capture flow.",
          )
        }
        if (!hasLikelySpeech) {
          throw IllegalStateException(
            "No clear speech detected. Remediation: hold and speak clearly for at least ${STT_MIN_SPEECH_MS}ms.",
          )
        }
        val trimmedSamples = trimSilenceForFinalTranscription(samples)
        if (trimmedSamples.isEmpty()) {
          throw IllegalStateException(
            "Final audio segment is empty after silence trim. Remediation: hold and speak clearly, then retry.",
          )
        }
        val minSamplesToKeep =
          (samples.size.toFloat() * STT_FINAL_TRIM_MIN_KEEP_RATIO).toInt().coerceAtLeast(1)
        val finalSamples =
          if (trimmedSamples.size < minSamplesToKeep) {
            Log.w(
              TAG,
              "Final trim too aggressive: full_samples=${samples.size} trimmed_samples=${trimmedSamples.size} min_keep=$minSamplesToKeep; using full capture for decode",
            )
            samples
          } else {
            trimmedSamples
          }
        val transcript =
          runCatching {
            val finalStartMs = System.currentTimeMillis()
            whisperSttEngine.transcribe(finalSamples).also {
              val elapsed = System.currentTimeMillis() - finalStartMs
              Log.i(TAG, "Final Whisper transcription in ${elapsed}ms samples=${finalSamples.size}")
              if (elapsed > STT_MAX_TRANSCRIPTION_LATENCY_MS) {
                throw IllegalStateException(
                  "Final Whisper transcription exceeded ${STT_MAX_TRANSCRIPTION_LATENCY_MS}ms cap (elapsed=${elapsed}ms). Remediation: reduce utterance length or use faster model quantization.",
                )
              }
              if (it.isBlank()) {
                throw IllegalStateException(
                  "Final Whisper transcription returned an empty transcript. Remediation: retry with clearer speech.",
                )
              }
            }
          }.getOrElse {
            throw IllegalStateException(
              "Final Whisper transcription failed: ${it.message}",
              it,
            )
          }
        Log.i(
          TAG,
          "stopVoiceRecordingAndAsk: selected transcript source=final full_samples=${samples.size} trimmed_samples=${trimmedSamples.size} decode_samples=${finalSamples.size} interim_chars=${interimTranscript.length} recorder/collector cleanup elapsed_ms=${System.currentTimeMillis() - stopStartMs}",
        )
        if (activeVoiceSessionId == stopSessionId) {
          activeVoiceSessionId = voiceSessionCounter.incrementAndGet()
        }

        transcript
      }.onSuccess { transcript ->
        isStoppingVoiceCapture = false
        if (transcript.trim().isEmpty()) {
          Log.w(TAG, "stopVoiceRecordingAndAsk: transcript is empty; not starting VLM request")
          _uiState.value =
            _uiState.value.copy(
              isTranscribingVoice = false,
              speechStatus = "ready",
              isRecordingVoice = false,
            )
          return@onSuccess
        }

        val transcriptPreview =
          if (transcript.length <= 120) {
            transcript
          } else {
            transcript.substring(0, 120)
          }
        Log.i(TAG, "Voice transcription success: '${transcriptPreview}'")
        _uiState.value =
          _uiState.value.copy(
            isTranscribingVoice = false,
            question = transcript,
            speechStatus = "transcribed",
          )
        if (
          prefillSessionAtStop != null &&
            activeVoicePrefillHandle == null &&
            activeVoicePrefillFailure == null
        ) {
          runCatching {
            withTimeout(VOICE_PREFILL_JOIN_TIMEOUT_MS) {
              activeVoicePrefillJob?.join()
            }
          }.onFailure { t ->
            activeVoicePrefillFailure =
              "Voice prefill exceeded ${VOICE_PREFILL_JOIN_TIMEOUT_MS}ms wait: ${t.message}"
            runCatching { orchestrator.cancelVoicePrefill(prefillSessionAtStop) }
          }
        }
        val prefillHandleForRequest = activeVoicePrefillHandle
        var prefillFailureForRequest = activeVoicePrefillFailure
        val prefillSessionForRequest = activeVoicePrefillSessionId
        if (prefillHandleForRequest == null && prefillFailureForRequest == null && prefillSessionForRequest != null) {
          prefillFailureForRequest =
            "Voice prefill unavailable for session '$prefillSessionForRequest'; using standard path."
        }
        activeVoicePrefillJob = null
        activeVoicePrefillSessionId = null
        activeVoicePrefillHandle = null
        activeVoicePrefillFailure = null
        if (prefillHandleForRequest == null || prefillFailureForRequest != null) {
          runCatching { orchestrator.cancelVoicePrefill(prefillSessionForRequest) }
        }
        Log.i(
          TAG,
          "Voice request prefill selection session_id=$prefillSessionForRequest using_prefill=${prefillHandleForRequest != null && prefillFailureForRequest == null} reason=${prefillFailureForRequest ?: "none"}",
        )
        val sttDoneAtMs = System.currentTimeMillis()
        startRequest(
          transcript,
          shouldAutoSpeak = true,
          sttDoneTimestampMs = sttDoneAtMs,
          voicePrefillHandle = prefillHandleForRequest,
          voicePrefillFailureReason = prefillFailureForRequest,
        )
      }.onFailure { t ->
        Log.e(TAG, "Voice transcription failed", t)
        isStoppingVoiceCapture = false
        if (activeVoiceSessionId == stopSessionId) {
          activeVoiceSessionId = voiceSessionCounter.incrementAndGet()
        }
        runCatching { voiceChunkCollectorJob?.cancel() }
        runCatching { voiceChunkChannel?.close() }
        runCatching { activeVoicePrefillJob?.cancel() }
        runCatching { orchestrator.cancelVoicePrefill(prefillSessionAtStop) }
        activeVoicePrefillJob = null
        activeVoicePrefillSessionId = null
        activeVoicePrefillHandle = null
        activeVoicePrefillFailure = null
        _uiState.value =
          _uiState.value.copy(
            isTranscribingVoice = false,
            isRecordingVoice = false,
            speechStatus = "transcribe_failed",
            transitionState = "idle",
            errorMessage = t.message ?: "Voice transcription failed",
          )
      }.also {
        stopVoiceFlowInProgress = false
      }
    }
  }

  private fun mergeTranscripts(previous: String, incoming: String): String {
    val normalized =
      incoming
        .replace("\n", " ")
        .replace(Regex("\\s+"), " ")
        .trim()
    if (normalized.isBlank()) {
      return previous
    }

    if (previous.isBlank()) {
      return normalized
    }

    val prevWords = previous.split(" ")
    val incomingWords = normalized.split(" ")
    val maxOverlap = max(1, minOf(4, prevWords.size, incomingWords.size))
    for (overlap in maxOverlap downTo 1) {
      val prevSuffix = prevWords.takeLast(overlap).joinToString(" ")
      val incomingPrefix = incomingWords.take(overlap).joinToString(" ")
      if (prevSuffix.equals(incomingPrefix, ignoreCase = true)) {
        val merged =
          (prevWords + incomingWords.subList(overlap, incomingWords.size)).joinToString(" ")
        return merged.ifEmpty { previous }
      }
    }
    return "${previous} ${normalized}"
  }

  fun cancel() {
    activeVoiceSessionId = voiceSessionCounter.incrementAndGet()
    streamingTtsCoordinator.abortStreaming()
    val prefillSessionId = activeVoicePrefillSessionId
    runCatching { activeVoicePrefillJob?.cancel() }
    runCatching { orchestrator.cancelVoicePrefill(prefillSessionId) }
    activeVoicePrefillJob = null
    activeVoicePrefillSessionId = null
    activeVoicePrefillHandle = null
    activeVoicePrefillFailure = null
    val requestId = _uiState.value.activeRequestId
    if (requestId != null) {
      val cancelled = orchestrator.cancel(requestId)
      _uiState.value =
        _uiState.value.copy(
          isRunning = false,
          activeRequestId = null,
          errorMessage = if (cancelled) "Request cancelled" else "No active request to cancel",
        )
    }

    viewModelScope.launch(Dispatchers.IO) {
      runCatching { voiceChunkCollectorJob?.cancel() }
      runCatching { voiceChunkChannel?.close() }
      runCatching { voiceChunkCollectorJob = null }
      runCatching { voiceChunkChannel = null }
      runCatching { voiceRecorder.cancelRecording() }
      runCatching { ttsSpeaker.stop() }
    }

    _uiState.value =
      _uiState.value.copy(
        isRecordingVoice = false,
        isTranscribingVoice = false,
        isSpeaking = false,
        transitionState = "idle",
        firstTokenLatencyMs = null,
        firstAudioLatencyMs = null,
        hiddenPrepareMs = null,
      )
    if (
      requestStartMs > 0L ||
        vlmFirstTokenMs > 0L ||
        vlmFirstTokenAbsoluteMs > 0L ||
        activeRequestId > 0L ||
        sttFinalDoneMs > 0L
    ) {
      requestStartMs = 0L
      vlmFirstTokenMs = 0L
      vlmFirstTokenAbsoluteMs = 0L
      activeRequestId = -1L
      ttsTimingRequestId = -1L
      sttFinalDoneMs = 0L
      ttsFirstChunkQueuedMs = 0L
      ttsAudioStartMs = 0L
    }
    autoSpeakOnDone = false
  }

  fun runBenchmark() {
    val question = _uiState.value.question.trim().ifEmpty { "Describe what you see." }
    if (_uiState.value.benchmarkRunning) return
    if (_uiState.value.initResult?.ok != true) {
      _uiState.value =
        _uiState.value.copy(
          errorMessage =
            _uiState.value.initResult?.message
              ?: "Runtime is not initialized. Remediation: restart app and complete initialization.",
        )
      return
    }
    if (!_uiState.value.cameraReady || !orchestrator.isCameraReady()) {
      _uiState.value =
        _uiState.value.copy(
          errorMessage =
            orchestrator.cameraNotReadyReason()
              ?: "Camera is not ready. Remediation: wait for preview initialization before running benchmark.",
        )
      return
    }

    _uiState.value =
      _uiState.value.copy(
        benchmarkRunning = true,
        benchmarkProgress = "Starting benchmark...",
        benchmarkSummary = "",
        benchmarkExportStatus = "",
        errorMessage = null,
      )

    viewModelScope.launch(Dispatchers.IO) {
      runCatching {
        Log.i(TAG, "benchmark: start question='${question.take(80)}'")
        orchestrator.runTenRunBenchmark(
          question = question,
          onProgress = { index, total ->
            _uiState.value = _uiState.value.copy(benchmarkProgress = "Run $index/$total")
          },
          onEvent = { event ->
            if (event.type == VqaEventType.ERROR) {
              _uiState.value = _uiState.value.copy(errorMessage = event.error?.message)
            }
          },
        )
      }.onSuccess { summary ->
        Log.i(TAG, "benchmark: done benchmark_id=${summary.benchmark_id}")
        _uiState.value =
          _uiState.value.copy(
            benchmarkRunning = false,
            benchmarkProgress = "",
            benchmarkSummary =
              "runs=${summary.run_count}, avg_ttft_ms=${"%.2f".format(summary.avg_ttft_ms)}, avg_decode_tps=${"%.2f".format(summary.avg_decode_toks_per_sec)}, avg_total_ms=${"%.2f".format(summary.avg_total_ms)}",
          )
      }.onFailure { t ->
        Log.e(TAG, "benchmark failed: ${t.message}", t)
        _uiState.value =
          _uiState.value.copy(
            benchmarkRunning = false,
            benchmarkProgress = "",
            errorMessage = t.message ?: "Benchmark failed",
          )
      }
    }
  }

  fun exportLatestBenchmark() {
    if (_uiState.value.benchmarkRunning || _uiState.value.isRunning) {
      _uiState.value =
        _uiState.value.copy(
          errorMessage = "Cannot export while a request is running. Remediation: wait for current run to finish.",
        )
      return
    }
    viewModelScope.launch(Dispatchers.IO) {
      runCatching { orchestrator.exportLatestBenchmarkReport() }
        .onSuccess { (jsonFile, textFile) ->
          _uiState.value =
            _uiState.value.copy(
              benchmarkExportStatus =
                "Exported benchmark reports:\n${jsonFile.absolutePath}\n${textFile.absolutePath}",
              errorMessage = null,
            )
        }
        .onFailure { t ->
          _uiState.value =
            _uiState.value.copy(
              benchmarkExportStatus = "",
              errorMessage = t.message ?: "Benchmark export failed",
            )
        }
    }
  }

  override fun onCleared() {
    super.onCleared()
    runCatching { streamingTtsCoordinator.abortStreaming() }
    runCatching { activeVoicePrefillJob?.cancel() }
    runCatching { orchestrator.cancelVoicePrefill(activeVoicePrefillSessionId) }
    activeVoicePrefillJob = null
    activeVoicePrefillSessionId = null
    activeVoicePrefillHandle = null
    activeVoicePrefillFailure = null
    runBlocking {
      runCatching { voiceChunkCollectorJob?.cancel() }
      runCatching { voiceChunkChannel?.close() }
      runCatching { voiceRecorder.cancelRecording() }
      runCatching { whisperSttEngine.release() }
      runCatching { ttsSpeaker.release() }
      voiceChunkCollectorJob = null
      voiceChunkChannel = null
    }
    orchestrator.shutdown()
  }

  private fun startRequest(
    question: String,
    shouldAutoSpeak: Boolean,
    sttDoneTimestampMs: Long,
    voicePrefillHandle: VoicePrefillHandle? = null,
    voicePrefillFailureReason: String? = null,
  ) {
    if (question.isEmpty()) {
      _uiState.value = _uiState.value.copy(errorMessage = "Question cannot be empty")
      return
    }
    if (_uiState.value.isRunning) {
      return
    }
    if (_uiState.value.initResult?.ok != true) {
      _uiState.value =
        _uiState.value.copy(
          errorMessage =
            _uiState.value.initResult?.message
              ?: "Runtime is not initialized. Remediation: restart app and complete initialization.",
        )
      return
    }
    if (!_uiState.value.cameraReady || !orchestrator.isCameraReady()) {
      _uiState.value =
        _uiState.value.copy(
          errorMessage =
            orchestrator.cameraNotReadyReason()
              ?: "Camera is not ready. Remediation: wait for preview initialization before requesting VQA.",
        )
      return
    }

    streamingTtsCoordinator.abortStreaming()
    if (shouldAutoSpeak) {
      streamingTtsCoordinator.startSession()
    }
    autoSpeakOnDone = shouldAutoSpeak
    sttFinalDoneMs = sttDoneTimestampMs
    ttsFirstChunkQueuedMs = 0L
    ttsAudioStartMs = 0L
    requestStartMs = System.currentTimeMillis()
    vlmFirstTokenMs = 0L
    vlmFirstTokenAbsoluteMs = 0L
    activeRequestId = -1L
    ttsTimingRequestId = -1L

    _uiState.value =
      _uiState.value.copy(
        answer = "",
        fallbackReason = voicePrefillFailureReason,
        metricsJson = "",
        errorMessage = null,
        isRunning = true,
        isSpeaking = false,
        speechStatus = "thinking",
        transitionState = "thinking",
        firstTokenLatencyMs = null,
        firstAudioLatencyMs = null,
        hiddenPrepareMs = null,
      )

    viewModelScope.launch(Dispatchers.IO) {
      try {
        Log.i(
          TAG,
          "request_start_ms=$requestStartMs question='${question.take(80)}' autoSpeak=$shouldAutoSpeak stt_final_done_ms=$sttDoneTimestampMs prefill_session=${voicePrefillHandle?.voiceSessionId ?: "none"}",
        )
        val requestId =
          orchestrator.runSingleQuestion(
            question = question,
            sttFinalDoneMs = sttFinalDoneMs,
            voicePrefillSessionId = voicePrefillHandle?.voiceSessionId,
          ) { event ->
            handleEvent(event)
          }
        activeRequestId = requestId
        ttsTimingRequestId = requestId
        Log.i(TAG, "ask: started requestId=$requestId")
        _uiState.value = _uiState.value.copy(activeRequestId = requestId)
      } catch (t: Throwable) {
        Log.e(TAG, "ask failed: ${t.message}", t)
        _uiState.value =
          _uiState.value.copy(
            isRunning = false,
            activeRequestId = null,
            errorMessage = t.message ?: "Request failed",
          )
        runCatching { orchestrator.cancelVoicePrefill(voicePrefillHandle?.voiceSessionId) }
        autoSpeakOnDone = false
        requestStartMs = 0L
        vlmFirstTokenMs = 0L
        vlmFirstTokenAbsoluteMs = 0L
        activeRequestId = -1L
        ttsTimingRequestId = -1L
        sttFinalDoneMs = 0L
        ttsFirstChunkQueuedMs = 0L
        ttsAudioStartMs = 0L
      }
    }
  }

  private fun handleEvent(event: VqaEvent) {
    if (activeRequestId > 0L && event.request_id != activeRequestId) {
      Log.w(
        TAG,
        "handleEvent: ignoring stale event request_id=${event.request_id} activeRequestId=$activeRequestId",
      )
      return
    }
    when (event.type) {
      VqaEventType.START -> {
        Log.i(
          TAG,
          "VQA START request_id=${event.request_id} backend_requested=${event.backend_status?.backend_config_requested} backend_actual=${event.backend_status?.backend_config_actual}",
        )
        _uiState.value =
          _uiState.value.copy(
            backendStatus =
              "requested=${event.backend_status?.backend_config_requested} actual=${event.backend_status?.backend_config_actual}",
            transitionState = "thinking",
            speechStatus = if (autoSpeakOnDone) "thinking" else _uiState.value.speechStatus,
          )
      }

      VqaEventType.TOKEN -> {
        val token = event.token.orEmpty()
        if (vlmFirstTokenMs <= 0L && requestStartMs > 0L) {
          val nowMs = System.currentTimeMillis()
          vlmFirstTokenMs = nowMs - requestStartMs
          vlmFirstTokenAbsoluteMs = nowMs
          Log.i(TAG, "VQA first token latency_ms=$vlmFirstTokenMs request_id=${event.request_id}")
        }
        _uiState.value =
          _uiState.value.copy(
            answer = _uiState.value.answer + token,
            firstTokenLatencyMs = if (vlmFirstTokenMs > 0L) vlmFirstTokenMs else null,
            transitionState = "answering_stream",
            speechStatus = if (autoSpeakOnDone) "answering_stream" else _uiState.value.speechStatus,
          )
        if (autoSpeakOnDone) {
          streamingTtsCoordinator.appendDelta(token)
        }
      }

      VqaEventType.FALLBACK -> {
        _uiState.value =
          _uiState.value.copy(
            fallbackReason = event.fallback?.fallback_reason,
            backendStatus =
              "requested=${event.backend_status?.backend_config_requested} actual=${event.backend_status?.backend_config_actual}",
          )
      }

      VqaEventType.METRICS -> {
        val metricsJson =
          event.metrics?.let { json.encodeToString(com.qidk.fastvlm.core.model.VqaMetrics.serializer(), it) }
            ?: ""
        _uiState.value =
          _uiState.value.copy(
            metricsJson = metricsJson,
            hiddenPrepareMs = computeHiddenPrepareMs(event.metrics?.transition_timings),
          )
        if (event.metrics != null && (ttsFirstChunkQueuedMs > 0L || ttsAudioStartMs > 0L)) {
          updateTransitionTimingsForRequest(event.request_id) { existing ->
            existing.copy(
              tts_first_chunk_queued_ms =
                if (ttsFirstChunkQueuedMs > 0L) {
                  ttsFirstChunkQueuedMs
                } else {
                  existing.tts_first_chunk_queued_ms
                },
              tts_audio_start_ms =
                if (ttsAudioStartMs > 0L) {
                  ttsAudioStartMs
                } else {
                  existing.tts_audio_start_ms
                },
            )
          }
        }
      }

      VqaEventType.DONE -> {
        val doneMs = if (requestStartMs > 0L) System.currentTimeMillis() - requestStartMs else 0L
        Log.i(
          TAG,
          "VQA DONE request_id=${event.request_id} latency_ms=$doneMs first_token_ms=$vlmFirstTokenMs answer_chars=${_uiState.value.answer.length} answer_preview='${_uiState.value.answer.take(200)}'",
        )
        _uiState.value =
          _uiState.value.copy(
            isRunning = false,
            activeRequestId = null,
            transitionState =
              if (_uiState.value.isSpeaking) {
                "speaking_live"
              } else {
                "idle"
              },
          )
        if (ttsFirstChunkQueuedMs > 0L || ttsAudioStartMs > 0L) {
          updateTransitionTimingsForRequest(event.request_id) { existing ->
            existing.copy(
              tts_first_chunk_queued_ms =
                if (ttsFirstChunkQueuedMs > 0L) {
                  ttsFirstChunkQueuedMs
                } else {
                  existing.tts_first_chunk_queued_ms
                },
              tts_audio_start_ms =
                if (ttsAudioStartMs > 0L) {
                  ttsAudioStartMs
                } else {
                  existing.tts_audio_start_ms
                },
            )
          }
        }
        requestStartMs = 0L
        vlmFirstTokenMs = 0L
        vlmFirstTokenAbsoluteMs = 0L
        activeRequestId = -1L
        ttsTimingRequestId = event.request_id
        sttFinalDoneMs = 0L
        val shouldSpeak = autoSpeakOnDone
        autoSpeakOnDone = false
        if (shouldSpeak) {
          streamingTtsCoordinator.finishStreaming()
        }
      }

      VqaEventType.ERROR -> {
        Log.e(TAG, "VQA ERROR request_id=${event.request_id} message=${event.error?.message}")
        autoSpeakOnDone = false
        _uiState.value =
          _uiState.value.copy(
            isRunning = false,
            activeRequestId = null,
            errorMessage = event.error?.message ?: "Inference failed",
            transitionState =
              if (_uiState.value.isSpeaking) {
                "speaking_live"
              } else {
                "idle"
              },
          )
        requestStartMs = 0L
        vlmFirstTokenMs = 0L
        vlmFirstTokenAbsoluteMs = 0L
        activeRequestId = -1L
        ttsTimingRequestId = event.request_id
        sttFinalDoneMs = 0L
      }
    }
  }

  private fun chunkPeakAmplitude(chunk: FloatArray): Float {
    var peak = 0f
    for (sample in chunk) {
      val abs = sample.absoluteValue
      if (abs > peak) {
        peak = abs
      }
    }
    return peak
  }

  private fun chunkRmsAmplitude(chunk: FloatArray): Float {
    if (chunk.isEmpty()) {
      return 0f
    }
    var sumSquares = 0.0
    for (sample in chunk) {
      val abs = sample.absoluteValue.toDouble()
      sumSquares += abs * abs
    }
    return sqrt(sumSquares / chunk.size.toDouble()).toFloat()
  }

  private fun appendInterimWindow(
    existing: FloatArray,
    incoming: FloatArray,
    maxSamples: Int,
  ): FloatArray {
    if (incoming.isEmpty()) {
      return existing
    }
    val merged = FloatArray(existing.size + incoming.size)
    System.arraycopy(existing, 0, merged, 0, existing.size)
    System.arraycopy(incoming, 0, merged, existing.size, incoming.size)
    return if (merged.size <= maxSamples) {
      merged
    } else {
      merged.copyOfRange(merged.size - maxSamples, merged.size)
    }
  }

  private fun trimSilenceForFinalTranscription(samples: FloatArray): FloatArray {
    if (samples.isEmpty()) {
      return samples
    }
    var peak = 0f
    for (sample in samples) {
      val abs = sample.absoluteValue
      if (abs > peak) {
        peak = abs
      }
    }
    if (peak <= 0f) {
      return FloatArray(0)
    }
    val threshold =
      (peak * STT_FINAL_TRIM_THRESHOLD_RATIO).coerceIn(STT_FINAL_TRIM_ABS_MIN, STT_FINAL_TRIM_ABS_MAX)
    var start = 0
    while (start < samples.size && samples[start].absoluteValue < threshold) {
      start += 1
    }
    var end = samples.size - 1
    while (end >= start && samples[end].absoluteValue < threshold) {
      end -= 1
    }
    if (start > end) {
      return FloatArray(0)
    }
    return samples.copyOfRange(start, end + 1)
  }

  private fun ensureSpeechRuntimeConfigFile(): File {
    val configDir = File(appContext.filesDir, "configs")
    if (!configDir.exists() && !configDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create speech config dir '${configDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val configFile = File(configDir, "whisper_stt.json")
    appContext.assets.open("whisper_stt.json").use { input ->
      configFile.outputStream().use { output ->
        input.copyTo(output)
      }
    }
    return configFile
  }
  
}
