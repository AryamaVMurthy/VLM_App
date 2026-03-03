package com.qidk.fastvlm.ui

import com.qidk.fastvlm.core.model.InitResult

data class AppUiState(
  val isInitializing: Boolean = true,
  val initResult: InitResult? = null,
  val speechReady: Boolean = false,
  val speechStatus: String = "not_initialized",
  val question: String = "",
  val cameraReady: Boolean = false,
  val answer: String = "",
  val backendStatus: String = "",
  val fallbackReason: String? = null,
  val metricsJson: String = "",
  val errorMessage: String? = null,
  val isRunning: Boolean = false,
  val activeRequestId: Long? = null,
  val isRecordingVoice: Boolean = false,
  val isTranscribingVoice: Boolean = false,
  val isSpeaking: Boolean = false,
  val benchmarkRunning: Boolean = false,
  val benchmarkProgress: String = "",
  val benchmarkSummary: String = "",
  val benchmarkExportStatus: String = "",
  val transitionState: String = "idle",
  val firstTokenLatencyMs: Long? = null,
  val firstAudioLatencyMs: Long? = null,
  val hiddenPrepareMs: Long? = null,
)
