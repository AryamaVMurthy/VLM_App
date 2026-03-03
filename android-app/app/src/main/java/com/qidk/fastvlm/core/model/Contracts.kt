package com.qidk.fastvlm.core.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class BackendTarget {
  @SerialName("NPU")
  NPU,

  @SerialName("CPU")
  CPU,
}

@Serializable
enum class VqaEventType {
  START,
  TOKEN,
  FALLBACK,
  METRICS,
  DONE,
  ERROR,
}

@Serializable
enum class PipelineStage {
  INIT,
  FRAME_CAPTURE,
  PREPROCESS,
  PREFILL,
  DECODE,
  TOTAL,
}

@Serializable
data class DeviceInfo(
  val soc: String,
  val device: String,
  val model: String,
  val fingerprint: String,
  val sdkInt: Int,
  val abi: String,
  val appVersion: String,
)

@Serializable
data class FallbackEvent(
  val stage: PipelineStage,
  val from_backend: BackendTarget,
  val to_backend: BackendTarget,
  val fallback_reason: String,
  val timestamp_ms: Long,
)

@Serializable
data class VqaError(
  val stage: PipelineStage,
  val code: String,
  val message: String,
  val remediation: String,
)

@Serializable
data class VqaRequest(
  val question: String,
  val image_path: String,
  val preferred_backend: BackendTarget,
  val allow_fallback: Boolean,
  val max_output_tokens: Int = 160,
  val image_source: String = "unknown",
  val image_age_ms: Long = -1L,
  val latency_mode: String = "voice_low_latency",
  val stt_final_done_ms: Long = 0L,
  val vlm_request_dispatched_ms: Long = 0L,
  val frame_capture_done_ms: Long = 0L,
  val preprocess_done_ms: Long = 0L,
  val voice_session_id: String? = null,
  val voice_prefill_started_ms: Long = 0L,
  val voice_prefill_done_ms: Long = 0L,
  val frame_capture_ms: Long = 0,
  val preprocess_ms: Long = 0,
)

@Serializable
data class StageTimings(
  val frame_capture_ms: Long,
  val preprocess_ms: Long,
  val prefill_ms: Long,
  val decode_ms: Long,
  val total_ms: Long,
)

@Serializable
data class TokenStats(
  val ttft_ms: Long,
  val decode_toks_per_sec: Double,
  val output_tokens: Int,
)

@Serializable
data class BackendStatus(
  val backend_config_requested: BackendTarget,
  val backend_config_actual: BackendTarget,
)

@Serializable
data class TransitionTimings(
  val stt_final_done_ms: Long? = null,
  val vlm_request_dispatched_ms: Long? = null,
  val frame_capture_done_ms: Long? = null,
  val preprocess_done_ms: Long? = null,
  val voice_session_id: String? = null,
  val voice_prefill_started_ms: Long? = null,
  val voice_prefill_done_ms: Long? = null,
  val conversation_create_start_ms: Long? = null,
  val conversation_create_done_ms: Long? = null,
  val send_message_called_ms: Long? = null,
  val first_callback_message_ms: Long? = null,
  val first_delta_emitted_ms: Long? = null,
  val done_callback_ms: Long? = null,
  val tts_first_chunk_queued_ms: Long? = null,
  val tts_audio_start_ms: Long? = null,
  val stream_delivery_lag_ms: Long? = null,
  val stt_to_dispatch_ms: Long? = null,
  val dispatch_to_vqa_start_ms: Long? = null,
  val vqa_start_to_first_callback_ms: Long? = null,
  val first_callback_to_first_delta_ms: Long? = null,
  val dispatch_to_first_delta_ms: Long? = null,
  val native_message_parse_ms: Long? = null,
  val native_render_prompt_ms: Long? = null,
  val native_to_input_data_ms: Long? = null,
  val native_preprocess_contents_ms: Long? = null,
  val native_vision_encode_ms: Long? = null,
  val native_prefill_wait_ms: Long? = null,
  val native_decode_to_first_chunk_ms: Long? = null,
  val first_delta_to_first_tts_audio_ms: Long? = null,
)

@Serializable
data class VqaMetrics(
  val request_id: Long,
  val device_info: DeviceInfo,
  val backend_status: BackendStatus,
  val stage_timings: StageTimings,
  val token_stats: TokenStats,
  val fallback_events: List<FallbackEvent>,
  val transition_timings: TransitionTimings = TransitionTimings(),
  val error: VqaError? = null,
)

@Serializable
data class VqaEvent(
  val type: VqaEventType,
  val request_id: Long,
  val token: String? = null,
  val backend_status: BackendStatus? = null,
  val fallback: FallbackEvent? = null,
  val metrics: VqaMetrics? = null,
  val error: VqaError? = null,
)

@Serializable
data class InitResult(
  val ok: Boolean,
  val model_path: String? = null,
  val preferred_backend: BackendTarget? = null,
  val fallback_backend: BackendTarget = BackendTarget.CPU,
  val device_info: DeviceInfo? = null,
  val message: String,
)

@Serializable
data class VqaRunContext(
  val frame_capture_ms: Long = 0,
  val preprocess_ms: Long = 0,
)
