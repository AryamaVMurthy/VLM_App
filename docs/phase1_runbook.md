# FastVLM Voice Phase Runbook (CPU-Only)

## Scope

This runbook covers the shipping voice pipeline:

- STT: Whisper.cpp tiny.en quantized model
- VLM: FastVLM 0.5B LiteRT model
- TTS: Android native TextToSpeech
- Input UX: hold-to-speak only
- Output UX: voice-first, optional text via `Show Text`

## Locked Model Artifacts

### VLM

- `model_id`: `litert-community/FastVLM-0.5B`
- `artifact`: `FastVLM-0.5B.litertlm`
- `revision`: `74e5aa3fb2adc196ca24bc7fb561ec92fc165e81`
- `sha256`: `87114d1331e0543455d57d81ceacf28a0d80d49d98691e1b24792e51903cf8c4`

### STT

- `model_id`: `ggerganov/whisper.cpp`
- `artifact`: `ggml-tiny.en-q8_0.bin`
- `revision`: `5359861c739e955e79d9a303bcbc70fb988958b1`
- `sha256`: `5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94`

## First-Run Provisioning

App downloads models into app-private storage on first successful run.

- Requires internet only for first provisioning.
- Post-provisioning: offline runtime supported.
- If first-run internet is unavailable, app fails fast with explicit remediation text.

## Build + Install

```bash
cd android-app
./gradlew :app:installDebug
```

## On-Device Validation

1. Launch `FastVLM Voice`.
2. Grant camera + microphone permissions.
3. Wait until top state is `Ready`.
4. Hold mic, speak, release.
5. Confirm:
   - STT transcript updates while speaking.
   - VLM starts quickly after release.
   - TTS starts while answer is still streaming.

## Blank Audio Behavior

Blank/noise capture must not send gibberish to VLM.

Expected:

- STT fails fast with `No clear speech detected...`
- No VLM request starts.

## Metrics

Per-request metrics are written in app storage:

- `files/metrics/requests/<request_id>.json`

Includes transition fields such as:

- `stt_final_done_ms`
- `vlm_request_dispatched_ms`
- `first_delta_emitted_ms`
- `tts_audio_start_ms`

## Failure Handling Standard

- No silent degradation.
- Errors are explicit and user-visible.
- Fallbacks must be explicit, logged, and include reason.

