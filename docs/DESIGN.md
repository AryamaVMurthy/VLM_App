# DESIGN: FastVLM Voice (CPU-Only Android)

## 1. Goals

- Deliver low-latency local voice interaction on Android:
  - `STT -> VLM -> TTS`
- Keep UX simple:
  - camera full-screen
  - hold-to-speak mic
  - voice-first output
- Maintain explicit observability and fail-fast errors.

## 2. Non-Goals

- Cloud inference.
- Hidden fallback behavior.
- Complex multi-turn memory management.

## 3. High-Level Architecture

### UI Layer

- `MainScreen`:
  - camera preview
  - top compact status bar
  - hold-to-speak mic surface
  - optional answer text panel (`Show Text`)
- `MainViewModel`:
  - state transitions
  - voice capture lifecycle
  - STT orchestration
  - VLM event handling
  - streaming TTS coordination

### Core Pipeline

- `PcmVoiceRecorder`
  - records 16kHz mono PCM chunks while press-hold is active.
- `WhisperSttEngine`
  - local transcription via whisper.cpp JNI wrapper.
  - supports interim chunk transcription + final decode.
- `VqaOrchestrator`
  - prepares camera frame and request metadata.
  - submits VQA request to native bridge.
- `FastVlmNativeBridge`
  - LiteRT conversation + streaming token callbacks.
  - metrics + timing instrumentation.
- `StreamingTtsCoordinator`
  - sentence/chunk queueing into Android TTS.
  - starts speaking before VLM DONE.

### Native + Model Layer

- FastVLM model executed through LiteRT runtime on CPU path.
- Whisper model executed through whisper.cpp native library.
- Android TextToSpeech voice configured to US-first selection.

## 4. Data Flow

1. User presses and holds mic.
2. Recorder starts; interim STT updates UI question text.
3. On release:
   - final STT transcript computed locally,
   - app validates transcript quality.
4. Request sent to VLM with latest prepared camera frame.
5. VLM token stream emits incremental deltas.
6. TTS begins from streamed chunks (not waiting for full answer).
7. Metrics persisted for each request.

## 5. Latency Strategy

- Keep capture and preprocessing small (preview frame path).
- Stream token deltas immediately from callback.
- Start TTS on early sentence boundaries / forced flush threshold.
- Maintain explicit timing markers:
  - STT final,
  - VLM dispatch,
  - first callback/delta,
  - first TTS audio start.

## 6. Offline Provisioning Model

- First successful run downloads and verifies models to app-private storage.
- Checksum and expected size validation enforced.
- After provisioning, pipeline runs without internet/laptop.
- Fresh offline install fails fast with explicit remediation.

## 7. Error Model

- No silent fallback policy.
- Fail-fast examples:
  - empty/invalid speech
  - provisioning/network errors
  - runtime initialization failures
- User-visible messages include remediation hints.

## 8. UX States

Top status bar states include:

- `Initializing`
- `Preparing camera`
- `Listening`
- `Transcribing`
- `Thinking`
- `Answering`
- `Speaking`
- `Ready`

Additional behaviors:

- `Show Text` reveals generated response text.
- During STT stage, a small `Heard: ...` strip shows spoken text before VLM send.

## 9. Packaging

- Module `app`: primary Android app.
- Module `whisperlib`: whisper JNI bridge and native bindings.
- Output APK:
  - `android-app/app/build/outputs/apk/debug/app-debug.apk`

## 10. Verification Checklist

- Voice query end-to-end executes locally.
- Blank audio fails without gibberish VLM request.
- App launches from launcher icon (`FastVLM Voice`).
- TTS locale/voice logs confirm US-first voice preference.

