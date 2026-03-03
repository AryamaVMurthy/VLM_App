# FastVLM Voice (CPU-Only Android)

On-device voice assistant pipeline for Android:

`STT (Whisper.cpp) -> VLM (FastVLM LiteRT) -> TTS (Android native)`

This repository is configured for **CPU-only** inference in app flow.

## What Works

- Camera-first full-screen UI with hold-to-speak mic interaction.
- Live local STT transcription while speaking.
- FastVLM visual question answering from camera frame + transcribed text.
- Streaming answer tokens from VLM.
- Streaming native Android TTS playback (starts before full VLM completion).
- Barge-in behavior: new hold-to-speak interrupts active TTS.
- Metrics logging for transition latency and stage timing.

## Offline Behavior

- First app run requires internet on device to download:
  - FastVLM model (`FastVLM-0.5B.litertlm`)
  - Whisper model (`ggml-tiny.en-q8_0.bin`)
- After successful first download, the app runs offline (no laptop, no internet required).
- Fresh install without internet fails fast with explicit provisioning error.

## Requirements

- Android 15 / SDK 35 device (tested on SM8750P target device).
- `arm64-v8a` ABI.
- Camera + microphone permissions.

## Build

```bash
cd android-app
./gradlew :app:assembleDebug
```

APK output:

- `android-app/app/build/outputs/apk/debug/app-debug.apk`

## Install

```bash
adb install -r android-app/app/build/outputs/apk/debug/app-debug.apk
```

Launch app name in launcher:

- **FastVLM Voice**

## Runtime Flow

1. Open app and wait for `Ready`.
2. Press and hold mic button.
3. Speak question.
4. Release to send transcript + camera frame to VLM.
5. Listen to streamed spoken answer.
6. Optional: tap `Show Text` to inspect generated text.

## Model Configs

- VLM config: `android-app/app/src/main/assets/fastvlm_phase1.json`
- Whisper config: `android-app/app/src/main/assets/whisper_stt.json`

## Design + Ops Docs

- Design doc: `docs/DESIGN.md`
- Runbook: `docs/phase1_runbook.md`
- Validation protocol: `docs/validation_protocol.md`

## Repo Layout

- App: `android-app/app`
- Whisper Android module: `android-app/whisperlib`
- Configs: `configs/models`
- Scripts: `scripts`
- Docs: `docs`

