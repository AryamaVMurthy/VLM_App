# Native Bridge (Phase 1)

Phase 1 exposes a stable bridge contract named after the requested JNI surface:

- `nativeInit(configPath: String): InitResult`
- `nativeRunVqa(requestJson: String, callback): Long requestId`
- `nativeCancel(requestId: Long): Boolean`
- `nativeGetLastMetrics(requestId: Long): String`

Implementation location in this phase:

- [`FastVlmNativeBridge.kt`](../android-app/app/src/main/java/com/qidk/fastvlm/core/bridge/FastVlmNativeBridge.kt)

Notes:

- The app uses LiteRT-LM Android API (`litertlm-android`) which itself is JNI-backed.
- The bridge keeps request orchestration, fallback policy, and metrics contract stable for future migration to a dedicated C++ JNI layer if required.
- No silent fallback is allowed. Every fallback emits a `FALLBACK` event with `fallback_reason`.
