# Known Issues (Phase 1)

1. Phase 1 intentionally excludes ASR/TTS; text-only query input is implemented.
2. First-run model download is large (~943 MB) and can take time on unstable network.
3. Device DNS/network issues can block first-run Hugging Face download; use `scripts/download_pinned_model.sh` and `scripts/provision_model_to_app.sh`.
4. On the tested runtime (`litertlm-android:0.9.0-alpha05`), NPU engine initialization for the pinned SM8750 model triggers native process abort (`SIGABRT` in `liblitertlm_jni`).
5. CPU fallback path currently returns tensor-contract errors (`Input tensor not found`) for this pinned Qualcomm-targeted artifact, so successful answer generation is blocked until runtime/model compatibility patching is applied.
