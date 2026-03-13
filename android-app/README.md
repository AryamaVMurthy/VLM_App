# Android App Surface

This directory contains the deployed Android runtime for GraphPilot-Edge.

## Important Modules

- `app/`
  - main Android application
- `app/src/main/java/com/qidk/fastvlm/core/graphpilot/`
  - GraphPilot runtime coordinator, scheduler, memory controller, and plan-bank logic
- `app/src/androidTest/`
  - connected instrumentation tests used to verify the deployed path
- `whisperlib/`
  - Android STT support module

## Verification

### Unit tests

```bash
cd android-app
ANDROID_HOME=/home/aryamavmurthy/android-sdk \
ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk \
./gradlew app:testDebugUnitTest --tests 'com.qidk.fastvlm.core.graphpilot.*'
```

### Connected GraphPilot tests

Start the daemon first:

```bash
./scripts/start_root_vlm_daemon_adb.sh
```

Then run:

```bash
cd android-app
ANDROID_HOME=/home/aryamavmurthy/android-sdk \
ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk \
./gradlew app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest
```

## Runtime Rules

- FastVLM on LiteRT NPU is the mainline VLM path.
- Do not introduce silent runtime fallback.
- Any regression in workflows A/B/C must be treated as a blocker.
