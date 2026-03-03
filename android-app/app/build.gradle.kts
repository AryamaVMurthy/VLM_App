import org.gradle.api.GradleException
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.api.tasks.Sync

plugins {
  id("com.android.application")
  kotlin("android")
  kotlin("plugin.serialization")
  kotlin("plugin.compose")
}

android {
  namespace = "com.qidk.fastvlm"
  compileSdk = 35

  defaultConfig {
    applicationId = "com.qidk.fastvlm"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "0.2.0-cpu-voice"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    ndk {
      abiFilters += "arm64-v8a"
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      buildConfigField("boolean", "ENABLE_DEV_ROOT_DAEMON_BRIDGE", "false")
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro",
      )
    }
    debug {
      isDebuggable = true
      buildConfigField("boolean", "ENABLE_DEV_ROOT_DAEMON_BRIDGE", "false")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  kotlinOptions {
    jvmTarget = "17"
    freeCompilerArgs += listOf("-Xjvm-default=all")
  }

  buildFeatures {
    compose = true
    buildConfig = true
  }

  packaging {
    jniLibs {
      useLegacyPackaging = true
    }
    resources {
      excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
  }

  sourceSets {
    getByName("main") {
      // Use local LiteRT-LM Kotlin API sources so app behavior is pinned to the
      // same runtime workspace used by the patched native builds.
      java.srcDir(layout.buildDirectory.dir("generated/phase1/litertlm-src"))
      jniLibs.srcDir(layout.buildDirectory.dir("generated/phase1/jniLibs"))
    }
  }

  testOptions {
    unitTests.isIncludeAndroidResources = true
  }
}

val litertLmRoot = layout.projectDirectory.dir("../../third_party/litert-lm")
val litertLmKotlinRoot = litertLmRoot.dir("kotlin/java/com/google/ai/edge/litertlm")
val litertLmPrebuiltArm64Dir = litertLmRoot.dir("prebuilt/android_arm64")

val litertLmJniHostLib =
  litertLmRoot.file("bazel-bin/kotlin/java/com/google/ai/edge/litertlm/jni/liblitertlm_jni.so")
val litertRuntimeHostLib =
  litertLmRoot.file("bazel-bin/external/litert/litert/c/libLiteRt.so")

val prepareLiteRtLmSources by
  tasks.registering(Sync::class) {
    from(litertLmKotlinRoot)
    include("*.kt")
    into(layout.buildDirectory.dir("generated/phase1/litertlm-src"))
    doFirst {
      val sourceDir = litertLmKotlinRoot.asFile
      if (!sourceDir.exists()) {
        throw GradleException(
          "Missing LiteRT-LM Kotlin sources at '${sourceDir.absolutePath}'. Remediation: ensure third_party/litert-lm is cloned.",
        )
      }
    }
  }

val preparePhase1JniLibs by
  tasks.registering(Sync::class) {
    from(litertLmPrebuiltArm64Dir)
    from(litertLmJniHostLib)
    from(litertRuntimeHostLib)
    into(layout.buildDirectory.dir("generated/phase1/jniLibs/arm64-v8a"))
    include("*.so")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    doFirst {
      delete(layout.buildDirectory.dir("generated/phase1/jniLibs"))
      val missing = mutableListOf<String>()
      val litertPrebuilt = litertLmPrebuiltArm64Dir.asFile
      if (!litertPrebuilt.exists()) {
        missing +=
          "Missing LiteRT-LM prebuilt arm64 directory: ${litertPrebuilt.absolutePath}. Remediation: ensure third_party/litert-lm is complete."
      }
      if (!litertLmJniHostLib.asFile.exists()) {
        missing +=
          "Missing patched JNI library: ${litertLmJniHostLib.asFile.absolutePath}. Build it first: " +
            "cd ../../third_party/litert-lm && bazel build --config=android_arm64 //kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni"
      }
      if (!litertRuntimeHostLib.asFile.exists()) {
        missing +=
          "Missing LiteRT runtime shared library: ${litertRuntimeHostLib.asFile.absolutePath}. Build it first: " +
            "cd ../../third_party/litert-lm && bazel build --config=android_arm64 @litert//litert/c:litert_runtime_c_api_so"
      }
      if (missing.isNotEmpty()) {
        throw GradleException(
          "CPU runtime prerequisites are missing.\n" + missing.joinToString(separator = "\n"),
        )
      }
    }
  }

tasks.named("preBuild") {
  dependsOn(prepareLiteRtLmSources, preparePhase1JniLibs)
}

dependencies {
  val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
  implementation(composeBom)
  androidTestImplementation(composeBom)

  implementation("androidx.core:core-ktx:1.15.0")
  implementation("androidx.activity:activity-compose:1.9.3")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
  implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
  implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.compose.ui:ui-tooling-preview")
  debugImplementation("androidx.compose.ui:ui-tooling")
  implementation("com.google.android.material:material:1.12.0")

  implementation("androidx.camera:camera-camera2:1.4.0")
  implementation("androidx.camera:camera-lifecycle:1.4.0")
  implementation("androidx.camera:camera-view:1.4.0")
  implementation("androidx.exifinterface:exifinterface:1.3.7")
  implementation(project(":whisperlib"))

  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
  implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
  implementation("com.squareup.okhttp3:okhttp:4.12.0")
  implementation("com.google.code.gson:gson:2.10.1")
  implementation(kotlin("reflect"))

  testImplementation("junit:junit:4.13.2")
  testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
  testImplementation("com.google.truth:truth:1.4.4")

  androidTestImplementation("androidx.test.ext:junit:1.2.1")
  androidTestImplementation("androidx.test:runner:1.6.2")
  androidTestImplementation("com.google.truth:truth:1.4.4")
}
