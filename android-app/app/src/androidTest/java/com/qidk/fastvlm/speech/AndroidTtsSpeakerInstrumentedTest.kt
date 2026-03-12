package com.qidk.fastvlm.speech

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import com.qidk.fastvlm.core.speech.AndroidTtsSpeaker
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AndroidTtsSpeakerInstrumentedTest {
  @Test
  fun initializeAndSpeakShortPhrase() = runBlocking {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val speaker = AndroidTtsSpeaker(context)
    try {
      val initStartMs = System.currentTimeMillis()
      withTimeout(20_000L) {
        speaker.ensureInitialized()
      }
      val initElapsedMs = System.currentTimeMillis() - initStartMs

      val speakStartMs = System.currentTimeMillis()
      withTimeout(20_000L) {
        speaker.speak("GraphPilot edge text to speech validation.")
      }
      val speakElapsedMs = System.currentTimeMillis() - speakStartMs

      Log.i(
        "AndroidTtsSpeakerInstrumentedTest",
        "TTS validated init_elapsed_ms=$initElapsedMs speak_elapsed_ms=$speakElapsedMs",
      )

      assertThat(initElapsedMs).isGreaterThan(0L)
      assertThat(speakElapsedMs).isGreaterThan(0L)
    } finally {
      runCatching { speaker.release() }
    }
  }
}
