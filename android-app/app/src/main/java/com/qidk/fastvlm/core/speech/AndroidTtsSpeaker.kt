package com.qidk.fastvlm.core.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class AndroidTtsSpeaker(private val context: Context) {
  companion object {
    private const val TAG = "AndroidTtsSpeaker"
    private const val TTS_SERVICE_ACTION = "android.intent.action.TTS_SERVICE"
    private const val DEFAULT_SPEECH_RATE = 0.94f
    private const val DEFAULT_PITCH = 1.03f
    private val LOCALE_PREFERENCE =
      listOf(
        Locale.US,
        Locale.UK,
        Locale("en", "IN"),
        Locale.CANADA,
        Locale("en", "AU"),
        Locale.ENGLISH,
      )
    private val VOICE_NAME_PREFERENCE_HINTS =
      listOf(
        "samantha",
        "allison",
        "ava",
        "mike",
        "en-us",
        "american",
        "us",
        "emma",
        "amy",
        "clb",
        "slt",
        "bdl",
        "en-gb",
        "uk",
      )
    private val GENERIC_VOICE_NAME_HINTS = listOf("default", "english")
  }

  private data class PendingUtterance(
    val completion: CompletableDeferred<Unit>,
    val onAudioStart: ((Long) -> Unit)? = null,
    val onDone: (() -> Unit)? = null,
  )

  private var tts: TextToSpeech? = null
  private val pendingUtterances = ConcurrentHashMap<String, PendingUtterance>()
  @Volatile private var listenerInstalled = false

  suspend fun ensureInitialized() = withContext(Dispatchers.Main) {
    if (tts != null) {
      return@withContext
    }

    val configuredEngine = resolveConfiguredEnginePackage()
    val installedEngines = discoverInstalledTtsEngines()
    Log.i(
      TAG,
      "Initializing TTS configured_engine=${configuredEngine ?: "<system-default>"} visible_engines=$installedEngines",
    )

    val engine =
      suspendCancellableCoroutine<TextToSpeech> { continuation ->
        lateinit var local: TextToSpeech
        local =
          if (configuredEngine.isNullOrBlank()) {
            TextToSpeech(context) { status ->
              if (status != TextToSpeech.SUCCESS) {
                continuation.resumeWithException(
                  IllegalStateException(
                    "Android TTS initialization failed with status=$status. configured_engine=<system-default> visible_engines=$installedEngines. Remediation: verify Android manifest package visibility for TTS_SERVICE and ensure the configured engine is installed and enabled on device.",
                  )
                )
                return@TextToSpeech
              }
              continuation.resume(local)
            }
          } else {
            TextToSpeech(context, { status ->
              if (status != TextToSpeech.SUCCESS) {
                continuation.resumeWithException(
                  IllegalStateException(
                    "Android TTS initialization failed with status=$status. configured_engine=$configuredEngine visible_engines=$installedEngines. Remediation: verify Android manifest package visibility for TTS_SERVICE and ensure the configured engine is installed and enabled on device.",
                  )
                )
                return@TextToSpeech
              }
              continuation.resume(local)
            }, configuredEngine)
          }
      }

    val selectedLocale = selectEnglishLocale(engine)
    if (selectedLocale == null) {
      engine.shutdown()
      throw IllegalStateException(
        "Android TTS English voice is unavailable. Remediation: install/enable at least one English TTS voice in system settings.",
      )
    }
    logAvailableVoices(engine, selectedLocale)
    engine.setSpeechRate(DEFAULT_SPEECH_RATE)
    engine.setPitch(DEFAULT_PITCH)
    val selectedVoice = selectPreferredVoice(engine, selectedLocale)
    if (selectedVoice != null) {
      val voiceResult = engine.setVoice(selectedVoice)
      Log.i(
        TAG,
        "Initialized TTS locale=${selectedLocale.toLanguageTag()} voice='${selectedVoice.name}' quality=${selectedVoice.quality} latency=${selectedVoice.latency} set_result=$voiceResult",
      )
    } else {
      Log.i(TAG, "Initialized TTS with locale=${selectedLocale.toLanguageTag()} (default voice)")
    }

    tts = engine
    installUtteranceListener(engine)
  }

  private fun resolveConfiguredEnginePackage(): String? {
    return Settings.Secure.getString(context.contentResolver, "tts_default_synth")
      ?.trim()
      ?.ifEmpty { null }
  }

  private fun discoverInstalledTtsEngines(): List<String> {
    return context.packageManager
      .queryIntentServices(Intent(TTS_SERVICE_ACTION), 0)
      .mapNotNull { it.serviceInfo?.packageName?.takeIf(String::isNotBlank) }
      .distinct()
      .sorted()
  }

  suspend fun speak(text: String) = withContext(Dispatchers.Main) {
    val engine =
      tts
        ?: throw IllegalStateException(
          "TTS is not initialized. Remediation: call ensureInitialized() before speak().",
        )
    val normalized = text.trim()
    if (normalized.isEmpty()) {
      throw IllegalArgumentException("TTS speak text cannot be empty.")
    }
    failPending(
      IllegalStateException("TTS queue flushed by full-text speak() request."),
    )
    engine.stop()
    speakChunkInternal(
      engine = engine,
      text = normalized,
      queueMode = TextToSpeech.QUEUE_FLUSH,
      onAudioStart = null,
      onDone = null,
      awaitCompletion = true,
    )
  }

  suspend fun speakChunk(
    text: String,
    queueMode: Int,
    onAudioStart: ((Long) -> Unit)? = null,
    onDone: (() -> Unit)? = null,
    awaitCompletion: Boolean = false,
  ): String =
    withContext(Dispatchers.Main) {
      val engine =
        tts
          ?: throw IllegalStateException(
            "TTS is not initialized. Remediation: call ensureInitialized() before speakChunk().",
          )
      val normalized = text.trim()
      if (normalized.isEmpty()) {
        throw IllegalArgumentException("TTS chunk cannot be empty.")
      }
      speakChunkInternal(
        engine = engine,
        text = normalized,
        queueMode = queueMode,
        onAudioStart = onAudioStart,
        onDone = onDone,
        awaitCompletion = awaitCompletion,
      )
    }

  suspend fun stop() = withContext(Dispatchers.Main) {
    failPending(
      IllegalStateException("TTS stopped by caller."),
    )
    tts?.stop()
  }

  suspend fun release() = withContext(Dispatchers.Main) {
    failPending(
      IllegalStateException("TTS released while utterances were pending."),
    )
    tts?.stop()
    tts?.shutdown()
    tts = null
    listenerInstalled = false
  }

  private suspend fun speakChunkInternal(
    engine: TextToSpeech,
    text: String,
    queueMode: Int,
    onAudioStart: ((Long) -> Unit)?,
    onDone: (() -> Unit)?,
    awaitCompletion: Boolean,
  ): String {
    installUtteranceListener(engine)
    val utteranceId = "utt_${UUID.randomUUID()}"
    val speakStartMs = System.currentTimeMillis()
    Log.i(
      TAG,
      "speakChunk:start utterance_id=$utteranceId chars=${text.length} queue_mode=$queueMode",
    )
    val completion = CompletableDeferred<Unit>()
    pendingUtterances[utteranceId] =
      PendingUtterance(
        completion = completion,
        onAudioStart = onAudioStart,
        onDone = onDone,
      )

    val params = Bundle().apply {
      putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
      putFloat(TextToSpeech.Engine.KEY_PARAM_PAN, 0.0f)
    }
    val result = engine.speak(text, queueMode, params, utteranceId)
    if (result != TextToSpeech.SUCCESS) {
      pendingUtterances.remove(utteranceId)
      throw IllegalStateException(
        "TTS speakChunk() failed with result=$result. Remediation: verify TTS engine availability and retry.",
      )
    }
    if (queueMode == TextToSpeech.QUEUE_FLUSH) {
      failPending(
        IllegalStateException("TTS queue flushed by a new utterance request."),
        exceptUtteranceId = utteranceId,
      )
    }
    if (awaitCompletion) {
      completion.await()
      val elapsed = System.currentTimeMillis() - speakStartMs
      Log.i(TAG, "speakChunk:complete utterance_id=$utteranceId elapsed_ms=$elapsed")
    }
    return utteranceId
  }

  private fun installUtteranceListener(engine: TextToSpeech) {
    if (listenerInstalled) {
      return
    }
    engine.setOnUtteranceProgressListener(
      object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {
          if (utteranceId == null) return
          pendingUtterances[utteranceId]?.onAudioStart?.invoke(System.currentTimeMillis())
          Log.i(TAG, "speak:onStart utterance_id=$utteranceId")
        }

        override fun onDone(utteranceId: String?) {
          if (utteranceId == null) return
          val pending = pendingUtterances.remove(utteranceId)
          pending?.completion?.complete(Unit)
          pending?.onDone?.invoke()
          Log.i(TAG, "speak:onDone utterance_id=$utteranceId")
        }

        override fun onError(utteranceId: String?) {
          if (utteranceId == null) return
          pendingUtterances
            .remove(utteranceId)
            ?.completion
            ?.completeExceptionally(
              IllegalStateException(
                "TTS synthesis failed. Remediation: retry and verify TTS engine health in system settings.",
              )
            )
          Log.e(TAG, "speak:onError utterance_id=$utteranceId")
        }
      },
    )
    listenerInstalled = true
  }

  private fun failPending(error: Throwable, exceptUtteranceId: String? = null) {
    val ids = pendingUtterances.keys.toList()
    ids.forEach { id ->
      if (exceptUtteranceId != null && id == exceptUtteranceId) {
        return@forEach
      }
      pendingUtterances.remove(id)?.completion?.completeExceptionally(error)
    }
  }

  private fun selectEnglishLocale(engine: TextToSpeech): Locale? {
    for (locale in LOCALE_PREFERENCE) {
      if (isLocaleUsable(engine, locale)) {
        return locale
      }
    }

    val availableEnglishLocales =
      engine.availableLanguages
        .filter { it.language == Locale.ENGLISH.language }
        .sortedBy { it.toLanguageTag() }
    for (locale in availableEnglishLocales) {
      if (isLocaleUsable(engine, locale)) {
        return locale
      }
    }
    return null
  }

  private fun isLocaleUsable(engine: TextToSpeech, locale: Locale): Boolean {
    val status = engine.setLanguage(locale)
    val usable = status != TextToSpeech.LANG_MISSING_DATA && status != TextToSpeech.LANG_NOT_SUPPORTED
    if (usable) {
      Log.i(TAG, "Using TTS locale=${locale.toLanguageTag()} status=$status")
    } else {
      Log.w(TAG, "TTS locale unavailable: ${locale.toLanguageTag()} status=$status")
    }
    return usable
  }

  private fun selectPreferredVoice(engine: TextToSpeech, locale: Locale): Voice? {
    val availableVoices = engine.voices ?: return null
    if (availableVoices.isEmpty()) {
      return null
    }

    val languageCandidates =
      availableVoices.filter { voice ->
        voice.locale?.language.equals(locale.language, ignoreCase = true)
      }
    if (languageCandidates.isEmpty()) {
      return null
    }

    val localVoices = languageCandidates.filterNot { it.isNetworkConnectionRequired }
    val rankedPool = if (localVoices.isNotEmpty()) localVoices else languageCandidates

    return rankedPool
      .sortedWith(compareByDescending<Voice> { scoreVoice(it, locale) })
      .firstOrNull()
  }

  private fun scoreVoice(voice: Voice, requestedLocale: Locale): Int {
    var score = 0
    val voiceLocale = voice.locale
    val requestedTag = requestedLocale.toLanguageTag().lowercase(Locale.ROOT)
    val voiceTag = voiceLocale?.toLanguageTag()?.lowercase(Locale.ROOT).orEmpty()
    val requestedCountry = requestedLocale.country.uppercase(Locale.ROOT)
    val voiceCountry = voiceLocale?.country?.uppercase(Locale.ROOT).orEmpty()
    val voiceName = voice.name.lowercase(Locale.ROOT)

    if (!voice.isNetworkConnectionRequired) {
      score += 600
    }
    if (voiceTag == requestedTag) {
      score += 600
    }
    if (requestedCountry.isNotBlank()) {
      score +=
        when {
          voiceCountry == requestedCountry -> 500
          voiceCountry.isBlank() -> 150
          else -> -600
        }
    }
    if (voiceTag.startsWith("en-gb")) {
      score += 420
    } else if (voiceTag.startsWith("en-in")) {
      score += 380
    } else if (voiceTag.startsWith("en-us")) {
      score += 320
    } else if (voiceTag.startsWith("en")) {
      score += 240
    }
    VOICE_NAME_PREFERENCE_HINTS.forEachIndexed { index, hint ->
      if (voiceName.contains(hint)) {
        score += (VOICE_NAME_PREFERENCE_HINTS.size - index) * 40
      }
    }
    if (GENERIC_VOICE_NAME_HINTS.any { voiceName == it || voiceName.startsWith("$it ") }) {
      score -= 220
    }
    score += voice.quality
    score -= voice.latency
    return score
  }

  private fun logAvailableVoices(engine: TextToSpeech, selectedLocale: Locale) {
    val availableVoices = engine.voices ?: return
    if (availableVoices.isEmpty()) {
      Log.w(TAG, "No TTS voices reported by engine.")
      return
    }
    Log.i(
      TAG,
      "TTS voice inventory: count=${availableVoices.size} selected_locale=${selectedLocale.toLanguageTag()}",
    )
    availableVoices
      .sortedWith(
        compareBy<Voice> { it.locale?.toLanguageTag() ?: "" }.thenBy { it.name.lowercase(Locale.ROOT) },
      )
      .forEach { voice ->
        Log.i(
          TAG,
          "TTS voice candidate: name='${voice.name}' locale='${voice.locale?.toLanguageTag()}' quality=${voice.quality} latency=${voice.latency} network=${voice.isNetworkConnectionRequired}",
        )
      }
  }
}
