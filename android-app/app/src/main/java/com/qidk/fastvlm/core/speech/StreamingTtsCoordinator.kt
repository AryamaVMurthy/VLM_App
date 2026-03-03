package com.qidk.fastvlm.core.speech

import android.speech.tts.TextToSpeech
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

private const val TAG = "StreamingTtsCoordinator"
private const val FORCE_FLUSH_CHARS = 96

class StreamingTtsCoordinator(
  private val ttsSpeaker: AndroidTtsSpeaker,
  private val scope: CoroutineScope,
  private val onFirstChunkQueued: (Long) -> Unit = {},
  private val onFirstAudioStart: (Long) -> Unit = {},
  private val onSpeechActive: () -> Unit = {},
  private val onSpeechIdle: () -> Unit = {},
) {
  private val lock = Any()
  private val pendingText = StringBuilder()
  private var sessionId = 0L
  private var hasQueuedChunk = false
  private var firstChunkQueuedReported = false
  private var firstAudioStartedReported = false
  private var pendingUtteranceCount = 0

  fun startSession() {
    synchronized(lock) {
      sessionId += 1L
      pendingText.clear()
      hasQueuedChunk = false
      firstChunkQueuedReported = false
      firstAudioStartedReported = false
      pendingUtteranceCount = 0
    }
  }

  fun appendDelta(delta: String) {
    if (delta.isBlank()) {
      return
    }
    val chunksToSpeak: List<String>
    val localSessionId: Long
    synchronized(lock) {
      pendingText.append(delta)
      localSessionId = sessionId
      chunksToSpeak = extractSpeakableChunks(finalFlush = false)
    }
    queueChunks(localSessionId, chunksToSpeak)
  }

  fun finishStreaming() {
    val chunksToSpeak: List<String>
    val localSessionId: Long
    synchronized(lock) {
      localSessionId = sessionId
      chunksToSpeak = extractSpeakableChunks(finalFlush = true)
    }
    queueChunks(localSessionId, chunksToSpeak)
  }

  fun abortStreaming() {
    val localSessionId: Long
    val hadActiveSpeech: Boolean
    synchronized(lock) {
      sessionId += 1L
      localSessionId = sessionId
      hadActiveSpeech = pendingUtteranceCount > 0 || hasQueuedChunk
      pendingText.clear()
      hasQueuedChunk = false
      firstChunkQueuedReported = false
      firstAudioStartedReported = false
      pendingUtteranceCount = 0
    }
    scope.launch(Dispatchers.IO) {
      runCatching { ttsSpeaker.stop() }
        .onFailure { t ->
          Log.w(TAG, "abortStreaming stop failed session=$localSessionId message=${t.message}")
        }
      if (hadActiveSpeech) {
        onSpeechIdle()
      }
    }
  }

  private fun queueChunks(localSessionId: Long, chunks: List<String>) {
    if (chunks.isEmpty()) {
      return
    }
    chunks.forEach { chunk ->
      val queueMode: Int
      val queuedAtMs = System.currentTimeMillis()
      var shouldQueue = true
      synchronized(lock) {
        if (localSessionId != sessionId) {
          shouldQueue = false
          queueMode = TextToSpeech.QUEUE_ADD
          return@synchronized
        }
        queueMode =
          if (hasQueuedChunk) {
            TextToSpeech.QUEUE_ADD
          } else {
            TextToSpeech.QUEUE_FLUSH
          }
        hasQueuedChunk = true
        pendingUtteranceCount += 1
        if (pendingUtteranceCount == 1) {
          onSpeechActive()
        }
        if (!firstChunkQueuedReported) {
          firstChunkQueuedReported = true
          onFirstChunkQueued(queuedAtMs)
        }
      }
      if (!shouldQueue) {
        return
      }
      scope.launch(Dispatchers.IO) {
        runCatching {
          ttsSpeaker.speakChunk(
            text = chunk,
            queueMode = queueMode,
            onAudioStart = { startedAtMs ->
              synchronized(lock) {
                if (localSessionId != sessionId || firstAudioStartedReported) {
                  return@speakChunk
                }
                firstAudioStartedReported = true
              }
              onFirstAudioStart(startedAtMs)
            },
            onDone = {
              var shouldMarkIdle = false
              synchronized(lock) {
                if (localSessionId != sessionId) {
                  return@speakChunk
                }
                if (pendingUtteranceCount > 0) {
                  pendingUtteranceCount -= 1
                }
                shouldMarkIdle = pendingUtteranceCount == 0
              }
              if (shouldMarkIdle) {
                onSpeechIdle()
              }
            },
            awaitCompletion = false,
          )
        }.onFailure { t ->
          var shouldMarkIdle = false
          synchronized(lock) {
            if (localSessionId == sessionId && pendingUtteranceCount > 0) {
              pendingUtteranceCount -= 1
              shouldMarkIdle = pendingUtteranceCount == 0
            }
          }
          if (shouldMarkIdle) {
            onSpeechIdle()
          }
          Log.e(TAG, "queue chunk failed session=$localSessionId message=${t.message}", t)
        }
      }
    }
  }

  private fun extractSpeakableChunks(finalFlush: Boolean): List<String> {
    val chunks = mutableListOf<String>()
    while (true) {
      val sentenceBoundary = findSentenceBoundary(pendingText)
      if (sentenceBoundary >= 0) {
        val chunk = pendingText.substring(0, sentenceBoundary + 1).trim()
        pendingText.delete(0, sentenceBoundary + 1)
        if (chunk.isNotEmpty()) {
          chunks += chunk
        }
        continue
      }

      if (!finalFlush && pendingText.length >= FORCE_FLUSH_CHARS) {
        val flushAt = findForceFlushIndex(pendingText, FORCE_FLUSH_CHARS)
        val chunk = pendingText.substring(0, flushAt).trim()
        pendingText.delete(0, flushAt)
        if (chunk.isNotEmpty()) {
          chunks += chunk
        }
        continue
      }

      if (finalFlush && pendingText.isNotBlank()) {
        val chunk = pendingText.toString().trim()
        pendingText.clear()
        if (chunk.isNotEmpty()) {
          chunks += chunk
        }
      }
      break
    }
    return chunks
  }

  private fun findSentenceBoundary(buffer: StringBuilder): Int {
    for (i in buffer.indices) {
      val c = buffer[i]
      if (c == '.' || c == '!' || c == '?') {
        return i
      }
    }
    return -1
  }

  private fun findForceFlushIndex(buffer: StringBuilder, threshold: Int): Int {
    val capped = threshold.coerceAtMost(buffer.length)
    for (i in capped downTo 1) {
      if (buffer[i - 1].isWhitespace()) {
        return i
      }
    }
    return capped
  }
}
