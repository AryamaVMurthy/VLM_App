package com.qidk.fastvlm.core.speech

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlin.math.max

private const val SAMPLE_RATE_HZ = 16_000
private const val DEFAULT_CHUNK_SIZE_MS = 1000
private const val RECORDER_STOP_TIMEOUT_MS = 2_000L

class PcmVoiceRecorder {
  private val scope: CoroutineScope =
    CoroutineScope(Executors.newSingleThreadExecutor().asCoroutineDispatcher())
  @Volatile private var recorder: AudioRecordThread? = null

  suspend fun startRecording(
    chunkDurationMs: Int = DEFAULT_CHUNK_SIZE_MS,
    onChunk: ((FloatArray) -> Unit)? = null,
  ) = withContext(scope.coroutineContext) {
    if (recorder != null) {
      throw IllegalStateException(
        "Voice recording already in progress. Remediation: stop current recording before starting a new one.",
      )
    }
    val next =
      if (onChunk == null) {
        AudioRecordThread(chunkSampleCount = 0, onChunk = null)
      } else {
        if (chunkDurationMs <= 0) {
          throw IllegalArgumentException(
            "Chunk duration must be greater than zero for streaming transcription.",
          )
        }
        AudioRecordThread(
          chunkSampleCount = (SAMPLE_RATE_HZ * max(1, chunkDurationMs) / 1000),
          onChunk = onChunk,
        )
      }
    recorder = next
    next.start()
  }

  suspend fun stopRecording(): FloatArray = withContext(scope.coroutineContext) {
    val active =
      recorder
        ?: throw IllegalStateException(
          "No active voice recording to stop. Remediation: start recording before stopping.",
        )
    active.stopRecording()
    @Suppress("BlockingMethodInNonBlockingContext")
    try {
      withTimeout(RECORDER_STOP_TIMEOUT_MS) {
        active.join()
      }
    } catch (t: TimeoutCancellationException) {
      throw IllegalStateException(
        "Voice recording stop timed out after ${RECORDER_STOP_TIMEOUT_MS}ms. Remediation: retry capture or restart app.",
        t,
      )
    }
    recorder = null
    active.consumeSamples()
  }

  fun hasActiveRecording(): Boolean = recorder != null

  suspend fun cancelRecording() = withContext(scope.coroutineContext) {
    val active = recorder ?: return@withContext
    active.stopRecording()
    @Suppress("BlockingMethodInNonBlockingContext")
    try {
      withTimeout(RECORDER_STOP_TIMEOUT_MS) {
        active.join()
      }
    } catch (_: TimeoutCancellationException) {
      // Best-effort stop. If stop takes too long, caller should fail visibly.
    }
    recorder = null
  }
}

private class AudioRecordThread(
  private val chunkSampleCount: Int,
  private val onChunk: ((FloatArray) -> Unit)? = null,
) : Thread("PcmVoiceRecorder") {
  private val quit = AtomicBoolean(false)
  private val allData = ArrayList<Short>(SAMPLE_RATE_HZ * 10)
  @Volatile private var failure: Throwable? = null
  @Volatile private var audioRecord: AudioRecord? = null
  private val chunkBuffer: ShortArray? =
    if (chunkSampleCount > 0) {
      ShortArray(chunkSampleCount)
    } else {
      null
    }
  private var chunkFill = 0

  @SuppressLint("MissingPermission")
  override fun run() {
    try {
      val minBuffer =
        AudioRecord.getMinBufferSize(
          SAMPLE_RATE_HZ,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
        )
      if (minBuffer <= 0) {
        throw IllegalStateException(
          "AudioRecord minimum buffer query failed with code=$minBuffer. Remediation: verify microphone is available and retry.",
        )
      }
      val bufferSize = minBuffer * 4
      val buffer = ShortArray(bufferSize / 2)

      val audioRecord =
        AudioRecord(
          MediaRecorder.AudioSource.VOICE_RECOGNITION,
          SAMPLE_RATE_HZ,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
          bufferSize,
        )
      this.audioRecord = audioRecord

      if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
        audioRecord.release()
        this.audioRecord = null
        throw IllegalStateException(
          "Failed to initialize AudioRecord. Remediation: grant microphone permission and ensure no other app holds the microphone.",
        )
      }

      try {
        audioRecord.startRecording()
        while (!quit.get()) {
          val read = audioRecord.read(buffer, 0, buffer.size)
          if (read <= 0) {
            if (quit.get()) {
              break
            }
            if (read == 0) {
              continue
            }
            throw IllegalStateException(
              "AudioRecord.read returned $read. Remediation: retry recording and ensure microphone access remains active.",
            )
          }
          for (i in 0 until read) {
            allData.add(buffer[i])
            onChunk?.let { chunk ->
              val localChunkBuffer = chunkBuffer ?: return@let
              localChunkBuffer[chunkFill++] = buffer[i]
              if (chunkFill >= localChunkBuffer.size) {
                emitChunk(localChunkBuffer, localChunkBuffer.size)
                chunkFill = 0
              }
            }
          }
        }
        onChunk?.let { _ ->
          chunkBuffer?.let { buffer ->
            if (chunkFill > 0) {
              emitChunk(buffer, chunkFill)
              chunkFill = 0
            }
          }
        }
        audioRecord.stop()
      } finally {
        runCatching { audioRecord.stop() }
        this.audioRecord = null
        audioRecord.release()
      }
    } catch (t: Throwable) {
      failure = t
    }
  }

  fun stopRecording() {
    quit.set(true)
    runCatching {
      val activeRecord = audioRecord
      if (activeRecord != null && activeRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        activeRecord.stop()
      }
    }
  }

  fun consumeSamples(): FloatArray {
    failure?.let { throw IllegalStateException(it.message ?: "Recording failed", it) }
    if (allData.isEmpty()) {
      throw IllegalStateException(
        "No audio captured from microphone. Remediation: hold record button while speaking and retry.",
      )
    }
    return FloatArray(allData.size) { index ->
      (allData[index] / 32767.0f).coerceIn(-1f, 1f)
    }
  }

  private fun emitChunk(source: ShortArray, length: Int) {
    if (length <= 0) return
    val chunk = FloatArray(length) { idx ->
      (source[idx] / 32767.0f).coerceIn(-1f, 1f)
    }
    try {
      onChunk?.invoke(chunk)
    } catch (_: Throwable) {
      // Non-fatal: chunk callback is best-effort. Recording continues.
    }
  }
}
