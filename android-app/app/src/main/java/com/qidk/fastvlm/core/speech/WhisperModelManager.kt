package com.qidk.fastvlm.core.speech

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request

class WhisperProvisionException(message: String) : IllegalStateException(message)

data class WhisperModelActivationResult(
  val modelPath: String,
)

class WhisperModelManager(
  private val context: Context,
  private val client: OkHttpClient,
) {
  companion object {
    private const val TAG = "WhisperModelManager"
  }

  private val modelRootDir: File by lazy { File(context.filesDir, "speech_models") }

  suspend fun ensureModelReady(config: WhisperModelConfig): WhisperModelActivationResult = withContext(Dispatchers.IO) {
    val versionDir = File(modelRootDir, config.hf_revision)
    if (!versionDir.exists() && !versionDir.mkdirs()) {
      throw WhisperProvisionException(
        "Failed to create whisper model cache directory '${versionDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val artifactFile = File(versionDir, config.artifact)
    if (artifactFile.exists()) {
      val checksum = sha256(artifactFile)
      if (!checksum.equals(config.sha256, ignoreCase = true)) {
        throw WhisperProvisionException(
          "Checksum mismatch for existing Whisper artifact '${artifactFile.absolutePath}'. expected='${config.sha256}', actual='${checksum}'. Remediation: delete corrupted file and retry.",
        )
      }
      return@withContext WhisperModelActivationResult(modelPath = artifactFile.absolutePath)
    }

    if (!provisionFromLocalMirror(config, artifactFile)) {
      downloadAndVerify(config, artifactFile)
    }
    WhisperModelActivationResult(modelPath = artifactFile.absolutePath)
  }

  private fun provisionFromLocalMirror(config: WhisperModelConfig, destination: File): Boolean {
    val mirrorPath = config.local_mirror_path?.trim().orEmpty()
    if (mirrorPath.isEmpty()) {
      return false
    }
    val source = File(mirrorPath)
    if (!source.exists()) {
      throw WhisperProvisionException(
        "Configured Whisper local mirror '${source.absolutePath}' is missing for artifact '${config.artifact}'. Remediation: stage the exact model artifact to the declared mirror path or update the config.",
      )
    }
    val checksum = sha256(source)
    if (!checksum.equals(config.sha256, ignoreCase = true)) {
      throw WhisperProvisionException(
        "Checksum mismatch for Whisper local mirror '${source.absolutePath}'. expected='${config.sha256}', actual='${checksum}'. Remediation: replace the mirror with the pinned artifact.",
      )
    }
    if (source.length() != config.size_bytes) {
      throw WhisperProvisionException(
        "Size mismatch for Whisper local mirror '${source.absolutePath}'. expected=${config.size_bytes}, actual=${source.length()}. Remediation: replace the mirror with the pinned artifact.",
      )
    }
    if (destination.exists() && !destination.delete()) {
      throw WhisperProvisionException(
        "Failed to replace existing Whisper destination '${destination.absolutePath}' while provisioning from local mirror.",
      )
    }
    source.copyTo(destination, overwrite = true)
    Log.i(TAG, "Whisper model provisioned from local mirror: ${source.absolutePath} -> ${destination.absolutePath}")
    return true
  }

  private fun downloadAndVerify(config: WhisperModelConfig, destination: File) {
    val downloadUrl = config.download_url?.trim().orEmpty()
    if (downloadUrl.isEmpty()) {
      throw WhisperProvisionException(
        "No download_url configured for Whisper artifact '${config.artifact}' and local mirror provisioning was unavailable. Remediation: stage the pinned artifact locally or add a pinned download URL.",
      )
    }
    val tempFile = File(destination.parentFile, "${destination.name}.download")
    if (tempFile.exists() && !tempFile.delete()) {
      throw WhisperProvisionException(
        "Failed to delete stale Whisper temp file '${tempFile.absolutePath}'. Remediation: clear app storage and retry.",
      )
    }

    val request = Request.Builder().url(downloadUrl).get().build()
    try {
      client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) {
          throw WhisperProvisionException(
            "Whisper model download failed with HTTP ${response.code}. URL='${downloadUrl}'. Remediation: verify network access and URL revision pin.",
          )
        }

        val body = response.body
          ?: throw WhisperProvisionException(
            "Whisper model download returned empty body. Remediation: retry and verify model availability.",
          )

        val digest = MessageDigest.getInstance("SHA-256")
        var written = 0L
        FileOutputStream(tempFile).use { output ->
          body.byteStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
              val read = input.read(buffer)
              if (read == -1) break
              output.write(buffer, 0, read)
              digest.update(buffer, 0, read)
              written += read
            }
          }
        }

        if (written != config.size_bytes) {
          if (tempFile.exists()) {
            tempFile.delete()
          }
          throw WhisperProvisionException(
            "Whisper downloaded size mismatch. expected=${config.size_bytes}, actual=${written}. Remediation: retry on stable network; verify pinned artifact revision.",
          )
        }

        val actualChecksum = digest.digest().joinToString("") { "%02x".format(it) }
        if (!actualChecksum.equals(config.sha256, ignoreCase = true)) {
          if (tempFile.exists()) {
            tempFile.delete()
          }
          throw WhisperProvisionException(
            "Whisper downloaded checksum mismatch. expected='${config.sha256}', actual='${actualChecksum}'. Remediation: verify pinned revision and checksum in config.",
          )
        }
      }
    } catch (e: IOException) {
      if (tempFile.exists()) {
        tempFile.delete()
      }
      throw WhisperProvisionException(
        "Whisper model download failed due to network I/O error: ${e.message}. Remediation: confirm device internet connectivity and retry.",
      )
    }

    if (destination.exists() && !destination.delete()) {
      throw WhisperProvisionException(
        "Failed to replace existing Whisper destination '${destination.absolutePath}'. Remediation: clear app storage and retry.",
      )
    }

    if (!tempFile.renameTo(destination)) {
      throw WhisperProvisionException(
        "Failed to move downloaded Whisper model into place at '${destination.absolutePath}'. Remediation: ensure free disk and retry.",
      )
    }
  }

  private fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
      val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
      while (true) {
        val read = input.read(buffer)
        if (read == -1) break
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }
}
