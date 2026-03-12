package com.qidk.fastvlm.core.model

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request

private const val TAG = "ModelManager"

class ModelProvisionException(message: String) : IllegalStateException(message)

@Serializable
data class ActiveModelPointer(
  val model_id: String,
  val artifact: String,
  val revision: String,
  val sha256: String,
  val active_path: String,
  val activated_at_ms: Long,
)

data class ModelActivationResult(
  val modelPath: String,
  val pointer: ActiveModelPointer,
)

class ModelManager(
  private val context: Context,
  private val client: OkHttpClient,
  private val json: Json,
) {

  private val modelRootDir: File by lazy { File(context.filesDir, "models") }

  suspend fun ensureModelReady(config: FastVlmModelConfig): ModelActivationResult = withContext(Dispatchers.IO) {
    val versionDir = File(modelRootDir, config.hf_revision)
    if (!versionDir.exists() && !versionDir.mkdirs()) {
      throw ModelProvisionException(
        "Failed to create model cache directory '${versionDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val artifactFile = File(versionDir, config.artifact)
    if (artifactFile.exists()) {
      val checksum = sha256(artifactFile)
      if (!checksum.equals(config.sha256, ignoreCase = true)) {
        throw ModelProvisionException(
          "Checksum mismatch for existing artifact '${artifactFile.absolutePath}'. expected='${config.sha256}', actual='${checksum}'. Remediation: delete the corrupted file and retry download.",
        )
      }
      Log.i(TAG, "Model already present and verified: ${artifactFile.absolutePath}")
      return@withContext activate(config, artifactFile)
    }

    if (!provisionFromLocalMirror(config, artifactFile)) {
      downloadAndVerify(config, artifactFile)
    }
    activate(config, artifactFile)
  }

  private fun activate(config: FastVlmModelConfig, artifactFile: File): ModelActivationResult {
    val pointer =
      ActiveModelPointer(
        model_id = config.model_id,
        artifact = config.artifact,
        revision = config.hf_revision,
        sha256 = config.sha256,
        active_path = artifactFile.absolutePath,
        activated_at_ms = System.currentTimeMillis(),
      )

    val pointerFile = File(modelRootDir, "active-pointer.json")
    pointerFile.writeText(json.encodeToString(ActiveModelPointer.serializer(), pointer))

    return ModelActivationResult(
      modelPath = artifactFile.absolutePath,
      pointer = pointer,
    )
  }

  private fun provisionFromLocalMirror(config: FastVlmModelConfig, destination: File): Boolean {
    val mirrorPath = config.local_mirror_path?.trim().orEmpty()
    if (mirrorPath.isEmpty()) {
      return false
    }
    val source = File(mirrorPath)
    if (!source.exists()) {
      throw ModelProvisionException(
        "Configured local mirror '${source.absolutePath}' is missing for artifact '${config.artifact}'. Remediation: stage the exact model artifact to the declared mirror path or update the config.",
      )
    }
    val checksum = sha256(source)
    if (!checksum.equals(config.sha256, ignoreCase = true)) {
      throw ModelProvisionException(
        "Checksum mismatch for local mirror '${source.absolutePath}'. expected='${config.sha256}', actual='${checksum}'. Remediation: replace the mirror with the pinned artifact.",
      )
    }
    if (source.length() != config.size_bytes) {
      throw ModelProvisionException(
        "Size mismatch for local mirror '${source.absolutePath}'. expected=${config.size_bytes}, actual=${source.length()}. Remediation: replace the mirror with the pinned artifact.",
      )
    }
    if (destination.exists() && !destination.delete()) {
      throw ModelProvisionException(
        "Failed to replace existing destination '${destination.absolutePath}' while provisioning from local mirror.",
      )
    }
    source.copyTo(destination, overwrite = true)
    Log.i(TAG, "Model provisioned from local mirror: ${source.absolutePath} -> ${destination.absolutePath}")
    return true
  }

  private fun downloadAndVerify(config: FastVlmModelConfig, destination: File) {
    val downloadUrl = config.download_url?.trim().orEmpty()
    if (downloadUrl.isEmpty()) {
      throw ModelProvisionException(
        "No download_url configured for artifact '${config.artifact}' and local mirror provisioning was unavailable. Remediation: stage the pinned artifact locally or add a pinned download URL.",
      )
    }
    val tempFile = File(destination.parentFile, "${destination.name}.download")
    if (tempFile.exists() && !tempFile.delete()) {
      throw ModelProvisionException(
        "Failed to delete stale temp file '${tempFile.absolutePath}'. Remediation: clear app storage and retry.",
      )
    }

    val request = Request.Builder().url(downloadUrl).get().build()
    try {
      client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) {
          throw ModelProvisionException(
            "Model download failed with HTTP ${response.code}. URL='${downloadUrl}'. Remediation: verify network access and URL revision pin.",
          )
        }

        val body = response.body
          ?: throw ModelProvisionException(
            "Model download returned empty body. Remediation: retry and verify Hugging Face model availability.",
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
          throw ModelProvisionException(
            "Downloaded size mismatch. expected=${config.size_bytes}, actual=${written}. Remediation: retry on stable network; verify pinned artifact revision.",
          )
        }

        val actualChecksum = digest.digest().joinToString("") { "%02x".format(it) }
        if (!actualChecksum.equals(config.sha256, ignoreCase = true)) {
          if (tempFile.exists()) {
            tempFile.delete()
          }
          throw ModelProvisionException(
            "Downloaded checksum mismatch. expected='${config.sha256}', actual='${actualChecksum}'. Remediation: verify pinned revision and checksum in config.",
          )
        }
      }
    } catch (e: IOException) {
      if (tempFile.exists()) {
        tempFile.delete()
      }
      throw ModelProvisionException(
        "Model download failed due to network I/O error: ${e.message}. Remediation: confirm device internet/DNS connectivity to huggingface.co and retry.",
      )
    }

    if (destination.exists() && !destination.delete()) {
      throw ModelProvisionException(
        "Failed to replace existing destination '${destination.absolutePath}'. Remediation: clear app storage and retry.",
      )
    }

    if (!tempFile.renameTo(destination)) {
      throw ModelProvisionException(
        "Failed to move downloaded model into place at '${destination.absolutePath}'. Remediation: ensure free disk and retry.",
      )
    }

    Log.i(TAG, "Model downloaded and verified: ${destination.absolutePath}")
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
