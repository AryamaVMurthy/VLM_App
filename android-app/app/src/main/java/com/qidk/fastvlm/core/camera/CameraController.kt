package com.qidk.fastvlm.core.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine

private const val TAG = "CameraController"

class CameraController(private val context: Context) {

  private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private var imageCapture: ImageCapture? = null
  @Volatile private var previewView: PreviewView? = null
  @Volatile private var lastBindError: String? = null
  private var isBound = false

  fun bind(
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    onReadyStateChanged: (ready: Boolean, reason: String?) -> Unit = { _, _ -> },
  ) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    cameraProviderFuture.addListener(
      {
        try {
          val cameraProvider = cameraProviderFuture.get()
          val preview =
            Preview.Builder().build().also {
              it.setSurfaceProvider(previewView.surfaceProvider)
            }
          this.previewView = previewView

          imageCapture =
            ImageCapture.Builder()
              .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
              .build()

          cameraProvider.unbindAll()
          val cameraSelectors = listOf(
            CameraSelector.DEFAULT_BACK_CAMERA,
            CameraSelector.DEFAULT_FRONT_CAMERA,
            CameraSelector.Builder().build(),
          )
          var bindFailure: Throwable? = null
          for (selector in cameraSelectors) {
            try {
              cameraProvider.bindToLifecycle(
                lifecycleOwner,
                selector,
                preview,
                imageCapture,
              )
              bindFailure = null
              break
            } catch (t: Throwable) {
              bindFailure = t
              Log.w(TAG, "Camera bind attempt failed for selector=$selector", t)
            }
          }
          if (bindFailure != null) {
            throw bindFailure
          }
          isBound = true
          lastBindError = null
          onReadyStateChanged(true, null)
        } catch (t: Throwable) {
          isBound = false
          imageCapture = null
          this.previewView = null
          lastBindError = "Camera bind failed: ${t.message}"
          Log.e(TAG, "Camera bind failed", t)
          onReadyStateChanged(
            false,
            "Camera bind failed: ${t.message}. Remediation: grant camera permission and ensure no other app holds exclusive camera lock.",
          )
        }
      },
      ContextCompat.getMainExecutor(context),
    )
  }

  fun isReady(): Boolean = isBound && imageCapture != null

  fun readinessFailureReason(): String? = lastBindError

  suspend fun captureFrameToFile(outputDir: File): File {
    if (!isBound || imageCapture == null) {
      throw IllegalStateException(
        "Camera is not ready. Remediation: wait for preview initialization before requesting VQA. ${lastBindError.orEmpty()}",
      )
    }

    if (!outputDir.exists() && !outputDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create capture directory '${outputDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val outputFile = File(outputDir, "frame_${System.currentTimeMillis()}.jpg")
    val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()

    return suspendCancellableCoroutine { continuation ->
      imageCapture!!.takePicture(
        outputOptions,
        cameraExecutor,
        object : ImageCapture.OnImageSavedCallback {
          override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
            continuation.resume(outputFile)
          }

          override fun onError(exception: ImageCaptureException) {
            continuation.resumeWithException(
              IllegalStateException(
                "Frame capture failed: ${exception.message}. Remediation: ensure camera stream is active and storage is writable.",
                exception,
              )
            )
          }
        },
      )
    }
  }

  suspend fun capturePreviewBitmapToFile(outputDir: File, quality: Int = 88): File {
    if (!isBound || previewView == null) {
      throw IllegalStateException(
        "Preview is not ready. Remediation: wait for camera preview initialization before capturing preview frame. ${lastBindError.orEmpty()}",
      )
    }

    if (!outputDir.exists() && !outputDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create preview capture directory '${outputDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val bitmap =
      suspendCancellableCoroutine<Bitmap> { continuation ->
        ContextCompat.getMainExecutor(context).execute {
          val currentPreview = previewView
          if (currentPreview == null) {
            continuation.resumeWithException(
              IllegalStateException(
                "Preview view is unavailable during capture. Remediation: keep camera preview visible and retry.",
              )
            )
            return@execute
          }
          val snapshot = currentPreview.bitmap
          if (snapshot == null) {
            continuation.resumeWithException(
              IllegalStateException(
                "Preview bitmap capture failed. Remediation: wait for live preview frames and retry.",
              )
            )
            return@execute
          }
          continuation.resume(snapshot)
        }
      }

    val outputFile = File(outputDir, "preview_${System.currentTimeMillis()}.jpg")
    FileOutputStream(outputFile).use { stream ->
      val ok = bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
      if (!ok) {
        throw IllegalStateException(
          "Preview bitmap encoding failed for '${outputFile.absolutePath}'. Remediation: retry with a fresh frame.",
        )
      }
    }
    bitmap.recycle()
    return outputFile
  }

  fun shutdown() {
    cameraExecutor.shutdown()
  }
}
