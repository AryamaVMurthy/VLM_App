package com.qidk.fastvlm.core.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileOutputStream

class ImagePreprocessor(
  private val extraRotationDegrees: Int = 0,
  private val targetLongestEdge: Int = 768,
  private val jpegQuality: Int = 90,
) {

  fun preprocessForFastVlm(inputFile: File, outputDir: File): File {
    if (!inputFile.exists()) {
      throw IllegalStateException(
        "Input image '${inputFile.absolutePath}' does not exist. Remediation: capture a fresh frame and retry.",
      )
    }

    if (!outputDir.exists() && !outputDir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create preprocessed directory '${outputDir.absolutePath}'. Remediation: free app storage and retry.",
      )
    }

    val decoded = decodeScaledBitmap(inputFile, targetLongestEdge)

    val oriented = applyExifRotation(decoded, inputFile)
    val compensated = applyFixedRotation(oriented, extraRotationDegrees)
    val resized = resizeLongestEdge(compensated, targetLongestEdge)

    val outputFile = File(outputDir, "preprocessed_${System.currentTimeMillis()}.jpg")
    FileOutputStream(outputFile).use { stream ->
      val ok = resized.compress(Bitmap.CompressFormat.JPEG, jpegQuality, stream)
      if (!ok) {
        throw IllegalStateException(
          "Failed to encode preprocessed image '${outputFile.absolutePath}'. Remediation: retry with a new frame.",
        )
      }
    }

    if (decoded != oriented) decoded.recycle()
    if (oriented != compensated) oriented.recycle()
    if (compensated != resized) compensated.recycle()
    resized.recycle()

    return outputFile
  }

  private fun decodeScaledBitmap(file: File, targetLongestEdge: Int): Bitmap {
    val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.absolutePath, boundsOptions)
    val sourceWidth = boundsOptions.outWidth
    val sourceHeight = boundsOptions.outHeight
    if (sourceWidth <= 0 || sourceHeight <= 0) {
      throw IllegalStateException(
        "Failed to decode captured image '${file.absolutePath}'. Remediation: capture a new frame and retry.",
      )
    }

    var inSampleSize = 1
    val longestEdge = maxOf(sourceWidth, sourceHeight)
    while (longestEdge / (inSampleSize * 2) >= targetLongestEdge) {
      inSampleSize *= 2
    }

    val decodeOptions =
      BitmapFactory.Options().apply {
        this.inSampleSize = inSampleSize
        inPreferredConfig = Bitmap.Config.ARGB_8888
      }

    return BitmapFactory.decodeFile(file.absolutePath, decodeOptions)
      ?: throw IllegalStateException(
        "Failed to decode captured image '${file.absolutePath}'. Remediation: capture a new frame and retry.",
      )
  }

  private fun applyExifRotation(bitmap: Bitmap, file: File): Bitmap {
    val exif = ExifInterface(file.absolutePath)
    val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    val matrix = Matrix()

    when (orientation) {
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
      else -> return bitmap
    }

    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
  }

  private fun resizeLongestEdge(bitmap: Bitmap, targetLongestEdge: Int): Bitmap {
    val width = bitmap.width
    val height = bitmap.height
    val longest = maxOf(width, height)
    if (longest <= targetLongestEdge) {
      return bitmap
    }

    val scale = targetLongestEdge.toFloat() / longest.toFloat()
    val targetWidth = (width * scale).toInt().coerceAtLeast(1)
    val targetHeight = (height * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
  }

  private fun applyFixedRotation(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    val normalized = ((rotationDegrees % 360) + 360) % 360
    if (normalized == 0) return bitmap
    val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
  }
}
