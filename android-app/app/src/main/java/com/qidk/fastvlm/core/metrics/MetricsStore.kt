package com.qidk.fastvlm.core.metrics

import android.content.Context
import android.util.Log
import com.qidk.fastvlm.core.model.TransitionTimings
import com.qidk.fastvlm.core.model.VqaMetrics
import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

private const val TAG = "MetricsStore"

@Serializable
data class BenchmarkRunSummary(
  val benchmark_id: String,
  val created_at_ms: Long,
  val question: String,
  val run_count: Int,
  val avg_ttft_ms: Double,
  val avg_decode_toks_per_sec: Double,
  val avg_total_ms: Double,
  val requests: List<Long>,
)

class MetricsStore(
  private val context: Context,
  private val json: Json,
) {
  private val root by lazy { File(context.filesDir, "metrics") }
  private val requestDir by lazy { File(root, "requests") }
  private val benchmarkDir by lazy { File(root, "benchmarks") }

  init {
    ensureDir(root)
    ensureDir(requestDir)
    ensureDir(benchmarkDir)
  }

  fun persistRequestMetrics(metrics: VqaMetrics, attemptTag: String? = null) {
    val file = requestMetricsFile(metrics.request_id, attemptTag)
    file.writeText(json.encodeToString(VqaMetrics.serializer(), metrics))
    Log.i(TAG, "Saved request metrics: ${file.absolutePath}")
  }

  fun updateRequestTransitionTimings(
    requestId: Long,
    update: (TransitionTimings) -> TransitionTimings,
  ): VqaMetrics {
    val file = requestMetricsFile(requestId, attemptTag = null)
    if (!file.exists()) {
      throw IllegalStateException(
        "Metrics file '${file.absolutePath}' not found for request_id=$requestId. " +
          "Remediation: run inference to completion before updating transition timings.",
      )
    }
    val existing =
      runCatching { json.decodeFromString(VqaMetrics.serializer(), file.readText()) }
        .getOrElse { t ->
          throw IllegalStateException(
            "Failed to decode metrics file '${file.absolutePath}' for request_id=$requestId: ${t.message}. " +
              "Remediation: clear app metrics storage and rerun request.",
            t,
          )
        }
    val updated = existing.copy(transition_timings = update(existing.transition_timings))
    file.writeText(json.encodeToString(VqaMetrics.serializer(), updated))
    Log.i(TAG, "Updated transition timings for request_id=$requestId path=${file.absolutePath}")
    return updated
  }

  fun persistBenchmarkSummary(summary: BenchmarkRunSummary): Pair<File, File> {
    val jsonFile = File(benchmarkDir, "${summary.benchmark_id}.json")
    jsonFile.writeText(json.encodeToString(BenchmarkRunSummary.serializer(), summary))

    val textFile = File(benchmarkDir, "${summary.benchmark_id}.txt")
    textFile.writeText(
      buildString {
        appendLine("benchmark_id=${summary.benchmark_id}")
        appendLine("created_at_ms=${summary.created_at_ms}")
        appendLine("run_count=${summary.run_count}")
        appendLine("question=${summary.question}")
        appendLine("avg_ttft_ms=${"%.2f".format(summary.avg_ttft_ms)}")
        appendLine("avg_decode_toks_per_sec=${"%.2f".format(summary.avg_decode_toks_per_sec)}")
        appendLine("avg_total_ms=${"%.2f".format(summary.avg_total_ms)}")
        appendLine("requests=${summary.requests.joinToString(",")}")
      }
    )

    Log.i(TAG, "Saved benchmark summary: ${jsonFile.absolutePath} and ${textFile.absolutePath}")
    return jsonFile to textFile
  }

  fun latestBenchmarkFile(): File? {
    return benchmarkDir
      .listFiles { file -> file.extension == "json" }
      ?.maxByOrNull { it.lastModified() }
  }

  fun exportLatestBenchmarkReport(): Pair<File, File> {
    val jsonFile =
      latestBenchmarkFile()
        ?: throw IllegalStateException(
          "No benchmark summary found to export. Remediation: run the 10x benchmark first.",
        )
    val textFile = File(benchmarkDir, "${jsonFile.nameWithoutExtension}.txt")
    if (!textFile.exists()) {
      throw IllegalStateException(
        "Missing benchmark text report '${textFile.absolutePath}'. Remediation: rerun benchmark to regenerate reports.",
      )
    }

    val externalRoot =
      context.getExternalFilesDir("benchmark_exports")
        ?: throw IllegalStateException(
          "External files directory is unavailable. Remediation: check storage availability and retry.",
        )
    ensureDir(externalRoot)

    val suffix = System.currentTimeMillis()
    val exportedJson = File(externalRoot, "${jsonFile.nameWithoutExtension}_$suffix.json")
    val exportedText = File(externalRoot, "${textFile.nameWithoutExtension}_$suffix.txt")
    jsonFile.copyTo(exportedJson, overwrite = true)
    textFile.copyTo(exportedText, overwrite = true)
    Log.i(TAG, "Exported benchmark report: ${exportedJson.absolutePath}, ${exportedText.absolutePath}")
    return exportedJson to exportedText
  }

  private fun ensureDir(dir: File) {
    if (!dir.exists() && !dir.mkdirs()) {
      throw IllegalStateException(
        "Failed to create metrics directory '${dir.absolutePath}'. Remediation: free app storage and restart app.",
      )
    }
  }

  private fun requestMetricsFile(requestId: Long, attemptTag: String?): File {
    val suffix = attemptTag?.trim().orEmpty()
    val safeSuffix = suffix.replace(Regex("[^a-zA-Z0-9_.-]"), "_").trim('_')
    val fileName =
      if (safeSuffix.isEmpty()) {
        "${requestId}.json"
      } else {
        "${requestId}_${safeSuffix}.json"
      }
    return File(requestDir, fileName)
  }
}
