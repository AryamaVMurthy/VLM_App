package com.qidk.fastvlm.core.graphpilot

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.LiteRtLmJni
import com.qidk.fastvlm.core.config.JsonCodec
import com.qidk.fastvlm.core.litert.LiteRtNpuRuntime
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable

private const val RETRIEVAL_TAG = "GraphPilotRetrieval"
private const val RETRIEVAL_MODEL = "/data/local/tmp/graphpilot_edge/embeddinggemma-300M_seq512_mixed-precision.tflite"
private const val RETRIEVAL_TOKENIZER = "/data/local/tmp/graphpilot_edge/tokenizer.model"
private const val RETRIEVAL_KB = "/data/local/tmp/graphpilot_edge/retrieval_kb.json"
private const val RETRIEVAL_GPU_DAEMON_HOST = "127.0.0.1"
private const val RETRIEVAL_GPU_DAEMON_PORT = 21910
private const val RETRIEVAL_GPU_DAEMON_PROTOCOL = "GRAPHPILOT_RETRIEVAL_V1"
private const val RETRIEVAL_GPU_DAEMON_CONNECT_TIMEOUT_MS = 2_000
private const val RETRIEVAL_GPU_DAEMON_REQUEST_TIMEOUT_MS = 60_000

internal data class RetrievalLibraryDirs(
  val runtimeLibraryDir: String,
  val dispatchLibraryDir: String,
)

internal fun buildRetrievalDaemonRequest(
  query: String,
  topK: Int,
  backend: Backend,
  requestId: Long,
): List<String> {
  require(query.isNotBlank()) { "Retrieval query cannot be blank." }
  require(topK > 0) { "topK must be positive." }
  require(backend == Backend.GPU) {
    "Retrieval daemon request is only defined for GPU backend."
  }
  return listOf(
    RETRIEVAL_GPU_DAEMON_PROTOCOL,
    requestId.toString(),
    Base64.getEncoder().encodeToString(query.toByteArray(Charsets.UTF_8)),
    topK.toString(),
  )
}

internal fun resolveRetrievalLibraryDirs(
  backend: Backend,
  gpuRuntimeLibraryDir: String,
  npuRuntimeLibraryDir: String,
): RetrievalLibraryDirs =
  when (backend) {
    Backend.CPU -> RetrievalLibraryDirs(runtimeLibraryDir = "", dispatchLibraryDir = "")
    Backend.GPU ->
      RetrievalLibraryDirs(
        runtimeLibraryDir = gpuRuntimeLibraryDir,
        dispatchLibraryDir = "",
      )
    Backend.NPU ->
      RetrievalLibraryDirs(
        runtimeLibraryDir = npuRuntimeLibraryDir,
        dispatchLibraryDir = npuRuntimeLibraryDir,
      )
  }

@Serializable
data class GraphPilotRetrievalHit(
  val doc_id: String,
  val title: String,
  val text: String,
  val score: Double,
)

@Serializable
data class GraphPilotRetrievalPayload(
  val mode: String,
  val accelerator: String,
  val sequence_length: Int,
  val query: String,
  val timings_ms: Map<String, Double>,
  val hits: List<GraphPilotRetrievalHit> = emptyList(),
)

class GraphPilotRetrievalBridge(
  private val context: Context,
) {
  private val json = JsonCodec.instance

  suspend fun retrieve(
    query: String,
    topK: Int = 2,
    backend: Backend = Backend.CPU,
  ): GraphPilotRetrievalPayload =
    withContext(Dispatchers.IO) {
      require(query.isNotBlank()) { "Retrieval query cannot be blank." }
      require(topK > 0) { "topK must be positive." }
      if (backend == Backend.GPU) {
        return@withContext runExternalGpuRetrieval(query, topK)
      }
      LiteRtLmJni.ensureLoaded()
      val npuRuntimeLibraryDir =
        when (backend) {
          Backend.NPU -> LiteRtNpuRuntime.ensurePrepared(context)
          Backend.CPU, Backend.GPU -> ""
        }
      val libraryDirs =
        resolveRetrievalLibraryDirs(
          backend = backend,
          gpuRuntimeLibraryDir = "",
          npuRuntimeLibraryDir = npuRuntimeLibraryDir,
        )
      val accelerator = backend.name.lowercase()
      val stdout =
        LiteRtLmJni.nativeGraphPilotRetrieve(
            modelPath = RETRIEVAL_MODEL,
            tokenizerPath = RETRIEVAL_TOKENIZER,
            kbPath = RETRIEVAL_KB,
            accelerator = accelerator,
            runtimeLibraryDir = libraryDirs.runtimeLibraryDir,
            dispatchLibraryDir = libraryDirs.dispatchLibraryDir,
            query = query,
            topK = topK,
          )
          .trim()
      parsePayload(stdout = stdout, accelerator = accelerator)
    }

  private fun runExternalGpuRetrieval(query: String, topK: Int): GraphPilotRetrievalPayload {
    val requestLines =
      buildRetrievalDaemonRequest(
        query = query,
        topK = topK,
        backend = Backend.GPU,
        requestId = System.currentTimeMillis(),
      )
    val socket = Socket()
    try {
      socket.connect(
        InetSocketAddress(RETRIEVAL_GPU_DAEMON_HOST, RETRIEVAL_GPU_DAEMON_PORT),
        RETRIEVAL_GPU_DAEMON_CONNECT_TIMEOUT_MS,
      )
      socket.soTimeout = RETRIEVAL_GPU_DAEMON_REQUEST_TIMEOUT_MS
      val writer = BufferedWriter(OutputStreamWriter(socket.getOutputStream(), Charsets.UTF_8))
      val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
      requestLines.forEach { line ->
        writer.write(line)
        writer.write("\n")
      }
      writer.flush()
      val response = mutableListOf<String>()
      while (true) {
        val line = reader.readLine() ?: break
        response += line
      }
      check(response.isNotEmpty()) {
        "GPU retrieval daemon returned no output. Remediation: inspect daemon logs and restart it with scripts/start_graphpilot_retrieval_gpu_daemon_adb.sh."
      }
      val lastLine = response.last().trim()
      check(!lastLine.startsWith("ERROR ")) {
        "GPU retrieval daemon failed: ${lastLine.removePrefix("ERROR ")}. Remediation: inspect daemon logs and restart it with scripts/start_graphpilot_retrieval_gpu_daemon_adb.sh."
      }
      return parsePayload(stdout = lastLine, accelerator = Backend.GPU.name.lowercase())
    } catch (e: Exception) {
      throw IllegalStateException(
        "Failed to query the GPU retrieval daemon at ${RETRIEVAL_GPU_DAEMON_HOST}:${RETRIEVAL_GPU_DAEMON_PORT}: ${e.message}. Remediation: run scripts/start_graphpilot_retrieval_gpu_daemon_adb.sh and retry.",
        e,
      )
    } finally {
      runCatching { socket.close() }
    }
  }

  private fun parsePayload(stdout: String, accelerator: String): GraphPilotRetrievalPayload {
    check(stdout.isNotEmpty()) {
      "GraphPilot retrieval returned empty stdout. Remediation: verify the staged EmbeddingGemma model, tokenizer.model, and retrieval_kb.json are readable from the app process."
    }
    val payload =
      try {
        json.decodeFromString<GraphPilotRetrievalPayload>(stdout.lineSequence().last())
      } catch (t: Throwable) {
        throw IllegalStateException(
          "Failed to parse retrieval JSON: ${t.message}. stdout='${stdout.take(256)}'",
          t,
        )
      }
    check(payload.mode == "retrieve") {
      "Unexpected retrieval mode '${payload.mode}'."
    }
    check(payload.accelerator.lowercase() == accelerator) {
      "Retrieval bridge expected backend='$accelerator', got accelerator='${payload.accelerator}'."
    }
    check(payload.hits.isNotEmpty()) {
      "Retrieval bridge returned zero hits. Remediation: rebuild retrieval_kb.json with the staged corpus and verify query embedding succeeds."
    }
    Log.i(
      RETRIEVAL_TAG,
      "GRAPH_PILOT_RETRIEVAL backend=${payload.accelerator.lowercase()} elapsed_ms=${payload.timings_ms["total"]?.toLong() ?: -1L} top_doc_id=${payload.hits.first().doc_id} top_score=${payload.hits.first().score} model=$RETRIEVAL_MODEL",
    )
    return payload
  }
}
