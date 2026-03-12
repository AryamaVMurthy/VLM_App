package com.qidk.fastvlm.core.text

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.LogSeverity
import com.google.ai.edge.litertlm.MessageSendOptions
import com.google.ai.edge.litertlm.SamplerConfig
import com.qidk.fastvlm.core.graphpilot.GraphPilotStreamingTextGenerator
import com.qidk.fastvlm.core.graphpilot.GraphPilotTextGenerationResult
import com.qidk.fastvlm.core.litert.LiteRtGpuRuntime
import com.qidk.fastvlm.core.litert.LiteRtNpuRuntime
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

enum class TextStageRole {
  PLANNER,
  RESPONDER,
}

data class TextStageModelSpec(
  val role: TextStageRole,
  val backend: Backend,
  val artifact: String,
  val mirrorPath: String,
  val maxNumTokens: Int,
  val systemInstruction: String,
) {
  companion object {
    fun planner(backend: Backend): TextStageModelSpec {
      return TextStageModelSpec(
        role = TextStageRole.PLANNER,
        backend = backend,
        artifact = artifactForBackend(backend),
        mirrorPath = mirrorPathForBackend(backend),
        maxNumTokens = 512,
        systemInstruction = "You are a concise planner. Answer in one short sentence.",
      )
    }

    fun responder(backend: Backend): TextStageModelSpec {
      return TextStageModelSpec(
        role = TextStageRole.RESPONDER,
        backend = backend,
        artifact = artifactForBackend(backend),
        mirrorPath = mirrorPathForBackend(backend),
        maxNumTokens = 1024,
        systemInstruction = "Answer directly and briefly.",
      )
    }

    private fun artifactForBackend(backend: Backend): String {
      return if (backend == Backend.NPU) {
        "Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm"
      } else {
        "gemma3-1b-it-int4.litertlm"
      }
    }

    private fun mirrorPathForBackend(backend: Backend): String {
      return "/data/local/tmp/graphpilot_edge/${artifactForBackend(backend)}"
    }
  }
}

class LiteRtLmTextStageAdapter(private val context: Context) : AutoCloseable, GraphPilotStreamingTextGenerator {
  companion object {
    private const val TAG = "LiteRtLmTextStage"
    private val SAMPLER = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)

    init {
      Engine.setNativeMinLogSeverity(LogSeverity.WARNING)
    }
  }

  private val lock = Mutex()
  private var activeSpec: TextStageModelSpec? = null
  private var engine: Engine? = null

  suspend fun ensureInitialized(spec: TextStageModelSpec): String = lock.withLock {
    if (activeSpec == spec && engine?.isInitialized() == true) {
      return@withLock requireNotNull(engine).engineConfig.modelPath
    }
    closeInternal()
    val modelPath = provisionModel(spec)
    val runtimeLibraryDir =
      when (spec.backend) {
        Backend.NPU -> LiteRtNpuRuntime.ensurePrepared(context)
        Backend.GPU -> {
          LiteRtGpuRuntime.ensurePrepared(context)
          ""
        }
        Backend.CPU -> ""
      }
    @OptIn(ExperimentalApi::class)
    ExperimentalFlags.npuLibrariesDir = runtimeLibraryDir
    val cacheDir = File(context.cacheDir, "graphpilot_edge/${spec.role.name.lowercase()}_${spec.backend.name.lowercase()}")
    if (!cacheDir.exists() && !cacheDir.mkdirs()) {
      error("Failed to create cache dir '${cacheDir.absolutePath}'")
    }
    val newEngine =
      Engine(
        EngineConfig(
          modelPath = modelPath,
          backend = spec.backend,
          maxNumTokens = spec.maxNumTokens,
          cacheDir = cacheDir.absolutePath,
        )
      )
    withContext(Dispatchers.IO) {
      newEngine.initialize()
    }
    engine = newEngine
    activeSpec = spec
    Log.i(
      TAG,
      "Initialized text stage role=${spec.role} backend=${spec.backend} model_path=$modelPath max_tokens=${spec.maxNumTokens}",
    )
    return@withLock modelPath
  }

  suspend fun generate(prompt: String, maxOutputTokens: Int): String = lock.withLock {
    val currentSpec =
      activeSpec ?: error("Text stage is not initialized. Remediation: call ensureInitialized() first.")
    val activeEngine =
      engine ?: error("Engine handle missing for role=${currentSpec.role} backend=${currentSpec.backend}.")
    require(prompt.isNotBlank()) { "Prompt cannot be blank." }
    return@withLock withContext(Dispatchers.IO) {
      activeEngine
        .createConversation(
          ConversationConfig(
            systemInstruction = Contents.of(currentSpec.systemInstruction),
            samplerConfig = SAMPLER,
          )
        ).use { conversation ->
          val response =
            conversation.sendMessage(
              prompt.trim(),
              MessageSendOptions(maxOutputTokens = maxOutputTokens),
            )
          response.toString().trim().also { output ->
            check(output.isNotEmpty()) {
              "LiteRT-LM returned an empty response for role=${currentSpec.role} backend=${currentSpec.backend}."
            }
            Log.i(
              TAG,
              "Generated text role=${currentSpec.role} backend=${currentSpec.backend} output='${output.take(120)}'",
            )
          }
        }
    }
  }

  override suspend fun generateStreaming(
    prompt: String,
    maxOutputTokens: Int,
    onDelta: suspend (String) -> Unit,
  ): GraphPilotTextGenerationResult = lock.withLock {
    val currentSpec =
      activeSpec ?: error("Text stage is not initialized. Remediation: call ensureInitialized() first.")
    val activeEngine =
      engine ?: error("Engine handle missing for role=${currentSpec.role} backend=${currentSpec.backend}.")
    require(prompt.isNotBlank()) { "Prompt cannot be blank." }
    return@withLock withContext(Dispatchers.IO) {
      activeEngine
        .createConversation(
          ConversationConfig(
            systemInstruction = Contents.of(currentSpec.systemInstruction),
            samplerConfig = SAMPLER,
          )
        ).use { conversation ->
          val output = StringBuilder()
          var firstTokenAtMs: Long? = null
          conversation
            .sendMessageAsync(
              prompt.trim(),
              MessageSendOptions(maxOutputTokens = maxOutputTokens),
            ).collect { partialMessage ->
              val incoming = partialMessage.toString()
              val delta = computeStreamingDelta(output.toString(), incoming)
              if (delta.isBlank()) {
                return@collect
              }
              if (firstTokenAtMs == null) {
                firstTokenAtMs = System.currentTimeMillis()
              }
              output.append(delta)
              onDelta(delta)
            }
          val text = output.toString().trim()
          check(text.isNotEmpty()) {
            "LiteRT-LM streaming returned an empty response for role=${currentSpec.role} backend=${currentSpec.backend}."
          }
          check(firstTokenAtMs != null) {
            "LiteRT-LM streaming returned no non-empty deltas for role=${currentSpec.role} backend=${currentSpec.backend}."
          }
          Log.i(
            TAG,
            "Generated streamed text role=${currentSpec.role} backend=${currentSpec.backend} output='${text.take(120)}'",
          )
          GraphPilotTextGenerationResult(
            text = text,
            firstTokenAtMs = firstTokenAtMs,
          )
        }
    }
  }

  override fun close() {
    closeInternal()
  }

  private fun closeInternal() {
    engine?.close()
    engine = null
    activeSpec = null
  }

  private fun provisionModel(spec: TextStageModelSpec): String {
    val source = File(spec.mirrorPath)
    check(source.exists()) {
      "Missing text-stage model mirror '${source.absolutePath}'. Remediation: push ${spec.artifact} to /data/local/tmp/graphpilot_edge/ before running GraphPilot text stage tests."
    }
    val targetDir = File(context.filesDir, "graphpilot_edge/models/${spec.role.name.lowercase()}")
    if (!targetDir.exists() && !targetDir.mkdirs()) {
      error("Failed to create text-stage model dir '${targetDir.absolutePath}'")
    }
    val target = File(targetDir, spec.artifact)
    if (!target.exists() || target.length() != source.length()) {
      source.copyTo(target, overwrite = true)
    }
    return target.absolutePath
  }

  private fun computeStreamingDelta(emittedText: String, incoming: String): String {
    if (incoming.isEmpty()) {
      return ""
    }
    if (incoming.startsWith(emittedText)) {
      return incoming.removePrefix(emittedText)
    }
    if (emittedText.startsWith(incoming)) {
      return ""
    }
    return incoming
  }
}
