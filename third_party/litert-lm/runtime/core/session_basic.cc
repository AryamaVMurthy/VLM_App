// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/core/session_basic.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

using TaskController = Engine::Session::TaskController;

absl::Status ApplyFloatPruningOverride(absl::string_view env_name,
                                       float& destination) {
  const char* value = std::getenv(std::string(env_name).c_str());
  if (value == nullptr || value[0] == '\0') {
    return absl::OkStatus();
  }
  float parsed_value = 0.0f;
  if (!absl::SimpleAtof(value, &parsed_value)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid float value for ", env_name, ": '", value, "'."));
  }
  destination = parsed_value;
  ABSL_LOG(INFO) << "Applied visual token pruning override " << env_name
                 << "=" << parsed_value;
  return absl::OkStatus();
}

absl::Status ApplyIntPruningOverride(absl::string_view env_name,
                                     int& destination) {
  const char* value = std::getenv(std::string(env_name).c_str());
  if (value == nullptr || value[0] == '\0') {
    return absl::OkStatus();
  }
  int parsed_value = 0;
  if (!absl::SimpleAtoi(value, &parsed_value)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid int value for ", env_name, ": '", value, "'."));
  }
  destination = parsed_value;
  ABSL_LOG(INFO) << "Applied visual token pruning override " << env_name
                 << "=" << parsed_value;
  return absl::OkStatus();
}

absl::Status ApplyBoolPruningOverride(absl::string_view env_name,
                                      bool& destination) {
  const char* value = std::getenv(std::string(env_name).c_str());
  if (value == nullptr || value[0] == '\0') {
    return absl::OkStatus();
  }
  const absl::string_view normalized(value);
  if (normalized == "1" || absl::EqualsIgnoreCase(normalized, "true") ||
      absl::EqualsIgnoreCase(normalized, "yes")) {
    destination = true;
  } else if (normalized == "0" ||
             absl::EqualsIgnoreCase(normalized, "false") ||
             absl::EqualsIgnoreCase(normalized, "no")) {
    destination = false;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid bool value for ", env_name, ": '", value, "'."));
  }
  ABSL_LOG(INFO) << "Applied visual token pruning override " << env_name
                 << "=" << (destination ? "true" : "false");
  return absl::OkStatus();
}

absl::StatusOr<VisionTokenPruningConfig> CreateVisionTokenPruningConfig(
    const SessionConfig& session_config) {
  VisionTokenPruningConfig config;
  ASSIGN_OR_RETURN(
      config.strategy,
      ParseVisionTokenPruningStrategy(
          session_config.GetVisualTokenPruningStrategy()));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_PROMPT_SIMILARITY_WEIGHT",
      config.prompt_similarity_weight));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_SALIENCE_WEIGHT", config.salience_weight));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_REDUNDANCY_PENALTY_WEIGHT",
      config.redundancy_penalty_weight));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_PROMPT_ATTENTION_LOGIT_SCALE",
      config.prompt_attention_logit_scale));
  RETURN_IF_ERROR(ApplyIntPruningOverride(
      "LITERT_LM_PRUNING_PROMPT_ATTENTION_TOP_K",
      config.prompt_attention_top_k));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_LOCAL_REFINEMENT_MIN_PROMPT_GAIN",
      config.local_refinement_min_prompt_gain));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_FRACTION",
      config.max_local_refinement_fraction));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_SALIENCE_DROP",
      config.max_local_refinement_salience_drop));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_MIN_GLOBAL_MEAN_PROMPT_SIMILARITY",
      config.min_global_mean_prompt_similarity));
  RETURN_IF_ERROR(ApplyFloatPruningOverride(
      "LITERT_LM_PRUNING_MIN_GLOBAL_MEAN_SALIENCE",
      config.min_global_mean_salience));
  RETURN_IF_ERROR(ApplyBoolPruningOverride(
      "LITERT_LM_PRUNING_FUSE_PROJECTION_PRUNE_PACK",
      config.fuse_projection_prune_pack));
  return config;
}

bool UsesPromptConditionedPruning(
    const std::optional<VisionTokenPruningConfig>& pruning_config) {
  return pruning_config.has_value() &&
         (pruning_config->strategy ==
              VisionTokenPruningStrategy::kPromptConditionedV1 ||
          pruning_config->strategy ==
              VisionTokenPruningStrategy::kPromptConditionedV2);
}

bool IsVisionSummaryLoggingEnabled() {
  const char* value = std::getenv("LITERT_LM_DEBUG_VISION_SUMMARY");
  if (value == nullptr) {
    return false;
  }
  const absl::string_view normalized(value);
  return normalized == "1" || absl::EqualsIgnoreCase(normalized, "true") ||
         absl::EqualsIgnoreCase(normalized, "yes");
}

bool IsDetailedHandoffLoggingEnabled() {
  const char* value = std::getenv("LITERT_LM_DEBUG_HANDOFF_DETAIL");
  if (value == nullptr) {
    return false;
  }
  const absl::string_view normalized(value);
  return normalized == "1" || absl::EqualsIgnoreCase(normalized, "true") ||
         absl::EqualsIgnoreCase(normalized, "yes");
}

absl::Status AppendTextTokenEmbeddingsToCache(
    EmbeddingLookupManager& embedding_lookup_manager,
    absl::Span<const int> text_token_ids,
    CachedTextEmbeddings& cached_text_embeddings) {
  auto* text_embedding_lookup =
      embedding_lookup_manager.GetTextEmbeddingLookup();
  if (text_embedding_lookup == nullptr) {
    return absl::FailedPreconditionError(
        "Prompt-conditioned pruning requires a real text embedding lookup, "
        "but the embedding lookup manager does not own one.");
  }

  const int floats_per_token =
      static_cast<int>(text_embedding_lookup->GetFloatsPerToken());
  if (cached_text_embeddings.token_count == 0) {
    cached_text_embeddings.floats_per_token = floats_per_token;
  } else if (cached_text_embeddings.floats_per_token != floats_per_token) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cached text embedding dimension changed from ",
        cached_text_embeddings.floats_per_token, " to ", floats_per_token,
        " while appending prompt text embeddings."));
  }

  std::vector<float> token_embedding(floats_per_token, 0.0f);
  for (const int token_id : text_token_ids) {
    if (token_id < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Prompt text token ids must be non-negative, but got: ", token_id));
    }
    RETURN_IF_ERROR(
        embedding_lookup_manager.LookupPrefill(token_id, token_embedding));
    cached_text_embeddings.values.insert(cached_text_embeddings.values.end(),
                                         token_embedding.begin(),
                                         token_embedding.end());
    ++cached_text_embeddings.token_count;
  }
  return cached_text_embeddings.Validate();
}

absl::string_view ExtractPromptConditioningText(absl::string_view raw_text) {
  raw_text = absl::StripAsciiWhitespace(raw_text);
  if (!absl::StartsWithIgnoreCase(raw_text, "Question:")) {
    return raw_text;
  }
  raw_text.remove_prefix(std::string_view("Question:").size());
  raw_text = absl::StripLeadingAsciiWhitespace(raw_text);
  const size_t newline = raw_text.find('\n');
  if (newline != absl::string_view::npos) {
    raw_text = raw_text.substr(0, newline);
  }
  return absl::StripAsciiWhitespace(raw_text);
}

absl::StatusOr<std::optional<CachedTextEmbeddings>>
BuildPromptConditioningCacheFromContents(
    const std::vector<InputData>& contents, Tokenizer& tokenizer,
    EmbeddingLookupManager& embedding_lookup_manager) {
  std::optional<CachedTextEmbeddings> cached_text_embeddings = std::nullopt;
  for (const auto& content : contents) {
    const auto* input_text = std::get_if<InputText>(&content);
    if (input_text == nullptr) {
      continue;
    }

    std::vector<int> token_ids;
    if (input_text->IsTensorBuffer()) {
      ASSIGN_OR_RETURN(const auto* token_ids_tensor,
                       input_text->GetPreprocessedTextTensor());
      LITERT_ASSIGN_OR_RETURN(auto token_ids_span,
                              ReferTensorBufferAsSpan<int>(*token_ids_tensor));
      token_ids.assign(token_ids_span.begin(), token_ids_span.end());
    } else {
      ASSIGN_OR_RETURN(absl::string_view raw_text,
                       input_text->GetRawTextString());
      const absl::string_view conditioning_text =
          ExtractPromptConditioningText(raw_text);
      if (conditioning_text.empty()) {
        continue;
      }
      ASSIGN_OR_RETURN(token_ids, tokenizer.TextToTokenIds(conditioning_text));
    }
    if (token_ids.empty()) {
      continue;
    }
    if (!cached_text_embeddings.has_value()) {
      cached_text_embeddings = CachedTextEmbeddings();
    }
    RETURN_IF_ERROR(AppendTextTokenEmbeddingsToCache(
        embedding_lookup_manager, token_ids, *cached_text_embeddings));
  }
  return cached_text_embeddings;
}

uint64_t HashBytes(absl::Span<const uint8_t> bytes) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  return hash;
}

absl::StatusOr<std::string> SummarizeTensorBuffer(
    const ::litert::TensorBuffer& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto packed_size, tensor.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto tensor_copy, tensor.Duplicate());
  std::vector<uint8_t> bytes(packed_size);
  if (auto read_status = tensor_copy.Read(absl::MakeSpan(bytes));
      !read_status.HasValue()) {
    return absl::InternalError(read_status.Error().Message());
  }
  return absl::StrCat("bytes=", packed_size, " hash=", HashBytes(bytes),
                      " dims=[",
                      absl::StrJoin(tensor_type.Layout().Dimensions(), ","),
                      "] elem_type=",
                      static_cast<int>(tensor_type.ElementType()));
}

std::string SummarizeCachedTextEmbeddings(
    const CachedTextEmbeddings& cached_text_embeddings) {
  const auto* begin =
      reinterpret_cast<const uint8_t*>(cached_text_embeddings.values.data());
  const size_t byte_count =
      cached_text_embeddings.values.size() * sizeof(float);
  return absl::StrCat("token_count=", cached_text_embeddings.token_count,
                      " floats_per_token=",
                      cached_text_embeddings.floats_per_token, " bytes=",
                      byte_count, " hash=",
                      HashBytes(absl::MakeConstSpan(begin, byte_count)));
}

absl::StatusOr<std::string> SummarizePrefillDecodeHandoff(
    const PrefillDecodeHandoff& handoff) {
  const auto* processed_begin =
      reinterpret_cast<const uint8_t*>(handoff.processed_token_ids.data());
  const size_t processed_bytes =
      handoff.processed_token_ids.size() * sizeof(int);
  const uint64_t processed_hash =
      HashBytes(absl::MakeConstSpan(processed_begin, processed_bytes));

  std::vector<std::string> cache_names;
  cache_names.reserve(handoff.kv_cache_buffers.size());
  for (const auto& [cache_name, _] : handoff.kv_cache_buffers) {
    cache_names.push_back(cache_name);
  }
  std::sort(cache_names.begin(), cache_names.end());

  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t kv_hash = 1469598103934665603ULL;
  for (const std::string& cache_name : cache_names) {
    const auto it = handoff.kv_cache_buffers.find(cache_name);
    if (it == handoff.kv_cache_buffers.end()) {
      return absl::InternalError(
          absl::StrCat("Missing KV cache buffer while summarizing handoff: ",
                       cache_name));
    }
    ASSIGN_OR_RETURN(const std::string tensor_summary,
                     SummarizeTensorBuffer(it->second));
    const auto* name_bytes =
        reinterpret_cast<const uint8_t*>(cache_name.data());
    kv_hash ^= HashBytes(
        absl::MakeConstSpan(name_bytes, cache_name.size()));
    kv_hash *= kFnvPrime;
    const auto* summary_bytes =
        reinterpret_cast<const uint8_t*>(tensor_summary.data());
    kv_hash ^= HashBytes(
        absl::MakeConstSpan(summary_bytes, tensor_summary.size()));
    kv_hash *= kFnvPrime;
  }

  return absl::StrCat("current_step=", handoff.current_step,
                      " last_prefill_token_id=",
                      handoff.last_prefill_token_id,
                      " processed_token_count=",
                      handoff.processed_token_ids.size(),
                      " processed_token_hash=", processed_hash,
                      " pending_token_id=", handoff.pending_token_id,
                      " kv_cache_count=", handoff.kv_cache_buffers.size(),
                      " kv_cache_hash=", kv_hash);
}

}

absl::flat_hash_set<LlmExecutor*>* SessionBasic::occupied_executors_ =
    new absl::flat_hash_set<LlmExecutor*>();
ABSL_CONST_INIT absl::Mutex SessionBasic::occupied_executors_mu_(
    absl::kConstInit);

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    VisionExecutor* vision_executor, AudioExecutor* audio_executor,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
  return Create(executor, tokenizer, vision_executor, audio_executor,
                /*prompt_embedding_lookup_manager=*/nullptr, session_config,
                std::move(benchmark_info), worker_thread_pool);
}

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    VisionExecutor* vision_executor, AudioExecutor* audio_executor,
    std::unique_ptr<EmbeddingLookupManager> prompt_embedding_lookup_manager,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
  // Check if the session already exists.
  absl::MutexLock lock(occupied_executors_mu_);  // NOLINT
  if (occupied_executors_->contains(executor)) {
    return absl::FailedPreconditionError(
        "A session already exists. Only one session is supported at a time. "
        "Please delete the existing session before creating a new one.");
  }
  auto sampler_backend = session_config.GetSamplerBackend();
  std::unique_ptr<Sampler> sampler;
  // If use CPU sampling, we create it here; For GPU sampling, we let executor
  // create it internally.
  if (sampler_backend == Backend::CPU) {
    ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(sampler_backend, session_config.GetNumOutputCandidates(),
                      session_config.GetSamplerParams()));
  } else if (sampler_backend != Backend::GPU &&
             sampler_backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend));
  }

  if (benchmark_info.has_value()) {
    ABSL_LOG(INFO) << "Benchmark is enabled.";
  }
  StopTokenDetector stop_token_detector(
      session_config.GetNumOutputCandidates());
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    RETURN_IF_ERROR(
        stop_token_detector.AddStopTokenSequence(stop_token_sequence));
  }

  occupied_executors_->insert(executor);
  return absl::WrapUnique(new SessionBasic(
      executor, tokenizer, vision_executor, audio_executor,
      std::move(prompt_embedding_lookup_manager), std::move(sampler),
      session_config, benchmark_info, worker_thread_pool,
      stop_token_detector));
}

SessionBasic::~SessionBasic() {
  auto status = executor_.Reset();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to reset executor: " << status;
  }
  if (audio_executor_ != nullptr) {
    status = audio_executor_->Reset();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to reset audio executor: " << status;
    }
  }
  absl::MutexLock lock(occupied_executors_mu_);  // NOLINT
  occupied_executors_->erase(&executor_);
}

absl::StatusOr<ExecutorInputs> SessionBasic::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents,
    const CachedTextEmbeddings* prompt_conditioning_cache) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  std::optional<CachedTextEmbeddings> prefill_cached_text_embeddings =
      std::nullopt;
  std::optional<VisionTokenPruningConfig> pruning_config = std::nullopt;
  if (session_config_.GetMaxVisualTokens() > 0) {
    ASSIGN_OR_RETURN(pruning_config,
                     CreateVisionTokenPruningConfig(session_config_));
  }
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* token_ids,
                       input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
      if (UsesPromptConditionedPruning(pruning_config)) {
        if (prompt_embedding_lookup_manager_ == nullptr &&
            prompt_conditioning_cache == nullptr) {
          return absl::FailedPreconditionError(absl::StrCat(
              VisionTokenPruningStrategyToString(pruning_config->strategy),
              " requires the real FastVLM text embedder, but no prompt "
              "embedding lookup manager is available."));
        }
        if (prompt_embedding_lookup_manager_ == nullptr) {
          continue;
        }
        if (!prefill_cached_text_embeddings.has_value()) {
          prefill_cached_text_embeddings = CachedTextEmbeddings();
        }
        if (benchmark_info_.has_value()) {
          RETURN_IF_ERROR(
              benchmark_info_->TimeMarkDelta("prompt_text_embedder"));
        }
        RETURN_IF_ERROR(AppendTextTokenEmbeddingsToCache(
            *prompt_embedding_lookup_manager_, ids_buffer_span,
            *prefill_cached_text_embeddings));
        if (benchmark_info_.has_value()) {
          RETURN_IF_ERROR(
              benchmark_info_->TimeMarkDelta("prompt_text_embedder"));
        }
      }
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* image_tensor,
                       input_image->GetPreprocessedImageTensor());
      if (image_tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Image tensor is null in preprocessed_contents.");
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor_->Encode(*image_tensor));
      const int max_visual_tokens = session_config_.GetMaxVisualTokens();
      if (max_visual_tokens > 0) {
        if (!debug_request_id_.empty() && IsVisionSummaryLoggingEnabled()) {
          ASSIGN_OR_RETURN(const auto* vision_embeddings,
                           single_image_data.GetEmbeddingsPtr());
          ASSIGN_OR_RETURN(const std::string vision_summary,
                           SummarizeTensorBuffer(*vision_embeddings));
          ABSL_LOG(INFO) << "Vision embedding summary: " << vision_summary
                         << " request_id=" << debug_request_id_;
          if (prompt_conditioning_cache != nullptr) {
            ABSL_LOG(INFO) << "Prompt conditioning cache summary: "
                           << SummarizeCachedTextEmbeddings(
                                  *prompt_conditioning_cache)
                           << " request_id=" << debug_request_id_;
          } else if (prefill_cached_text_embeddings.has_value()) {
            ABSL_LOG(INFO) << "Prompt conditioning cache summary: "
                           << SummarizeCachedTextEmbeddings(
                                  *prefill_cached_text_embeddings)
                           << " request_id=" << debug_request_id_;
          }
        }
        ASSIGN_OR_RETURN(const int original_image_token_num,
                         GetExecutorVisionTokenCount(single_image_data));
        if (benchmark_info_.has_value()) {
          RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_token_budget"));
        }
        ASSIGN_OR_RETURN(
            auto pruned_image_data,
            PruneExecutorVisionData(single_image_data, max_visual_tokens,
                                    prompt_conditioning_cache != nullptr
                                        ? prompt_conditioning_cache
                                        : (prefill_cached_text_embeddings
                                                   .has_value()
                                               ? &prefill_cached_text_embeddings
                                                      .value()
                                               : nullptr),
                                    *pruning_config));
        single_image_data = std::move(pruned_image_data.vision_data);
        if (benchmark_info_.has_value()) {
          RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_token_budget"));
        }
        ASSIGN_OR_RETURN(const int pruned_image_token_num,
                         GetExecutorVisionTokenCount(single_image_data));
        ABSL_LOG(INFO) << "Applied visual token budget: kept "
                       << pruned_image_token_num << " of "
                       << original_image_token_num
                       << " projected vision tokens."
                       << (debug_request_id_.empty()
                               ? ""
                               : absl::StrCat(" request_id=",
                                              debug_request_id_));
        ABSL_LOG(INFO) << "Visual token pruning decision: "
                       << pruned_image_data.decision.ToLogString()
                       << (debug_request_id_.empty()
                               ? ""
                               : absl::StrCat(" request_id=",
                                              debug_request_id_));
        if (pruning_config->fuse_projection_prune_pack &&
            single_image_data.GetSelectedTokenIndices().has_value()) {
          ABSL_LOG(INFO)
              << "Using fused projection-prune-pack sparse token view: kept "
              << single_image_data.GetSelectedTokenIndices()->size()
              << " projected tokens without materializing a sliced tensor."
              << (debug_request_id_.empty()
                      ? ""
                      : absl::StrCat(" request_id=", debug_request_id_));
        }
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(const int image_token_num,
                       GetExecutorVisionTokenCount(single_image_data));
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                       input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor_->Encode(*spectrogram_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported input data type in preprocessed_contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ASSIGN_OR_RETURN(combined_image_data,
                     CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ASSIGN_OR_RETURN(combined_audio_data,
                     CombineExecutorAudioData(all_audio_data));
  }

  ASSIGN_OR_RETURN(auto token_ids_buffer,
                   tokenizer_.TokenIdsToTensorBuffer(combined_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  if (prefill_cached_text_embeddings.has_value()) {
    RETURN_IF_ERROR(prefill_cached_text_embeddings->Validate());
    // Prompt-conditioned pruning uses real prompt embeddings for scoring only.
    // Do not attach them to the executor inputs, otherwise the pruning path
    // changes text-embedding execution and no longer matches the uniform path
    // when token selections are identical.
  }
  ExecutorInputs inputs(std::move(text_data), std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::Status SessionBasic::PrefillInternal(
    const std::vector<InputData>& preprocessed_contents,
    bool wait_for_completion,
    const CachedTextEmbeddings* prompt_conditioning_cache) {
  ASSIGN_OR_RETURN(ExecutorInputs inputs,
                   ProcessAndCombineContents(preprocessed_contents,
                                            prompt_conditioning_cache));
  ASSIGN_OR_RETURN(
      last_prefill_token_id_,
      Prefill(executor_, inputs, wait_for_completion, benchmark_info_));
  decode_prompt_finalized_ = false;
  session_state_ = SessionState::kPrefilled;
  return absl::OkStatus();
}

absl::Status SessionBasic::FinalizeDecodePromptIfNeeded() {
  if (decode_prompt_finalized_) {
    return absl::OkStatus();
  }
  if (!session_config_.GetApplyPromptTemplateInSession()) {
    decode_prompt_finalized_ = true;
    return absl::OkStatus();
  }
  std::vector<InputData> contents;
  contents.emplace_back(InputText(""));
  ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                   ApplyPromptTemplates(contents, ContentType::kLast,
                                        session_config_, tokenizer_,
                                        /*is_first_turn=*/false));
  if (templated_contents.empty()) {
    decode_prompt_finalized_ = true;
    return absl::OkStatus();
  }
  ASSIGN_OR_RETURN(std::vector<InputData> preprocessed_contents,
                   PreprocessContents(templated_contents, session_config_,
                                      tokenizer_, benchmark_info_));
  RETURN_IF_ERROR(
      PrefillInternal(preprocessed_contents, /*wait_for_completion=*/true,
                      /*prompt_conditioning_cache=*/nullptr));
  decode_prompt_finalized_ = true;
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ABSL_LOG(INFO) << "RunPrefill: ";
  for (const auto& content : contents) {
    ABSL_LOG(INFO) << content;
  }

  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  std::optional<CachedTextEmbeddings> prompt_conditioning_cache = std::nullopt;
  if (session_config_.GetMaxVisualTokens() > 0) {
    ASSIGN_OR_RETURN(auto pruning_config,
                     CreateVisionTokenPruningConfig(session_config_));
    if (UsesPromptConditionedPruning(pruning_config) &&
        prompt_embedding_lookup_manager_ != nullptr) {
      ASSIGN_OR_RETURN(prompt_conditioning_cache,
                       BuildPromptConditioningCacheFromContents(
                           contents, tokenizer_, *prompt_embedding_lookup_manager_));
    }
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info_->TimeMarkDelta("session_preprocess_contents"));
  }
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(contents, session_config_, tokenizer_,
                                        benchmark_info_));
  } else {
    bool is_first_turn = session_state_ == SessionState::kFresh;
    ContentType content_type;
    if (session_config_.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, content_type, session_config_,
                             tokenizer_, is_first_turn));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents, session_config_,
                                        tokenizer_, benchmark_info_));
  }
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info_->TimeMarkDelta("session_preprocess_contents"));
  }

  return PrefillInternal(
      preprocessed_contents, /*wait_for_completion=*/true,
      prompt_conditioning_cache.has_value() ? &*prompt_conditioning_cache
                                            : nullptr);
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ABSL_LOG(INFO) << "RunPrefillAsync: ";
  for (const auto& content : contents) {
    ABSL_LOG(INFO) << content;
  }

  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  std::optional<CachedTextEmbeddings> prompt_conditioning_cache = std::nullopt;
  if (session_config_.GetMaxVisualTokens() > 0) {
    ASSIGN_OR_RETURN(auto pruning_config,
                     CreateVisionTokenPruningConfig(session_config_));
    if (UsesPromptConditionedPruning(pruning_config) &&
        prompt_embedding_lookup_manager_ != nullptr) {
      ASSIGN_OR_RETURN(prompt_conditioning_cache,
                       BuildPromptConditioningCacheFromContents(
                           contents, tokenizer_, *prompt_embedding_lookup_manager_));
    }
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info_->TimeMarkDelta("session_preprocess_contents"));
  }
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(contents, session_config_, tokenizer_,
                                        benchmark_info_));
  } else {
    bool is_first_turn = session_state_ == SessionState::kFresh;
    ContentType content_type;
    if (session_config_.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, content_type, session_config_,
                             tokenizer_, is_first_turn));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents, session_config_,
                                        tokenizer_, benchmark_info_));
  }
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(
        benchmark_info_->TimeMarkDelta("session_preprocess_contents"));
  }
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       prompt_conditioning_cache = std::move(prompt_conditioning_cache),
       callback = std::move(callback)]() mutable {
        absl::Status status = this->PrefillInternal(
            preprocessed_contents, /*wait_for_completion=*/false,
            prompt_conditioning_cache.has_value()
                ? &*prompt_conditioning_cache
                : nullptr);
        ABSL_LOG(INFO) << "RunPrefillAsync status: " << status;
        if (cancelled_.load()) {
          callback(
              absl::CancelledError("Session is cancelled during prefill."));
          return;
        }
        if (!status.ok()) {
          callback(status);
        } else {
          callback(Responses(TaskState::kDone));
        }
      }));
  return nullptr;
}

absl::StatusOr<Responses> SessionBasic::DecodeInternal(
    const DecodeConfig& decode_config) {
  if (session_state_ != SessionState::kPrefilled) {
    return absl::InternalError("Session is not prefilled yet.");
  }

  RETURN_IF_ERROR(FinalizeDecodePromptIfNeeded());
  session_state_ = SessionState::kDecoded;

  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(
        auto responses,
        Decode(executor_, tokenizer_, stop_token_detector_,
               session_config_.GetNumOutputCandidates(),
               decode_config.GetConstraint(), benchmark_info_, &cancelled_,
               decode_config.GetMaxOutputTokens().value_or(
                   session_config_.GetMaxOutputTokens())));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    LITERT_ASSIGN_OR_RETURN(
        auto decoded_ids_buffer,
        CopyToTensorBuffer<int>(decoded_ids,
                                {session_config_.GetNumOutputCandidates(), 1}));
    ASSIGN_OR_RETURN(
        auto responses,
        DecodeCustomSampling(executor_, tokenizer_, stop_token_detector_,
                             session_config_.GetNumOutputCandidates(),
                             *sampler_, std::move(decoded_ids_buffer),
                             decode_config.GetConstraint(), benchmark_info_,
                             &cancelled_,
                             decode_config.GetMaxOutputTokens().value_or(
                                 session_config_.GetMaxOutputTokens())));
    return responses;
  }
}

absl::Status SessionBasic::DecodeInternalStreaming(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(DecodeStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), decode_config.GetConstraint(),
        benchmark_info_, std::move(callback), &cancelled_,
        decode_config.GetMaxOutputTokens().value_or(
            session_config_.GetMaxOutputTokens())));
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    LITERT_ASSIGN_OR_RETURN(
        auto decoded_ids_buffer,
        CopyToTensorBuffer<int>(decoded_ids,
                                {session_config_.GetNumOutputCandidates(), 1}));

    RETURN_IF_ERROR(DecodeCustomSamplingStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), *sampler_,
        std::move(decoded_ids_buffer), decode_config.GetConstraint(),
        benchmark_info_, std::move(callback), &cancelled_,
        decode_config.GetMaxOutputTokens().value_or(
            session_config_.GetMaxOutputTokens())));
  }
  return absl::OkStatus();
}

absl::StatusOr<PrefillDecodeHandoff> SessionBasic::ExportPrefillDecodeHandoff() {
  if (session_state_ != SessionState::kPrefilled) {
    return absl::FailedPreconditionError(
        "Session must be prefilled before exporting decode handoff.");
  }
  RETURN_IF_ERROR(FinalizeDecodePromptIfNeeded());
  ASSIGN_OR_RETURN(auto handoff,
                   executor_.ExportPrefillDecodeHandoff(last_prefill_token_id_));
  if (!debug_request_id_.empty()) {
    ASSIGN_OR_RETURN(const std::string handoff_summary,
                     SummarizePrefillDecodeHandoff(handoff));
    ABSL_LOG(INFO) << "Exported prefill handoff summary: " << handoff_summary
                   << " request_id=" << debug_request_id_;
    if (IsDetailedHandoffLoggingEnabled()) {
      std::vector<std::string> cache_names;
      cache_names.reserve(handoff.kv_cache_buffers.size());
      for (const auto& [cache_name, _] : handoff.kv_cache_buffers) {
        cache_names.push_back(cache_name);
      }
      std::sort(cache_names.begin(), cache_names.end());
      for (const std::string& cache_name : cache_names) {
        const auto it = handoff.kv_cache_buffers.find(cache_name);
        if (it == handoff.kv_cache_buffers.end()) {
          return absl::InternalError(
              absl::StrCat("Missing KV cache buffer while logging handoff: ",
                           cache_name));
        }
        ASSIGN_OR_RETURN(const std::string tensor_summary,
                         SummarizeTensorBuffer(it->second));
        ABSL_LOG(INFO) << "Exported prefill handoff tensor: name="
                       << cache_name << " " << tensor_summary
                       << " request_id=" << debug_request_id_;
      }
    }
  }
  return handoff;
}

absl::Status SessionBasic::ImportPrefillDecodeHandoff(
    const PrefillDecodeHandoff& handoff) {
  if (session_state_ != SessionState::kFresh) {
    return absl::FailedPreconditionError(
        "Session must be fresh before importing decode handoff.");
  }
  RETURN_IF_ERROR(handoff.Validate());
  RETURN_IF_ERROR(executor_.ImportPrefillDecodeHandoff(handoff));
  last_prefill_token_id_ = handoff.last_prefill_token_id;
  session_state_ = SessionState::kPrefilled;
  decode_prompt_finalized_ = true;
  return absl::OkStatus();
}

absl::Status SessionBasic::ResetForReuse() {
  RETURN_IF_ERROR(executor_.Reset());
  cancelled_ = false;
  last_prefill_token_id_ = 0;
  decode_prompt_finalized_ = false;
  session_state_ = SessionState::kFresh;
  debug_request_id_.clear();
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionBasic::RunDecode(
    const DecodeConfig& decode_config) {
  ABSL_LOG(INFO) << "RunDecodeSync";
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  return DecodeInternal(decode_config);
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  ABSL_LOG(INFO) << "RunDecodeAsync";
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, callback = std::move(callback), decode_config]() mutable {
        this->DecodeInternalStreaming(std::move(callback), decode_config)
            .IgnoreError();
      }));
  return nullptr;
}

absl::StatusOr<Responses> SessionBasic::GenerateContent(
    const std::vector<InputData>& contents) {
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionBasic::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths) {
  absl::StatusOr<Responses> collected_responses;
  auto scoring_sync_callback =
      [&collected_responses](absl::StatusOr<Responses> responses) {
        collected_responses = std::move(responses);
      };

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunTextScoringAsync(target_text, std::move(scoring_sync_callback),
                          store_token_lengths));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
SessionBasic::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    bool store_token_lengths) {
  if (target_text.size() != 1) {
    return absl::InvalidArgumentError("Target text size should be 1.");
  }

  // TODO(b/435040163): Handle the temperature. Should it be calculated from
  // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
  // testing.
  auto temperature = 1.0f;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, callback = std::move(callback), target_text, store_token_lengths,
       temperature]() mutable {
        std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                     last_prefill_token_id_);
        auto decoded_ids_buffer = CopyToTensorBuffer<int>(
            decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
        if (!decoded_ids_buffer.HasValue()) {
          callback(absl::InternalError(decoded_ids_buffer.Error().Message()));
          return;
        }
        callback(ScoreCustomSampling(
            executor_, tokenizer_, target_text, temperature,
            std::move(decoded_ids_buffer.Value()), store_token_lengths));
      }));
  return nullptr;
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return GenerateContentStream(contents, std::move(callback),
                               DecodeConfig::CreateDefault());
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunPrefillAsync(
          contents,
          [this, callback = std::move(callback), decode_config = decode_config](
              absl::StatusOr<Responses> responses) mutable {
            if (!responses.ok()) {
              callback(responses.status());
            } else {
              if (cancelled_.load()) {
                callback(absl::CancelledError(
                    "Session is cancelled during prefill."));
                return;
              }
              auto status = RunDecodeAsync(std::move(callback), decode_config);
            }
          }));
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> SessionBasic::GetBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

absl::StatusOr<BenchmarkInfo*> SessionBasic::GetMutableBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return &benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

}  // namespace litert::lm
