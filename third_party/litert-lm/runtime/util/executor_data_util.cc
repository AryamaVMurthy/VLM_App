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

#include "runtime/util/executor_data_util.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

struct TokenFeatureMatrix {
  int token_count = 0;
  int feature_dim = 0;
  std::vector<float> values;
};

absl::StatusOr<int> GetTensorTokenCount(const ::litert::TensorBuffer& tensor,
                                        const std::string& tensor_name) {
  const auto& dims = TensorBufferDims(tensor);
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have at least 2 dimensions, but got rank ",
        dims.size()));
  }
  const int token_count = dims[dims.size() - 2];
  if (token_count <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have a positive token dimension, but got ",
        token_count));
  }
  return token_count;
}

int64_t Product(absl::Span<const int> dims) {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

std::string JoinIndices(absl::Span<const int> token_indices) {
  std::string joined = "[";
  for (int index = 0; index < token_indices.size(); ++index) {
    if (index > 0) {
      absl::StrAppend(&joined, ",");
    }
    absl::StrAppend(&joined, token_indices[index]);
  }
  absl::StrAppend(&joined, "]");
  return joined;
}

float L2Norm(absl::Span<const float> values) {
  float squared_norm = 0.0f;
  for (const float value : values) {
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

void NormalizeInPlace(std::vector<float>& values) {
  const float norm = L2Norm(values);
  if (norm <= 1e-6f) {
    return;
  }
  for (float& value : values) {
    value /= norm;
  }
}

float DotProduct(absl::Span<const float> lhs, absl::Span<const float> rhs) {
  float dot = 0.0f;
  for (int index = 0; index < lhs.size(); ++index) {
    dot += lhs[index] * rhs[index];
  }
  return dot;
}

std::vector<float> Softmax(absl::Span<const float> logits) {
  std::vector<float> probabilities(logits.size(), 0.0f);
  if (logits.empty()) {
    return probabilities;
  }
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (int index = 0; index < logits.size(); ++index) {
    probabilities[index] = std::exp(logits[index] - max_logit);
    sum += probabilities[index];
  }
  if (sum <= 1e-6f) {
    const float uniform_probability = 1.0f / static_cast<float>(logits.size());
    std::fill(probabilities.begin(), probabilities.end(), uniform_probability);
    return probabilities;
  }
  for (float& probability : probabilities) {
    probability /= sum;
  }
  return probabilities;
}

absl::StatusOr<std::vector<float>> BuildPromptAttentionProxyScores(
    const PromptConditioningSignals& prompt_conditioning_signals,
    absl::Span<const float> normalized_token_features, int token_count) {
  if (token_count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("token_count must be positive, but got: ", token_count));
  }
  if (prompt_conditioning_signals.feature_dim <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "prompt_conditioning_signals.feature_dim must be positive, but got: ",
        prompt_conditioning_signals.feature_dim));
  }
  if (normalized_token_features.size() !=
      static_cast<size_t>(token_count) *
          static_cast<size_t>(prompt_conditioning_signals.feature_dim)) {
    return absl::InvalidArgumentError(
        "normalized_token_features size does not match token_count * feature_dim.");
  }
  const float scale =
      1.0f / std::sqrt(static_cast<float>(prompt_conditioning_signals.feature_dim));
  std::vector<float> attention_proxy_scores(token_count, 0.0f);
  std::vector<float> logits(token_count, 0.0f);
  for (int prompt_token_index = 0;
       prompt_token_index < prompt_conditioning_signals.prompt_token_count;
       ++prompt_token_index) {
    const int prompt_token_offset =
        prompt_token_index * prompt_conditioning_signals.feature_dim;
    const auto prompt_token_features =
        absl::MakeConstSpan(prompt_conditioning_signals
                                .normalized_prompt_token_features)
            .subspan(prompt_token_offset,
                     prompt_conditioning_signals.feature_dim);
    for (int token_index = 0; token_index < token_count; ++token_index) {
      const int token_offset =
          token_index * prompt_conditioning_signals.feature_dim;
      logits[token_index] =
          DotProduct(prompt_token_features,
                     normalized_token_features.subspan(
                         token_offset,
                         prompt_conditioning_signals.feature_dim)) *
          scale;
    }
    const auto attention = Softmax(logits);
    for (int token_index = 0; token_index < token_count; ++token_index) {
      attention_proxy_scores[token_index] += attention[token_index];
    }
  }
  for (float& score : attention_proxy_scores) {
    score /= static_cast<float>(prompt_conditioning_signals.prompt_token_count);
  }
  return attention_proxy_scores;
}

std::string NormalizeStrategyString(const std::string& strategy) {
  std::string normalized;
  normalized.reserve(strategy.size());
  for (const unsigned char character : strategy) {
    if (character == '-') {
      normalized.push_back('_');
    } else {
      normalized.push_back(std::tolower(character));
    }
  }
  return normalized;
}

template <typename T>
void AccumulateTokenFeaturesFromSpan(absl::Span<const T> source, int64_t outer_groups,
                                     int token_count, int feature_dim,
                                     std::vector<float>& destination) {
  for (int64_t outer_group = 0; outer_group < outer_groups; ++outer_group) {
    for (int token_index = 0; token_index < token_count; ++token_index) {
      const int64_t source_offset =
          (outer_group * token_count + token_index) * feature_dim;
      const int64_t destination_offset = token_index * feature_dim;
      for (int feature_index = 0; feature_index < feature_dim; ++feature_index) {
        destination[destination_offset + feature_index] +=
            static_cast<float>(source[source_offset + feature_index]);
      }
    }
  }
}

absl::StatusOr<TokenFeatureMatrix> ExtractTokenFeatureMatrix(
    const ::litert::TensorBuffer& tensor, const std::string& tensor_name) {
  const auto& dims = TensorBufferDims(tensor);
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have at least 2 dimensions, but got rank ",
        dims.size()));
  }
  const int token_count = dims[dims.size() - 2];
  const int feature_dim = dims.back();
  if (token_count <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have a positive token dimension, but got ",
        token_count));
  }
  if (feature_dim <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have a positive feature dimension, but got ",
        feature_dim));
  }
  const int64_t outer_groups = Product(
      absl::MakeConstSpan(dims).subspan(0, dims.size() - 2));
  if (outer_groups <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(tensor_name, " must have a positive outer-group size."));
  }

  LITERT_ASSIGN_OR_RETURN(const auto tensor_type, tensor.TensorType());
  TokenFeatureMatrix matrix;
  matrix.token_count = token_count;
  matrix.feature_dim = feature_dim;
  matrix.values.assign(token_count * feature_dim, 0.0f);

  LITERT_ASSIGN_OR_RETURN(
      auto input_lock, ::litert::TensorBufferScopedLock::Create(
                           tensor, ::litert::TensorBuffer::LockMode::kRead));
  switch (tensor_type.ElementType()) {
    case ::litert::ElementType::Float32: {
      const auto* data = static_cast<const float*>(input_lock.second);
      AccumulateTokenFeaturesFromSpan<float>(
          absl::MakeConstSpan(data, outer_groups * token_count * feature_dim),
          outer_groups, token_count, feature_dim, matrix.values);
      break;
    }
    case ::litert::ElementType::Float16: {
      const auto* data = static_cast<const tflite::half*>(input_lock.second);
      AccumulateTokenFeaturesFromSpan<tflite::half>(
          absl::MakeConstSpan(data, outer_groups * token_count * feature_dim),
          outer_groups, token_count, feature_dim, matrix.values);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat(tensor_name,
                       " must use Float32 or Float16 element types for "
                       "prompt-conditioned pruning."));
  }

  for (float& value : matrix.values) {
    value /= static_cast<float>(outer_groups);
  }
  return matrix;
}

absl::StatusOr<TokenFeatureMatrix> ExtractVisionTokenFeatureMatrix(
    const ExecutorVisionData& vision_data) {
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    return ExtractTokenFeatureMatrix(**embeddings_status, "Vision embeddings");
  }
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    return ExtractTokenFeatureMatrix(**per_layer_status,
                                     "Vision per-layer embeddings");
  }
  return absl::NotFoundError(
      "ExecutorVisionData does not contain embeddings or per-layer "
      "embeddings.");
}

absl::Status ValidateTokenIndices(absl::Span<const int> token_indices,
                                  int total_tokens) {
  if (token_indices.empty()) {
    return absl::InvalidArgumentError(
        "token_indices must contain at least one token.");
  }
  int previous_index = -1;
  for (const int token_index : token_indices) {
    if (token_index < 0 || token_index >= total_tokens) {
      return absl::InvalidArgumentError(absl::StrCat(
          "token index ", token_index, " is out of range [0, ", total_tokens,
          ")."));
    }
    if (token_index <= previous_index) {
      return absl::InvalidArgumentError(
          "token_indices must be strictly increasing.");
    }
    previous_index = token_index;
  }
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer> SliceTensorBufferOnTokenAxis(
    const ::litert::TensorBuffer& tensor, absl::Span<const int> token_indices,
    const std::string& tensor_name) {
  ASSIGN_OR_RETURN(const int total_tokens,
                   GetTensorTokenCount(tensor, tensor_name));
  RETURN_IF_ERROR(ValidateTokenIndices(token_indices, total_tokens));

  LITERT_ASSIGN_OR_RETURN(const auto tensor_type, tensor.TensorType());
  const auto& input_dims = TensorBufferDims(tensor);
  const size_t token_axis = input_dims.size() - 2;
  const int64_t outer_groups = Product(absl::MakeConstSpan(input_dims).subspan(
      0, token_axis));
  const int64_t elements_per_token = Product(
      absl::MakeConstSpan(input_dims).subspan(token_axis + 1));
  const int64_t total_elements = outer_groups * total_tokens * elements_per_token;
  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, tensor.PackedSize());
  if (total_elements <= 0 || packed_size % total_elements != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " packed size ", packed_size,
        " is incompatible with the tensor shape."));
  }
  const size_t bytes_per_token =
      packed_size / static_cast<size_t>(outer_groups * total_tokens);

  std::vector<int> output_dims(input_dims.begin(), input_dims.end());
  output_dims[token_axis] = token_indices.size();
  Dimensions output_layout_dims;
  output_layout_dims.insert(output_layout_dims.end(), output_dims.begin(),
                            output_dims.end());
  ::litert::RankedTensorType output_tensor_type(
      tensor_type.ElementType(), Layout(std::move(output_layout_dims)));

  LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                          ::litert::TensorBuffer::CreateManagedHostMemory(
                              output_tensor_type,
                              bytes_per_token * outer_groups *
                                  token_indices.size()));
  LITERT_ASSIGN_OR_RETURN(
      auto output_lock,
      ::litert::TensorBufferScopedLock::Create(output_buffer,
                                               ::litert::TensorBuffer::LockMode::
                                                   kWrite));
  LITERT_ASSIGN_OR_RETURN(
      auto input_lock, ::litert::TensorBufferScopedLock::Create(
                           tensor, ::litert::TensorBuffer::LockMode::kRead));

  auto* output_ptr = static_cast<uint8_t*>(output_lock.second);
  auto* input_ptr = static_cast<const uint8_t*>(input_lock.second);
  const size_t input_group_stride = total_tokens * bytes_per_token;
  const size_t output_group_stride = token_indices.size() * bytes_per_token;
  for (int64_t outer_group = 0; outer_group < outer_groups; ++outer_group) {
    const uint8_t* input_group_ptr = input_ptr + outer_group * input_group_stride;
    uint8_t* output_group_ptr = output_ptr + outer_group * output_group_stride;
    for (int token_index = 0; token_index < token_indices.size();
         ++token_index) {
      memcpy(output_group_ptr + token_index * bytes_per_token,
             input_group_ptr + token_indices[token_index] * bytes_per_token,
             bytes_per_token);
    }
  }
  return output_buffer;
}

template <typename T>
absl::StatusOr<T> CombineExecutorDataImpl(std::vector<T>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    // If there is only one image, we can just move it to the combined image
    // data.
    return std::move(executor_data[0]);
  }
  // If there are multiple executor data, we need to first combine them into a
  // TensorBuffer, then create a single ExecutorVisionData from the
  // TensorBuffer.
  int num_executor_data = executor_data.size();
  ASSIGN_OR_RETURN(const auto* first_tensor,
                   executor_data[0].GetEmbeddingsPtr());
  LITERT_ASSIGN_OR_RETURN(auto first_tensor_type, first_tensor->TensorType());
  auto first_tensor_dims = TensorBufferDims(*first_tensor);
  int total_token_num = 0;
  int total_packed_size = 0;
  std::vector<int> combined_token_num;
  for (const auto& executor_data : executor_data) {
    ASSIGN_OR_RETURN(const auto* embeddings_ptr,
                     executor_data.GetEmbeddingsPtr());
    auto dims = TensorBufferDims(*embeddings_ptr);
    if (dims.size() != 3 && dims.size() != 4) {
      return absl::InvalidArgumentError(
          "The embedding tensor type must have 3 or 4 dimensions.");
    }
    combined_token_num.push_back(dims[dims.size() - 2]);
    total_token_num += dims[dims.size() - 2];
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, embeddings_ptr->PackedSize());
    total_packed_size += packed_size;
  }
  Layout combined_layout;
  if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 3) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], 1, total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 4) {
    combined_layout =
        Layout(Dimensions({first_tensor_dims[0], first_tensor_dims[1],
                           total_token_num, first_tensor_dims[3]}));
  }
  ::litert::RankedTensorType combined_tensor_type(
      first_tensor_type.ElementType(), std::move(combined_layout));

  LITERT_ASSIGN_OR_RETURN(auto combined_tensor_buffer,
                          TensorBuffer::CreateManagedHostMemory(
                              combined_tensor_type, total_packed_size));
  LITERT_ASSIGN_OR_RETURN(
      auto combined_embeddings_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(combined_tensor_buffer,
                                               TensorBuffer::LockMode::kWrite));
  char* combined_tensor_buffer_ptr =
      static_cast<char*>(combined_embeddings_lock_and_addr.second);
  for (int i = 0; i < num_executor_data; ++i) {
    ASSIGN_OR_RETURN(auto embeddings_ptr,
                     executor_data[i].GetMutableEmbeddingsPtr());
    LITERT_ASSIGN_OR_RETURN(auto embeddings_size, embeddings_ptr->PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto embeddings_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            *embeddings_ptr, TensorBuffer::LockMode::kRead));
    memcpy(combined_tensor_buffer_ptr, embeddings_lock_and_addr.second,
           embeddings_size);
    combined_tensor_buffer_ptr += embeddings_size;
  }
  if constexpr (std::is_same_v<T, ExecutorVisionData>) {
    return ExecutorVisionData(std::move(combined_tensor_buffer),
                              /*per_layer_embeddings=*/std::nullopt);
  } else if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    int num_audio_tokens = 0;
    for (const auto& executor_data : executor_data) {
      num_audio_tokens += executor_data.GetValidTokens();
    }
    return ExecutorAudioData(std::move(combined_tensor_buffer),
                             /*per_layer_embeddings=*/std::nullopt,
                             num_audio_tokens);
  } else {
    return absl::InvalidArgumentError("Executor data type is not supported.");
  }
}

}  // namespace

const char* VisionTokenPruningStrategyToString(
    VisionTokenPruningStrategy strategy) {
  switch (strategy) {
    case VisionTokenPruningStrategy::kUniform:
      return "uniform";
    case VisionTokenPruningStrategy::kPromptConditionedV1:
      return "prompt_conditioned_v1";
  }
  return "unknown";
}

absl::StatusOr<VisionTokenPruningStrategy> ParseVisionTokenPruningStrategy(
    const std::string& strategy) {
  const std::string normalized = NormalizeStrategyString(strategy);
  if (normalized == "uniform") {
    return VisionTokenPruningStrategy::kUniform;
  }
  if (normalized == "prompt_conditioned_v1") {
    return VisionTokenPruningStrategy::kPromptConditionedV1;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported visual token pruning strategy: ", strategy,
      ". Expected one of: uniform, prompt_conditioned_v1."));
}

absl::StatusOr<int> GetExecutorVisionTokenCount(
    const ExecutorVisionData& vision_data) {
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    return GetTensorTokenCount(**embeddings_status, "Vision embeddings");
  }
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    return GetTensorTokenCount(**per_layer_status,
                               "Vision per-layer embeddings");
  }
  return absl::NotFoundError(
      "ExecutorVisionData does not contain embeddings or per-layer "
      "embeddings.");
}

absl::StatusOr<std::vector<int>> BuildUniformTokenSelection(int total_tokens,
                                                            int target_tokens) {
  if (total_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_tokens must be positive, but got: ",
                     total_tokens));
  }
  if (target_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("target_tokens must be positive, but got: ",
                     target_tokens));
  }
  if (target_tokens >= total_tokens) {
    std::vector<int> token_indices(total_tokens);
    std::iota(token_indices.begin(), token_indices.end(), 0);
    return token_indices;
  }
  if (target_tokens == 1) {
    return std::vector<int>{0};
  }
  std::vector<int> token_indices;
  token_indices.reserve(target_tokens);
  const double step = static_cast<double>(total_tokens - 1) /
                      static_cast<double>(target_tokens - 1);
  int previous_index = -1;
  for (int token_slot = 0; token_slot < target_tokens; ++token_slot) {
    int token_index = static_cast<int>(std::lround(token_slot * step));
    token_index = std::clamp(token_index, 0, total_tokens - 1);
    if (token_index <= previous_index) {
      token_index = previous_index + 1;
    }
    token_indices.push_back(token_index);
    previous_index = token_index;
  }
  token_indices.back() = total_tokens - 1;
  return token_indices;
}

absl::StatusOr<ExecutorVisionData> SelectExecutorVisionTokens(
    const ExecutorVisionData& vision_data, absl::Span<const int> token_indices) {
  ASSIGN_OR_RETURN(const int total_tokens, GetExecutorVisionTokenCount(vision_data));
  RETURN_IF_ERROR(ValidateTokenIndices(token_indices, total_tokens));

  std::optional<::litert::TensorBuffer> pruned_embeddings = std::nullopt;
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    ASSIGN_OR_RETURN(pruned_embeddings,
                     SliceTensorBufferOnTokenAxis(**embeddings_status,
                                                 token_indices,
                                                 "Vision embeddings"));
  }

  std::optional<::litert::TensorBuffer> pruned_per_layer_embeddings =
      std::nullopt;
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    ASSIGN_OR_RETURN(const int per_layer_token_count,
                     GetTensorTokenCount(**per_layer_status,
                                         "Vision per-layer embeddings"));
    if (per_layer_token_count != total_tokens) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Vision embeddings token count ", total_tokens,
          " does not match per-layer embeddings token count ",
          per_layer_token_count));
    }
    ASSIGN_OR_RETURN(pruned_per_layer_embeddings,
                     SliceTensorBufferOnTokenAxis(**per_layer_status,
                                                 token_indices,
                                                 "Vision per-layer embeddings"));
  }

  return ExecutorVisionData(std::move(pruned_embeddings),
                            std::move(pruned_per_layer_embeddings));
}

absl::StatusOr<ExecutorVisionData> PruneExecutorVisionData(
    const ExecutorVisionData& vision_data, int target_tokens) {
  ASSIGN_OR_RETURN(const int total_tokens,
                   GetExecutorVisionTokenCount(vision_data));
  ASSIGN_OR_RETURN(auto token_indices,
                   BuildUniformTokenSelection(total_tokens, target_tokens));
  return SelectExecutorVisionTokens(vision_data, token_indices);
}

absl::StatusOr<std::vector<float>> BuildPromptConditioningVector(
    const CachedTextEmbeddings& cached_text_embeddings, int feature_dim) {
  if (feature_dim <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "feature_dim must be positive, but got: ", feature_dim));
  }
  RETURN_IF_ERROR(cached_text_embeddings.Validate());
  if (cached_text_embeddings.token_count == 0) {
    return absl::InvalidArgumentError(
        "cached_text_embeddings must contain at least one text token.");
  }
  if (cached_text_embeddings.floats_per_token != feature_dim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cached_text_embeddings.floats_per_token ",
        cached_text_embeddings.floats_per_token,
        " does not match feature_dim ", feature_dim, "."));
  }

  std::vector<float> prompt_features(feature_dim, 0.0f);
  for (int token_index = 0; token_index < cached_text_embeddings.token_count;
       ++token_index) {
    ASSIGN_OR_RETURN(
        auto token_embedding,
        cached_text_embeddings.GetTokenEmbedding(token_index));
    for (int feature_index = 0; feature_index < feature_dim; ++feature_index) {
      prompt_features[feature_index] += token_embedding[feature_index];
    }
  }
  for (float& value : prompt_features) {
    value /= static_cast<float>(cached_text_embeddings.token_count);
  }
  NormalizeInPlace(prompt_features);
  if (L2Norm(prompt_features) <= 1e-6f) {
    return absl::InvalidArgumentError(
        "cached_text_embeddings produced a degenerate prompt conditioning "
        "vector.");
  }
  return prompt_features;
}

absl::StatusOr<PromptConditioningSignals> BuildPromptConditioningSignals(
    const CachedTextEmbeddings& cached_text_embeddings, int feature_dim) {
  RETURN_IF_ERROR(cached_text_embeddings.Validate());
  ASSIGN_OR_RETURN(auto pooled_prompt_features,
                   BuildPromptConditioningVector(cached_text_embeddings,
                                                feature_dim));
  PromptConditioningSignals signals;
  signals.pooled_prompt_features = std::move(pooled_prompt_features);
  signals.prompt_token_count = cached_text_embeddings.token_count;
  signals.feature_dim = feature_dim;
  signals.normalized_prompt_token_features.reserve(
      static_cast<size_t>(signals.prompt_token_count) *
      static_cast<size_t>(feature_dim));
  for (int token_index = 0; token_index < cached_text_embeddings.token_count;
       ++token_index) {
    ASSIGN_OR_RETURN(
        auto token_embedding,
        cached_text_embeddings.GetTokenEmbedding(token_index));
    std::vector<float> normalized_token_embedding(token_embedding.begin(),
                                                  token_embedding.end());
    NormalizeInPlace(normalized_token_embedding);
    if (L2Norm(normalized_token_embedding) <= 1e-6f) {
      return absl::InvalidArgumentError(absl::StrCat(
          "cached_text_embeddings token ", token_index,
          " produced a degenerate prompt token embedding."));
    }
    signals.normalized_prompt_token_features.insert(
        signals.normalized_prompt_token_features.end(),
        normalized_token_embedding.begin(), normalized_token_embedding.end());
  }
  return signals;
}

std::string VisionTokenPruningDecision::ToLogString() const {
  return absl::StrCat(
      "strategy=", VisionTokenPruningStrategyToString(strategy),
      " kept=", token_indices.size(), " original=", original_token_count,
      " mean_prompt_similarity=", mean_prompt_similarity,
      " mean_salience=", mean_salience, " selected_token_indices=",
      JoinIndices(logged_token_indices));
}

absl::StatusOr<VisionTokenPruningDecision> BuildVisionTokenPruningDecision(
    const ExecutorVisionData& vision_data, int target_tokens,
    const PromptConditioningSignals* prompt_conditioning_signals,
    const VisionTokenPruningConfig& config) {
  ASSIGN_OR_RETURN(const int total_tokens,
                   GetExecutorVisionTokenCount(vision_data));
  if (target_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("target_tokens must be positive, but got: ",
                     target_tokens));
  }

  VisionTokenPruningDecision decision;
  decision.strategy = config.strategy;
  decision.original_token_count = total_tokens;

  if (config.max_logged_token_indices <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_logged_token_indices must be positive, but got: ",
        config.max_logged_token_indices));
  }

  if (config.strategy == VisionTokenPruningStrategy::kUniform) {
    ASSIGN_OR_RETURN(decision.token_indices,
                     BuildUniformTokenSelection(total_tokens, target_tokens));
  } else if (config.strategy ==
             VisionTokenPruningStrategy::kPromptConditionedV1) {
    if (target_tokens >= total_tokens) {
      ASSIGN_OR_RETURN(decision.token_indices,
                       BuildUniformTokenSelection(total_tokens, total_tokens));
    } else {
      ASSIGN_OR_RETURN(TokenFeatureMatrix feature_matrix,
                       ExtractVisionTokenFeatureMatrix(vision_data));
      if (prompt_conditioning_signals == nullptr) {
        return absl::InvalidArgumentError(
            "prompt_conditioning_signals must be provided when using "
            "prompt_conditioned_v1 pruning.");
      }
      if (prompt_conditioning_signals->feature_dim != feature_matrix.feature_dim) {
        return absl::InvalidArgumentError(absl::StrCat(
            "prompt_conditioning_signals feature_dim ",
            prompt_conditioning_signals->feature_dim,
            " does not match the vision feature dimension ",
            feature_matrix.feature_dim, "."));
      }
      if (prompt_conditioning_signals->prompt_token_count <= 0) {
        return absl::InvalidArgumentError(
            "prompt_conditioning_signals must contain at least one prompt "
            "token embedding.");
      }
      if (prompt_conditioning_signals->normalized_prompt_token_features.size() !=
          static_cast<size_t>(prompt_conditioning_signals->prompt_token_count) *
              static_cast<size_t>(feature_matrix.feature_dim)) {
        return absl::InvalidArgumentError(
            "prompt_conditioning_signals normalized_prompt_token_features "
            "size does not match prompt_token_count * feature_dim.");
      }

      std::vector<float> mean_feature(feature_matrix.feature_dim, 0.0f);
      for (int token_index = 0; token_index < feature_matrix.token_count;
           ++token_index) {
        const int token_offset = token_index * feature_matrix.feature_dim;
        for (int feature_index = 0; feature_index < feature_matrix.feature_dim;
             ++feature_index) {
          mean_feature[feature_index] +=
              feature_matrix.values[token_offset + feature_index];
        }
      }
      for (float& value : mean_feature) {
        value /= static_cast<float>(feature_matrix.token_count);
      }

      std::vector<float> prompt_similarities(feature_matrix.token_count, 0.0f);
      std::vector<float> saliences(feature_matrix.token_count, 0.0f);
      std::vector<float> normalized_token_features(feature_matrix.values.size(),
                                                   0.0f);
      float max_centered_norm = 0.0f;
      for (int token_index = 0; token_index < feature_matrix.token_count;
           ++token_index) {
        const int token_offset = token_index * feature_matrix.feature_dim;
        std::vector<float> centered_feature(feature_matrix.feature_dim, 0.0f);
        std::vector<float> token_feature(feature_matrix.feature_dim, 0.0f);
        for (int feature_index = 0; feature_index < feature_matrix.feature_dim;
             ++feature_index) {
          const float value =
              feature_matrix.values[token_offset + feature_index];
          token_feature[feature_index] = value;
          centered_feature[feature_index] = value - mean_feature[feature_index];
        }
        const float token_norm = L2Norm(token_feature);
        if (token_norm > 1e-6f) {
          for (int feature_index = 0; feature_index < feature_matrix.feature_dim;
               ++feature_index) {
            normalized_token_features[token_offset + feature_index] =
                token_feature[feature_index] / token_norm;
          }
        }
        saliences[token_index] = L2Norm(centered_feature);
        max_centered_norm = std::max(max_centered_norm, saliences[token_index]);
      }
      ASSIGN_OR_RETURN(
          prompt_similarities,
          BuildPromptAttentionProxyScores(
              *prompt_conditioning_signals,
              absl::MakeConstSpan(normalized_token_features),
              feature_matrix.token_count));
      if (max_centered_norm > 1e-6f) {
        for (float& salience : saliences) {
          salience /= max_centered_norm;
        }
      }

      std::vector<float> base_scores(feature_matrix.token_count, 0.0f);
      for (int token_index = 0; token_index < feature_matrix.token_count;
           ++token_index) {
        base_scores[token_index] =
            config.prompt_similarity_weight * prompt_similarities[token_index] +
            config.salience_weight * saliences[token_index];
      }

      const float redundancy_weight =
          std::clamp(config.redundancy_penalty_weight, 0.0f, 1.0f);
      std::vector<int> selected_order;
      selected_order.reserve(target_tokens);
      std::vector<bool> already_selected(feature_matrix.token_count, false);
      while (selected_order.size() < target_tokens) {
        int best_index = -1;
        float best_adjusted_score = -std::numeric_limits<float>::infinity();
        float best_redundancy = std::numeric_limits<float>::infinity();
        float best_base_score = -std::numeric_limits<float>::infinity();
        for (int token_index = 0; token_index < feature_matrix.token_count;
             ++token_index) {
          if (already_selected[token_index]) {
            continue;
          }
          const int token_offset = token_index * feature_matrix.feature_dim;
          float max_redundancy = 0.0f;
          for (const int selected_index : selected_order) {
            const int selected_offset =
                selected_index * feature_matrix.feature_dim;
            max_redundancy = std::max(
                max_redundancy,
                DotProduct(
                    absl::MakeConstSpan(normalized_token_features)
                        .subspan(token_offset, feature_matrix.feature_dim),
                    absl::MakeConstSpan(normalized_token_features)
                        .subspan(selected_offset, feature_matrix.feature_dim)));
          }
          const float redundancy_scale =
              1.0f -
              redundancy_weight * max_redundancy /
                  std::max(1.0e-3f, 1.0f - max_redundancy);
          const float adjusted_score =
              base_scores[token_index] * redundancy_scale;
          const bool is_better =
              adjusted_score > best_adjusted_score + 1e-6f ||
              (std::abs(adjusted_score - best_adjusted_score) <= 1e-6f &&
               (max_redundancy < best_redundancy - 1e-6f ||
                (std::abs(max_redundancy - best_redundancy) <= 1e-6f &&
                 (base_scores[token_index] > best_base_score + 1e-6f ||
                  (std::abs(base_scores[token_index] - best_base_score) <=
                       1e-6f &&
                   token_index < best_index)))));
          if (is_better) {
            best_index = token_index;
            best_adjusted_score = adjusted_score;
            best_redundancy = max_redundancy;
            best_base_score = base_scores[token_index];
          }
        }
        if (best_index < 0) {
          return absl::InternalError(
              "Failed to select a vision token during prompt-conditioned "
              "pruning.");
        }
        already_selected[best_index] = true;
        selected_order.push_back(best_index);
        decision.mean_prompt_similarity += prompt_similarities[best_index];
        decision.mean_salience += saliences[best_index];
      }
      decision.token_indices = selected_order;
      std::sort(decision.token_indices.begin(), decision.token_indices.end());
      decision.mean_prompt_similarity /=
          static_cast<float>(decision.token_indices.size());
      decision.mean_salience /=
          static_cast<float>(decision.token_indices.size());
    }
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported visual token pruning strategy enum: ",
        static_cast<int>(config.strategy)));
  }

  const int num_logged_indices =
      std::min<int>(config.max_logged_token_indices, decision.token_indices.size());
  decision.logged_token_indices.assign(decision.token_indices.begin(),
                                       decision.token_indices.begin() +
                                           num_logged_indices);
  return decision;
}

absl::StatusOr<PrunedExecutorVisionData> PruneExecutorVisionData(
    const ExecutorVisionData& vision_data, int target_tokens,
    const CachedTextEmbeddings* cached_text_embeddings,
    const VisionTokenPruningConfig& config) {
  ASSIGN_OR_RETURN(const int total_tokens,
                   GetExecutorVisionTokenCount(vision_data));
  if (target_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("target_tokens must be positive, but got: ",
                     target_tokens));
  }

  std::optional<PromptConditioningSignals> prompt_conditioning_signals =
      std::nullopt;
  if (config.strategy == VisionTokenPruningStrategy::kPromptConditionedV1 &&
      target_tokens < total_tokens) {
    if (cached_text_embeddings == nullptr) {
      return absl::InvalidArgumentError(
          "cached_text_embeddings must be provided when using "
          "prompt_conditioned_v1 pruning.");
    }
    ASSIGN_OR_RETURN(TokenFeatureMatrix feature_matrix,
                     ExtractVisionTokenFeatureMatrix(vision_data));
    ASSIGN_OR_RETURN(
        prompt_conditioning_signals,
        BuildPromptConditioningSignals(*cached_text_embeddings,
                                       feature_matrix.feature_dim));
  }
  ASSIGN_OR_RETURN(auto decision,
                   BuildVisionTokenPruningDecision(
                       vision_data, target_tokens,
                       prompt_conditioning_signals.has_value()
                           ? &prompt_conditioning_signals.value()
                           : nullptr,
                       config));
  ASSIGN_OR_RETURN(auto pruned_vision_data,
                   SelectExecutorVisionTokens(vision_data,
                                              decision.token_indices));
  return PrunedExecutorVisionData{std::move(pruned_vision_data),
                                  std::move(decision)};
}

absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

}  // namespace litert::lm
