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

struct PromptAttentionProxy {
  int prompt_token_count = 0;
  int token_count = 0;
  std::vector<float> per_prompt_token_scores;
  std::vector<float> averaged_scores;
};

struct UniformTokenBucket {
  int start_index = 0;
  int end_index = 0;
  int anchor_index = 0;
};

absl::StatusOr<::litert::TensorBuffer> SliceTensorBufferOnTokenAxis(
    const ::litert::TensorBuffer& tensor, absl::Span<const int> token_indices,
    const std::string& tensor_name);

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

bool IsPromptConditionedStrategy(VisionTokenPruningStrategy strategy) {
  return strategy == VisionTokenPruningStrategy::kPromptConditionedV1 ||
         strategy == VisionTokenPruningStrategy::kPromptConditionedV2;
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

absl::StatusOr<PromptAttentionProxy> BuildPromptAttentionProxyScores(
    const PromptConditioningSignals& prompt_conditioning_signals,
    absl::Span<const float> normalized_token_features, int token_count,
    float logit_scale) {
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
  if (!(logit_scale > 0.0f)) {
    return absl::InvalidArgumentError(
        absl::StrCat("logit_scale must be positive, but got: ", logit_scale));
  }
  PromptAttentionProxy attention_proxy;
  attention_proxy.prompt_token_count =
      prompt_conditioning_signals.prompt_token_count;
  attention_proxy.token_count = token_count;
  attention_proxy.per_prompt_token_scores.reserve(
      static_cast<size_t>(prompt_conditioning_signals.prompt_token_count) *
      static_cast<size_t>(token_count));
  attention_proxy.averaged_scores.assign(token_count, 0.0f);
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
          logit_scale;
    }
    const auto attention = Softmax(logits);
    attention_proxy.per_prompt_token_scores.insert(
        attention_proxy.per_prompt_token_scores.end(), attention.begin(),
        attention.end());
    for (int token_index = 0; token_index < token_count; ++token_index) {
      attention_proxy.averaged_scores[token_index] += attention[token_index];
    }
  }
  for (float& score : attention_proxy.averaged_scores) {
    score /= static_cast<float>(prompt_conditioning_signals.prompt_token_count);
  }
  return attention_proxy;
}

void NormalizeByMax(std::vector<float>& values) {
  if (values.empty()) {
    return;
  }
  const float max_value = *std::max_element(values.begin(), values.end());
  if (max_value <= 1e-6f) {
    return;
  }
  for (float& value : values) {
    value /= max_value;
  }
}

absl::StatusOr<std::vector<float>> BuildPromptConditionedV2Scores(
    const PromptAttentionProxy& attention_proxy, int prompt_attention_top_k) {
  if (attention_proxy.prompt_token_count <= 0) {
    return absl::InvalidArgumentError(
        "attention_proxy.prompt_token_count must be positive.");
  }
  if (attention_proxy.token_count <= 0) {
    return absl::InvalidArgumentError(
        "attention_proxy.token_count must be positive.");
  }
  if (prompt_attention_top_k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("prompt_attention_top_k must be positive, but got: ",
                     prompt_attention_top_k));
  }
  if (attention_proxy.per_prompt_token_scores.size() !=
      static_cast<size_t>(attention_proxy.prompt_token_count) *
          static_cast<size_t>(attention_proxy.token_count)) {
    return absl::InvalidArgumentError(
        "attention_proxy.per_prompt_token_scores size does not match "
        "prompt_token_count * token_count.");
  }

  std::vector<float> per_prompt_token_max(
      attention_proxy.prompt_token_count, 0.0f);
  for (int prompt_token_index = 0;
       prompt_token_index < attention_proxy.prompt_token_count;
       ++prompt_token_index) {
    float max_value = 0.0f;
    for (int token_index = 0; token_index < attention_proxy.token_count;
         ++token_index) {
      max_value = std::max(
          max_value,
          attention_proxy.per_prompt_token_scores
              [prompt_token_index * attention_proxy.token_count + token_index]);
    }
    per_prompt_token_max[prompt_token_index] = max_value;
  }

  const int top_k =
      std::min(prompt_attention_top_k, attention_proxy.prompt_token_count);
  std::vector<float> token_scores(attention_proxy.token_count, 0.0f);
  std::vector<float> scratch;
  scratch.reserve(attention_proxy.prompt_token_count);
  for (int token_index = 0; token_index < attention_proxy.token_count;
       ++token_index) {
    scratch.clear();
    for (int prompt_token_index = 0;
         prompt_token_index < attention_proxy.prompt_token_count;
         ++prompt_token_index) {
      const float raw_score =
          attention_proxy.per_prompt_token_scores
              [prompt_token_index * attention_proxy.token_count + token_index];
      const float denom = per_prompt_token_max[prompt_token_index];
      scratch.push_back(denom > 1e-6f ? raw_score / denom : 0.0f);
    }
    if (top_k < scratch.size()) {
      std::nth_element(scratch.begin(), scratch.begin() + top_k, scratch.end(),
                       std::greater<float>());
    } else {
      std::sort(scratch.begin(), scratch.end(), std::greater<float>());
    }
    float score = 0.0f;
    for (int i = 0; i < top_k; ++i) {
      score += scratch[i];
    }
    token_scores[token_index] = score / static_cast<float>(top_k);
  }
  NormalizeByMax(token_scores);
  return token_scores;
}

absl::StatusOr<std::vector<UniformTokenBucket>> BuildUniformTokenBuckets(
    int total_tokens, absl::Span<const int> uniform_indices) {
  if (total_tokens <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_tokens must be positive, but got: ", total_tokens));
  }
  if (uniform_indices.empty()) {
    return absl::InvalidArgumentError("uniform_indices must not be empty.");
  }
  int previous_index = -1;
  for (const int token_index : uniform_indices) {
    if (token_index < 0 || token_index >= total_tokens) {
      return absl::InvalidArgumentError(absl::StrCat(
          "uniform token index ", token_index, " is outside [0, ",
          total_tokens, ")."));
    }
    if (token_index <= previous_index) {
      return absl::InvalidArgumentError(
          "uniform_indices must be strictly increasing.");
    }
    previous_index = token_index;
  }

  std::vector<UniformTokenBucket> buckets;
  buckets.reserve(uniform_indices.size());
  int start_index = 0;
  for (int i = 0; i < uniform_indices.size(); ++i) {
    const int anchor_index = uniform_indices[i];
    const int end_index =
        (i + 1 < uniform_indices.size())
            ? ((anchor_index + uniform_indices[i + 1]) / 2 + 1)
            : total_tokens;
    if (end_index <= start_index || anchor_index < start_index ||
        anchor_index >= end_index) {
      return absl::InternalError("Failed to build valid uniform token buckets.");
    }
    buckets.push_back(UniformTokenBucket{
        .start_index = start_index,
        .end_index = end_index,
        .anchor_index = anchor_index,
    });
    start_index = end_index;
  }
  if (start_index != total_tokens) {
    return absl::InternalError("Uniform token buckets do not cover all tokens.");
  }
  return buckets;
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

absl::Status ValidateSelectedTokenMetadata(
    const ExecutorVisionData& vision_data) {
  const auto& selected_token_indices = vision_data.GetSelectedTokenIndices();
  if (!selected_token_indices.has_value()) {
    return absl::OkStatus();
  }
  if (selected_token_indices->empty()) {
    return absl::InvalidArgumentError(
        "ExecutorVisionData selected_token_indices must not be empty.");
  }
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    ASSIGN_OR_RETURN(const int total_tokens,
                     GetTensorTokenCount(**embeddings_status,
                                         "Vision embeddings"));
    return ValidateTokenIndices(*selected_token_indices, total_tokens);
  }
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    ASSIGN_OR_RETURN(const int total_tokens,
                     GetTensorTokenCount(**per_layer_status,
                                         "Vision per-layer embeddings"));
    return ValidateTokenIndices(*selected_token_indices, total_tokens);
  }
  return absl::NotFoundError(
      "ExecutorVisionData selected_token_indices require embeddings or "
      "per-layer embeddings.");
}

absl::Status ValidateSparseSelectionPackingSupport(
    const ::litert::TensorBuffer& tensor, const std::string& tensor_name) {
  const auto& dims = TensorBufferDims(tensor);
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        tensor_name, " must have at least 2 dimensions, but got rank ",
        dims.size()));
  }
  const size_t token_axis = dims.size() - 2;
  const int64_t outer_groups = Product(absl::MakeConstSpan(dims).subspan(
      0, token_axis));
  if (outer_groups != 1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Fused projection-prune-pack does not support ", tensor_name,
        " layouts with outer_groups=", outer_groups,
        ". Expected outer_groups=1 so retained tokens stay contiguous in the "
        "source buffer."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> ResolveUnderlyingTokenIndices(
    const ExecutorVisionData& vision_data, absl::Span<const int> token_indices) {
  ASSIGN_OR_RETURN(const int visible_tokens, GetExecutorVisionTokenCount(vision_data));
  RETURN_IF_ERROR(ValidateTokenIndices(token_indices, visible_tokens));
  if (!vision_data.GetSelectedTokenIndices().has_value()) {
    return std::vector<int>(token_indices.begin(), token_indices.end());
  }
  std::vector<int> resolved_token_indices;
  resolved_token_indices.reserve(token_indices.size());
  for (const int token_index : token_indices) {
    resolved_token_indices.push_back(
        vision_data.GetSelectedTokenIndices()->at(token_index));
  }
  return resolved_token_indices;
}

absl::StatusOr<ExecutorVisionData> MaterializeSelectedExecutorVisionData(
    const ExecutorVisionData& vision_data) {
  if (!vision_data.GetSelectedTokenIndices().has_value()) {
    return absl::InvalidArgumentError(
        "MaterializeSelectedExecutorVisionData requires selected_token_indices.");
  }
  const auto& selected_token_indices = *vision_data.GetSelectedTokenIndices();
  std::optional<::litert::TensorBuffer> pruned_embeddings = std::nullopt;
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    ASSIGN_OR_RETURN(pruned_embeddings,
                     SliceTensorBufferOnTokenAxis(**embeddings_status,
                                                 selected_token_indices,
                                                 "Vision embeddings"));
  }
  std::optional<::litert::TensorBuffer> pruned_per_layer_embeddings =
      std::nullopt;
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    ASSIGN_OR_RETURN(pruned_per_layer_embeddings,
                     SliceTensorBufferOnTokenAxis(**per_layer_status,
                                                 selected_token_indices,
                                                 "Vision per-layer embeddings"));
  }
  return ExecutorVisionData(std::move(pruned_embeddings),
                            std::move(pruned_per_layer_embeddings));
}

absl::StatusOr<ExecutorVisionData> CreateSparseSelectedExecutorVisionData(
    const ExecutorVisionData& vision_data, absl::Span<const int> token_indices) {
  if (vision_data.GetPerLayerEmbeddingsPtr().ok()) {
    return absl::FailedPreconditionError(
        "Fused projection-prune-pack does not support vision per-layer "
        "embeddings.");
  }
  ASSIGN_OR_RETURN(auto resolved_token_indices,
                   ResolveUnderlyingTokenIndices(vision_data, token_indices));
  ASSIGN_OR_RETURN(const auto* embeddings_ptr, vision_data.GetEmbeddingsPtr());
  RETURN_IF_ERROR(ValidateSparseSelectionPackingSupport(*embeddings_ptr,
                                                        "Vision embeddings"));
  ASSIGN_OR_RETURN(const int total_tokens,
                   GetTensorTokenCount(*embeddings_ptr, "Vision embeddings"));
  RETURN_IF_ERROR(ValidateTokenIndices(resolved_token_indices, total_tokens));
  LITERT_ASSIGN_OR_RETURN(auto duplicated_embeddings, embeddings_ptr->Duplicate());
  ExecutorVisionData sparse_vision_data(std::move(duplicated_embeddings),
                                        /*per_layer_embeddings=*/std::nullopt);
  sparse_vision_data.SetSelectedTokenIndices(std::move(resolved_token_indices));
  return sparse_vision_data;
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
  if constexpr (std::is_same_v<T, ExecutorVisionData>) {
    for (auto& single_executor_data : executor_data) {
      if (!single_executor_data.GetSelectedTokenIndices().has_value()) {
        continue;
      }
      ASSIGN_OR_RETURN(single_executor_data,
                       MaterializeSelectedExecutorVisionData(
                           single_executor_data));
    }
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
    case VisionTokenPruningStrategy::kPromptConditionedV2:
      return "prompt_conditioned_v2";
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
  if (normalized == "prompt_conditioned_v2") {
    return VisionTokenPruningStrategy::kPromptConditionedV2;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported visual token pruning strategy: ", strategy,
      ". Expected one of: uniform, prompt_conditioned_v1, "
      "prompt_conditioned_v2."));
}

absl::StatusOr<int> GetExecutorVisionTokenCount(
    const ExecutorVisionData& vision_data) {
  RETURN_IF_ERROR(ValidateSelectedTokenMetadata(vision_data));
  if (vision_data.GetSelectedTokenIndices().has_value()) {
    return static_cast<int>(vision_data.GetSelectedTokenIndices()->size());
  }
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
  ASSIGN_OR_RETURN(auto resolved_token_indices,
                   ResolveUnderlyingTokenIndices(vision_data, token_indices));

  std::optional<::litert::TensorBuffer> pruned_embeddings = std::nullopt;
  std::optional<int> embedding_total_tokens = std::nullopt;
  if (auto embeddings_status = vision_data.GetEmbeddingsPtr();
      embeddings_status.ok()) {
    ASSIGN_OR_RETURN(embedding_total_tokens,
                     GetTensorTokenCount(**embeddings_status,
                                         "Vision embeddings"));
    RETURN_IF_ERROR(
        ValidateTokenIndices(resolved_token_indices, *embedding_total_tokens));
    ASSIGN_OR_RETURN(pruned_embeddings,
                     SliceTensorBufferOnTokenAxis(**embeddings_status,
                                                 resolved_token_indices,
                                                 "Vision embeddings"));
  }

  std::optional<::litert::TensorBuffer> pruned_per_layer_embeddings =
      std::nullopt;
  if (auto per_layer_status = vision_data.GetPerLayerEmbeddingsPtr();
      per_layer_status.ok()) {
    ASSIGN_OR_RETURN(const int per_layer_token_count,
                     GetTensorTokenCount(**per_layer_status,
                                         "Vision per-layer embeddings"));
    if (embedding_total_tokens.has_value() &&
        per_layer_token_count != *embedding_total_tokens) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Vision embeddings token count ", *embedding_total_tokens,
          " does not match per-layer embeddings token count ",
          per_layer_token_count));
    }
    RETURN_IF_ERROR(
        ValidateTokenIndices(resolved_token_indices, per_layer_token_count));
    ASSIGN_OR_RETURN(pruned_per_layer_embeddings,
                     SliceTensorBufferOnTokenAxis(**per_layer_status,
                                                 resolved_token_indices,
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
      " mean_salience=", mean_salience,
      " mean_reference_prompt_similarity=", mean_reference_prompt_similarity,
      " mean_reference_salience=", mean_reference_salience,
      " local_refinement_count=", local_refinement_count,
      " proposed_local_refinement_count=", proposed_local_refinement_count,
      " local_refinement_candidate_count=", local_refinement_candidate_count,
      " mean_local_refinement_prompt_gain=",
      mean_local_refinement_prompt_gain,
      " max_local_refinement_prompt_gain=",
      max_local_refinement_prompt_gain,
      " controller_kept_uniform=", controller_kept_uniform ? "true" : "false",
      " controller_reason=", controller_reason.empty() ? "none"
                                                       : controller_reason,
      " selected_token_indices=",
      JoinIndices(logged_token_indices), " reference_token_indices=",
      JoinIndices(logged_reference_token_indices));
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
    decision.reference_token_indices = decision.token_indices;
  } else if (IsPromptConditionedStrategy(config.strategy)) {
    if (target_tokens >= total_tokens) {
      ASSIGN_OR_RETURN(decision.token_indices,
                       BuildUniformTokenSelection(total_tokens, total_tokens));
      decision.reference_token_indices = decision.token_indices;
    } else {
      ASSIGN_OR_RETURN(TokenFeatureMatrix feature_matrix,
                       ExtractVisionTokenFeatureMatrix(vision_data));
      if (prompt_conditioning_signals == nullptr) {
        return absl::InvalidArgumentError(
            "prompt_conditioning_signals must be provided when using "
            "prompt-conditioned visual pruning.");
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
          auto attention_proxy,
          BuildPromptAttentionProxyScores(
              *prompt_conditioning_signals,
              absl::MakeConstSpan(normalized_token_features),
              feature_matrix.token_count,
              config.strategy == VisionTokenPruningStrategy::kPromptConditionedV1
                  ? 1.0f / std::sqrt(static_cast<float>(
                                prompt_conditioning_signals->feature_dim))
                  : config.prompt_attention_logit_scale));
      prompt_similarities = std::move(attention_proxy.averaged_scores);
      if (config.strategy == VisionTokenPruningStrategy::kPromptConditionedV2) {
        ASSIGN_OR_RETURN(prompt_similarities,
                         BuildPromptConditionedV2Scores(
                             attention_proxy, config.prompt_attention_top_k));
      }
      if (max_centered_norm > 1e-6f) {
        for (float& salience : saliences) {
          salience /= max_centered_norm;
        }
      }

      if (config.strategy == VisionTokenPruningStrategy::kPromptConditionedV2) {
        struct BucketRefinementCandidate {
          int bucket_index = 0;
          int selected_index = 0;
          float prompt_gain = 0.0f;
          float selected_prompt_score = 0.0f;
          float selected_salience = 0.0f;
        };
        ASSIGN_OR_RETURN(auto uniform_indices,
                         BuildUniformTokenSelection(feature_matrix.token_count,
                                                    target_tokens));
        decision.reference_token_indices = uniform_indices;
        ASSIGN_OR_RETURN(auto uniform_buckets,
                         BuildUniformTokenBuckets(feature_matrix.token_count,
                                                 uniform_indices));
        std::vector<int> selected_indices = uniform_indices;
        std::vector<float> selected_prompt_scores;
        selected_prompt_scores.reserve(target_tokens);
        std::vector<float> selected_saliences;
        selected_saliences.reserve(target_tokens);
        std::vector<BucketRefinementCandidate> refinement_candidates;
        refinement_candidates.reserve(target_tokens);
        for (int bucket_index = 0; bucket_index < uniform_buckets.size();
             ++bucket_index) {
          const auto& bucket = uniform_buckets[bucket_index];
          const int anchor_index = bucket.anchor_index;
          int selected_index = anchor_index;
          float selected_prompt_score = prompt_similarities[anchor_index];
          float selected_salience = saliences[anchor_index];
          const float anchor_prompt_score = selected_prompt_score;
          for (int token_index = bucket.start_index; token_index < bucket.end_index;
               ++token_index) {
            const float candidate_prompt_score = prompt_similarities[token_index];
            const float candidate_salience = saliences[token_index];
            const bool is_better =
                candidate_prompt_score > selected_prompt_score + 1e-6f ||
                (std::abs(candidate_prompt_score - selected_prompt_score) <=
                     1e-6f &&
                 (candidate_salience > selected_salience + 1e-6f ||
                  (std::abs(candidate_salience - selected_salience) <= 1e-6f &&
                   token_index < selected_index)));
            if (is_better) {
              selected_index = token_index;
              selected_prompt_score = candidate_prompt_score;
              selected_salience = candidate_salience;
            }
          }
          selected_prompt_scores.push_back(prompt_similarities[anchor_index]);
          selected_saliences.push_back(saliences[anchor_index]);
          decision.mean_reference_prompt_similarity +=
              prompt_similarities[anchor_index];
          decision.mean_reference_salience += saliences[anchor_index];
          const float prompt_gain = selected_prompt_score - anchor_prompt_score;
          const float salience_drop =
              std::max(0.0f, saliences[anchor_index] - selected_salience);
          if (selected_index != anchor_index &&
              prompt_gain >= config.local_refinement_min_prompt_gain &&
              salience_drop <= config.max_local_refinement_salience_drop) {
            refinement_candidates.push_back(BucketRefinementCandidate{
                .bucket_index = bucket_index,
                .selected_index = selected_index,
                .prompt_gain = prompt_gain,
                .selected_prompt_score = selected_prompt_score,
                .selected_salience = selected_salience,
            });
          }
        }
        const int max_local_refinements = std::clamp(
            static_cast<int>(std::ceil(static_cast<double>(target_tokens) *
                                       static_cast<double>(
                                           config.max_local_refinement_fraction))),
            0, target_tokens);
        std::sort(refinement_candidates.begin(), refinement_candidates.end(),
                  [](const BucketRefinementCandidate& lhs,
                     const BucketRefinementCandidate& rhs) {
                    return lhs.prompt_gain > rhs.prompt_gain + 1e-6f ||
                           (std::abs(lhs.prompt_gain - rhs.prompt_gain) <= 1e-6f &&
                            (lhs.selected_prompt_score >
                                 rhs.selected_prompt_score + 1e-6f ||
                             (std::abs(lhs.selected_prompt_score -
                                       rhs.selected_prompt_score) <= 1e-6f &&
                              lhs.bucket_index < rhs.bucket_index)));
                  });
        for (int candidate_index = 0;
             candidate_index < refinement_candidates.size() &&
             candidate_index < max_local_refinements;
             ++candidate_index) {
          const auto& candidate = refinement_candidates[candidate_index];
          decision.proposed_local_refinement_count += 1;
          decision.local_refinement_count += 1;
          decision.mean_local_refinement_prompt_gain += candidate.prompt_gain;
          decision.max_local_refinement_prompt_gain = std::max(
              decision.max_local_refinement_prompt_gain, candidate.prompt_gain);
          selected_indices[candidate.bucket_index] = candidate.selected_index;
          selected_prompt_scores[candidate.bucket_index] =
              candidate.selected_prompt_score;
          selected_saliences[candidate.bucket_index] = candidate.selected_salience;
        }
        decision.local_refinement_candidate_count =
            refinement_candidates.size();
        if (decision.local_refinement_count > 0) {
          decision.mean_local_refinement_prompt_gain /=
              static_cast<float>(decision.local_refinement_count);
        }
        decision.token_indices = std::move(selected_indices);
        for (int bucket_index = 0; bucket_index < decision.token_indices.size();
             ++bucket_index) {
          decision.mean_prompt_similarity += selected_prompt_scores[bucket_index];
          decision.mean_salience += selected_saliences[bucket_index];
        }
      } else {
        std::vector<float> static_scores(feature_matrix.token_count, 0.0f);
        for (int token_index = 0; token_index < feature_matrix.token_count;
             ++token_index) {
          static_scores[token_index] =
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
            const float candidate_base_score =
                config.prompt_similarity_weight * prompt_similarities[token_index] +
                static_scores[token_index];
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
            const float adjusted_score = candidate_base_score * redundancy_scale;
            const bool is_better =
                adjusted_score > best_adjusted_score + 1e-6f ||
                (std::abs(adjusted_score - best_adjusted_score) <= 1e-6f &&
                 (max_redundancy < best_redundancy - 1e-6f ||
                  (std::abs(max_redundancy - best_redundancy) <= 1e-6f &&
                   (candidate_base_score > best_base_score + 1e-6f ||
                    (std::abs(candidate_base_score - best_base_score) <=
                         1e-6f &&
                     token_index < best_index)))));
            if (is_better) {
              best_index = token_index;
              best_adjusted_score = adjusted_score;
              best_redundancy = max_redundancy;
              best_base_score = candidate_base_score;
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
        decision.token_indices = std::move(selected_order);
        decision.reference_token_indices = decision.token_indices;
      }
      std::sort(decision.token_indices.begin(), decision.token_indices.end());
      decision.mean_prompt_similarity /=
          static_cast<float>(decision.token_indices.size());
      decision.mean_salience /=
          static_cast<float>(decision.token_indices.size());
      if (!decision.reference_token_indices.empty()) {
        decision.mean_reference_prompt_similarity /=
            static_cast<float>(decision.reference_token_indices.size());
        decision.mean_reference_salience /=
            static_cast<float>(decision.reference_token_indices.size());
      }
      if (config.strategy == VisionTokenPruningStrategy::kPromptConditionedV2) {
        if (config.min_global_mean_prompt_similarity > 0.0f &&
            decision.mean_prompt_similarity <
                config.min_global_mean_prompt_similarity) {
          decision.controller_kept_uniform = true;
          decision.controller_reason = "low_mean_prompt_similarity";
        }
        if (config.min_global_mean_salience > 0.0f &&
            decision.mean_salience < config.min_global_mean_salience) {
          decision.controller_kept_uniform = true;
          if (decision.controller_reason.empty()) {
            decision.controller_reason = "low_mean_salience";
          } else {
            decision.controller_reason = absl::StrCat(
                decision.controller_reason, "+low_mean_salience");
          }
        }
        if (decision.controller_kept_uniform) {
          decision.token_indices = decision.reference_token_indices;
          decision.mean_prompt_similarity =
              decision.mean_reference_prompt_similarity;
          decision.mean_salience = decision.mean_reference_salience;
          decision.local_refinement_count = 0;
          decision.mean_local_refinement_prompt_gain = 0.0f;
          decision.max_local_refinement_prompt_gain = 0.0f;
        }
      }
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
  const int num_logged_reference_indices = std::min<int>(
      config.max_logged_token_indices, decision.reference_token_indices.size());
  decision.logged_reference_token_indices.assign(
      decision.reference_token_indices.begin(),
      decision.reference_token_indices.begin() + num_logged_reference_indices);
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
  if (IsPromptConditionedStrategy(config.strategy) &&
      target_tokens < total_tokens) {
    if (cached_text_embeddings == nullptr) {
      return absl::InvalidArgumentError(
          "cached_text_embeddings must be provided when using "
          "prompt-conditioned visual pruning.");
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
  absl::StatusOr<ExecutorVisionData> pruned_vision_data_status =
      config.fuse_projection_prune_pack && decision.token_indices.size() < total_tokens
          ? CreateSparseSelectedExecutorVisionData(vision_data,
                                                   decision.token_indices)
          : SelectExecutorVisionTokens(vision_data, decision.token_indices);
  ASSIGN_OR_RETURN(auto pruned_vision_data, std::move(pruned_vision_data_status));
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
