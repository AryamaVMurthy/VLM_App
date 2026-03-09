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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_

#include <string>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

using CachedTextEmbeddings = ExecutorTextData::CachedTextEmbeddings;

enum class VisionTokenPruningStrategy {
  kUniform,
  kPromptConditionedV1,
};

const char* VisionTokenPruningStrategyToString(
    VisionTokenPruningStrategy strategy);

absl::StatusOr<VisionTokenPruningStrategy> ParseVisionTokenPruningStrategy(
    const std::string& strategy);

struct VisionTokenPruningConfig {
  VisionTokenPruningStrategy strategy =
      VisionTokenPruningStrategy::kUniform;
  float prompt_similarity_weight = 0.6f;
  float salience_weight = 0.3f;
  float redundancy_penalty_weight = 0.1f;
  int max_logged_token_indices = 8;
};

struct VisionTokenPruningDecision {
  VisionTokenPruningStrategy strategy =
      VisionTokenPruningStrategy::kUniform;
  std::vector<int> token_indices;
  std::vector<int> logged_token_indices;
  int original_token_count = 0;
  float mean_prompt_similarity = 0.0f;
  float mean_salience = 0.0f;

  std::string ToLogString() const;
};

struct PromptConditioningSignals {
  std::vector<float> pooled_prompt_features;
  std::vector<float> normalized_prompt_token_features;
  int prompt_token_count = 0;
  int feature_dim = 0;
};

struct PrunedExecutorVisionData {
  ExecutorVisionData vision_data;
  VisionTokenPruningDecision decision;
};

// Returns the number of vision tokens encoded in the vision data. The token
// axis is the second-to-last dimension of the embeddings tensor.
absl::StatusOr<int> GetExecutorVisionTokenCount(
    const ExecutorVisionData& vision_data);

// Selects a uniformly spaced subset of token indices. The returned indices are
// in ascending order and preserve the first and last token whenever more than
// one token is requested.
absl::StatusOr<std::vector<int>> BuildUniformTokenSelection(
    int total_tokens, int target_tokens);

// Returns a copy of the vision data with only the selected token indices kept.
// The token axis is the second-to-last dimension of the embeddings and
// per-layer embeddings tensors.
absl::StatusOr<ExecutorVisionData> SelectExecutorVisionTokens(
    const ExecutorVisionData& vision_data, absl::Span<const int> token_indices);

// Builds a dense prompt-conditioning vector in the same dimensionality as the
// projected vision tokens. The vector is mean-pooled from cached real FastVLM
// text embeddings.
absl::StatusOr<std::vector<float>> BuildPromptConditioningVector(
    const CachedTextEmbeddings& cached_text_embeddings, int feature_dim);

// Builds prompt-conditioning signals from cached real FastVLM text embeddings.
// The returned token features are normalized per token and flattened in
// token-major order.
absl::StatusOr<PromptConditioningSignals> BuildPromptConditioningSignals(
    const CachedTextEmbeddings& cached_text_embeddings, int feature_dim);

// Builds a token-selection decision for the requested target token count.
// The prompt-conditioned strategy uses a lightweight prompt-to-vision
// attention proxy, token salience, and a slight redundancy penalty to choose
// token indices.
absl::StatusOr<VisionTokenPruningDecision> BuildVisionTokenPruningDecision(
    const ExecutorVisionData& vision_data, int target_tokens,
    const PromptConditioningSignals* prompt_conditioning_signals,
    const VisionTokenPruningConfig& config);

// Applies a fixed token budget to the vision data using a uniform token
// selection strategy. If the budget is greater than or equal to the current
// token count, the original tensor shapes are preserved.
absl::StatusOr<ExecutorVisionData> PruneExecutorVisionData(
    const ExecutorVisionData& vision_data, int target_tokens);

// Applies a fixed token budget to the vision data using the requested pruning
// strategy and returns both the pruned tensors and the decision metadata.
absl::StatusOr<PrunedExecutorVisionData> PruneExecutorVisionData(
    const ExecutorVisionData& vision_data, int target_tokens,
    const CachedTextEmbeddings* cached_text_embeddings,
    const VisionTokenPruningConfig& config);

// Util function for combining multiple ExecutorVisionData into a single
// ExecutorVisionData, by concatenating the vision embeddings in a single
// tensor buffer.
//
// Specifically, if the elements of input ExecutorVisionData have TensorBuffer
// with shapes,
//  [batch_size, num_token_1, feature_dim].
//  [batch_size, num_token_2, feature_dim].
//  ...
//  [batch_size, num_token_n, feature_dim].
// The output ExecutorVisionData will have TensorBuffer with shape,
// [batch_size, 1, num_token_1 + num_token_2 + ... + num_token_n,
// feature_dim].
//
// Or if the elements of input ExecutorVisionData have TensorBuffer
// with shapes,
//  [batch_size, dim1, num_token_1, feature_dim].
//  [batch_size, dim1, num_token_2, feature_dim].
//  ...
//  [batch_size, dim1, num_token_n, feature_dim].
// The output ExecutorVisionData will have TensorBuffer with shape,
// [batch_size, dim1, num_token_1 + num_token_2 + ... + num_token_n,
// feature_dim].
absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data);

// Util function for combining multiple ExecutorAudioData into a single
// ExecutorAudioData, by concatenating the audio embeddings in a single tensor
// buffer.
//
// Specifically, if the elements of input ExecutorAudioData have TensorBuffer
// with shapes,
//  [batch_size, num_token_1, feature_dim].
//  [batch_size, num_token_2, feature_dim].
//  ...
//  [batch_size, num_token_n, feature_dim].
// The output ExecutorAudioData will have TensorBuffer with shape,
// [batch_size, num_token_1 + num_token_2 + ... + num_token_n, feature_dim].
absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_EXECUTOR_DATA_UTIL_H_
