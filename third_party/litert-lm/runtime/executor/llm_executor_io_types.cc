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

#include "runtime/executor/llm_executor_io_types.h"

#include <atomic>
#include <ios>
#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/util/logging_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

constexpr char kFieldIndent[] = "  ";

absl::Status KvCacheQuantizationParams::Validate(
    absl::string_view tensor_name) const {
  if (source_element_type == ::litert::ElementType::None) {
    return absl::InvalidArgumentError(absl::StrCat(
        "KvCacheQuantizationParams for '", tensor_name,
        "' must specify a source_element_type."));
  }
  if (!std::isfinite(scale) || scale <= 0.0f) {
    return absl::InvalidArgumentError(
        absl::StrCat("KvCacheQuantizationParams for '", tensor_name,
                     "' must specify a positive finite scale."));
  }
  return absl::OkStatus();
}

absl::Status PrefillDecodeHandoff::Validate() const {
  if (current_step <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("PrefillDecodeHandoff::current_step must be positive, got ",
                     current_step, "."));
  }
  if (processed_token_ids.size() + 1 != current_step) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrefillDecodeHandoff token state is inconsistent: processed tokens=",
        processed_token_ids.size(), ", current_step=", current_step, "."));
  }
  if (kv_cache_buffers.empty()) {
    return absl::InvalidArgumentError(
        "PrefillDecodeHandoff::kv_cache_buffers must not be empty.");
  }
  for (const auto& [tensor_name, quantization_params] : kv_cache_quantization) {
    if (!kv_cache_buffers.contains(tensor_name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrefillDecodeHandoff::kv_cache_quantization contains "
                       "metadata for missing tensor '",
                       tensor_name, "'."));
    }
    RETURN_IF_ERROR(quantization_params.Validate(tensor_name));
  }
  return absl::OkStatus();
}

absl::Status ExecutorTextData::CachedTextEmbeddings::Validate() const {
  if (token_count < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("CachedTextEmbeddings::token_count must be non-negative, "
                     "but got: ",
                     token_count));
  }
  if (floats_per_token <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("CachedTextEmbeddings::floats_per_token must be "
                     "positive, but got: ",
                     floats_per_token));
  }
  if (values.size() !=
      static_cast<size_t>(token_count) *
          static_cast<size_t>(floats_per_token)) {
    return absl::InvalidArgumentError(
        absl::StrCat("CachedTextEmbeddings::values size ", values.size(),
                     " does not match token_count * floats_per_token = ",
                     token_count, " * ", floats_per_token, "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<const float>>
ExecutorTextData::CachedTextEmbeddings::GetTokenEmbedding(
    int token_index) const {
  RETURN_IF_ERROR(Validate());
  if (token_index < 0 || token_index >= token_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("token_index must be in [0, ", token_count,
                     "), but got: ", token_index));
  }
  const size_t offset =
      static_cast<size_t>(token_index) * static_cast<size_t>(floats_per_token);
  return absl::MakeConstSpan(values).subspan(offset, floats_per_token);
}

ExecutorTextData::ExecutorTextData(::litert::TensorBuffer&& token_ids)
    : token_ids_(std::move(token_ids)) {}

const ::litert::TensorBuffer& ExecutorTextData::GetTokenIds() const {
  return token_ids_;
}

::litert::TensorBuffer& ExecutorTextData::GetMutableTokenIds() {
  return token_ids_;
}

void ExecutorTextData::SetTokenIds(::litert::TensorBuffer&& token_ids) {
  token_ids_ = std::move(token_ids);
}

void ExecutorTextData::SetCachedTextEmbeddings(
    CachedTextEmbeddings&& cached_text_embeddings) {
  cached_text_embeddings_ = std::move(cached_text_embeddings);
}

void ExecutorTextData::SetCachedTextEmbeddings(
    std::optional<CachedTextEmbeddings>&& cached_text_embeddings) {
  cached_text_embeddings_ = std::move(cached_text_embeddings);
}

std::ostream& operator<<(std::ostream& os, const ExecutorTextData& text_data) {
  os << "ExecutorTextData: {\n"
     << kFieldIndent << "TokenIds: " << text_data.GetTokenIds() << "\n";
  os << kFieldIndent << "CachedTextEmbeddings: ";
  if (text_data.GetCachedTextEmbeddings().has_value()) {
    const auto& cached_text_embeddings =
        text_data.GetCachedTextEmbeddings().value();
    os << "{ token_count=" << cached_text_embeddings.token_count
       << ", floats_per_token=" << cached_text_embeddings.floats_per_token
       << " }";
  } else {
    os << "nullopt";
  }
  os << "\n"
     << "}";
  return os;
}

ExecutorVisionData::ExecutorVisionData(
    std::optional<::litert::TensorBuffer>&& embeddings,
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings)
    : embeddings_(std::move(embeddings)),
      per_layer_embeddings_(std::move(per_layer_embeddings)) {}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorVisionData::GetEmbeddingsPtr() const {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError("ExecutorVisionData::embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorVisionData::GetMutableEmbeddingsPtr() {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::embeddings_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorVisionData::GetPerLayerEmbeddingsPtr() const {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::per_layer_embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorVisionData::GetMutablePerLayerEmbeddingsPtr() {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorVisionData::per_layer_embeddings_ is not set.");
}

void ExecutorVisionData::SetEmbeddings(
    std::optional<::litert::TensorBuffer>&& embeddings) {
  embeddings_ = std::move(embeddings);
}

void ExecutorVisionData::SetPerLayerEmbeddings(
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
  per_layer_embeddings_ = std::move(per_layer_embeddings);
}

void ExecutorVisionData::SetSelectedTokenIndices(
    const std::vector<int>& selected_token_indices) {
  selected_token_indices_ = selected_token_indices;
}

void ExecutorVisionData::SetSelectedTokenIndices(
    std::vector<int>&& selected_token_indices) {
  selected_token_indices_ = std::move(selected_token_indices);
}

void ExecutorVisionData::SetSelectedTokenIndices(
    std::optional<std::vector<int>>&& selected_token_indices) {
  selected_token_indices_ = std::move(selected_token_indices);
}

void ExecutorVisionData::ClearSelectedTokenIndices() {
  selected_token_indices_.reset();
}

// Helper function to print a field from StatusOr<const TensorBuffer*>
static void PrintOptionalTensorBufferFieldFromStatusOr(
    std::ostream& os, const std::string& field_name,
    const absl::StatusOr<const ::litert::TensorBuffer*>& opt_buffer_status,
    const std::string& indent) {
  os << indent << field_name << ": ";
  if (opt_buffer_status.ok()) {
    const ::litert::TensorBuffer* buffer_ptr = opt_buffer_status.value();
    if (buffer_ptr) {  // Should always be true.
      os << *buffer_ptr;
    } else {  // Should not happen if status is ok and value is a pointer
      os << "null (unexpected)";
    }
  } else {
    os << "nullopt (" << opt_buffer_status.status().message() << ")";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorVisionData& vision_data) {
  os << "ExecutorVisionData: {\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "Embeddings", vision_data.GetEmbeddingsPtr(), kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "PerLayerEmbeddings", vision_data.GetPerLayerEmbeddingsPtr(),
      kFieldIndent);
  os << "\n" << kFieldIndent << "SelectedTokenIndices: ";
  if (vision_data.GetSelectedTokenIndices().has_value()) {
    os << "["
       << absl::StrJoin(*vision_data.GetSelectedTokenIndices(), ", ")
       << "]";
  } else {
    os << "nullopt";
  }
  os << "\n"
     << "}";
  return os;
}

ExecutorAudioData::ExecutorAudioData(
    std::optional<::litert::TensorBuffer>&& embeddings,
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings,
    int valid_tokens)
    : embeddings_(std::move(embeddings)),
      per_layer_embeddings_(std::move(per_layer_embeddings)),
      valid_tokens_(valid_tokens) {}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorAudioData::GetEmbeddingsPtr() const {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError("ExecutorAudioData::embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorAudioData::GetMutableEmbeddingsPtr() {
  if (embeddings_.has_value()) {
    return &embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::embeddings_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorAudioData::GetPerLayerEmbeddingsPtr() const {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::per_layer_embeddings_ is not set.");
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorAudioData::GetMutablePerLayerEmbeddingsPtr() {
  if (per_layer_embeddings_.has_value()) {
    return &per_layer_embeddings_.value();
  }
  return absl::NotFoundError(
      "ExecutorAudioData::per_layer_embeddings_ is not set.");
}

int ExecutorAudioData::GetValidTokens() const { return valid_tokens_; }

void ExecutorAudioData::SetEmbeddings(
    std::optional<::litert::TensorBuffer>&& embeddings) {
  embeddings_ = std::move(embeddings);
}

void ExecutorAudioData::SetPerLayerEmbeddings(
    std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
  per_layer_embeddings_ = std::move(per_layer_embeddings);
}

void ExecutorAudioData::SetValidTokens(int valid_tokens) {
  valid_tokens_ = valid_tokens;
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorAudioData& audio_data) {
  os << "ExecutorAudioData: {\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "Embeddings", audio_data.GetEmbeddingsPtr(), kFieldIndent);
  os << "\n";
  PrintOptionalTensorBufferFieldFromStatusOr(
      os, "PerLayerEmbeddings", audio_data.GetPerLayerEmbeddingsPtr(),
      kFieldIndent);
  os << "\n";
  os << kFieldIndent << "ValidTokens: " << audio_data.GetValidTokens();
  os << "\n"
     << "}";
  return os;
}

ExecutorInputs::ExecutorInputs(std::optional<ExecutorTextData>&& text_data,
                               std::optional<ExecutorVisionData>&& vision_data,
                               std::optional<ExecutorAudioData>&& audio_data)
    : text_data_(std::move(text_data)),
      vision_data_(std::move(vision_data)),
      audio_data_(std::move(audio_data)) {}

absl::StatusOr<const ExecutorTextData*> ExecutorInputs::GetTextDataPtr() const {
  if (text_data_.has_value()) {
    return &text_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::text_data_ is not set.");
}

absl::StatusOr<ExecutorTextData*> ExecutorInputs::GetMutableTextDataPtr() {
  if (text_data_.has_value()) {
    return &text_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::text_data_ is not set.");
}

absl::StatusOr<const ExecutorVisionData*> ExecutorInputs::GetVisionDataPtr()
    const {
  if (vision_data_.has_value()) {
    return &vision_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::vision_data_ is not set.");
}

absl::StatusOr<ExecutorVisionData*> ExecutorInputs::GetMutableVisionDataPtr() {
  if (vision_data_.has_value()) {
    return &vision_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::vision_data_ is not set.");
}

absl::StatusOr<const ExecutorAudioData*> ExecutorInputs::GetAudioDataPtr()
    const {
  if (audio_data_.has_value()) {
    return &audio_data_.value();
  }
  return absl::NotFoundError("ExecutorInputs::audio_data_ is not set.");
}

absl::StatusOr<ExecutorAudioData*> ExecutorInputs::GetMutableAudioDataPtr() {
  if (audio_data_.has_value()) {
    return &audio_data_.value();
  }
  return absl::NotFoundError(
      "ExecutorInputs::audio_data_ is not set.");
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetTextTokenIdsPtr() const {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for TokenIds).");
  }
  return &(text_data_->GetTokenIds());
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableTextTokenIdsPtr() {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for "
        "TokenIds).");
  }
  return &(text_data_->GetMutableTokenIds());
}

absl::StatusOr<const ExecutorTextData::CachedTextEmbeddings*>
ExecutorInputs::GetCachedTextEmbeddingsPtr() const {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for cached text "
        "embeddings).");
  }
  if (!text_data_->GetCachedTextEmbeddings().has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_.cached_text_embeddings_ is not set.");
  }
  return &text_data_->GetCachedTextEmbeddings().value();
}

absl::StatusOr<ExecutorTextData::CachedTextEmbeddings*>
ExecutorInputs::GetMutableCachedTextEmbeddingsPtr() {
  if (!text_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_ is not set (required for mutable cached "
        "text embeddings).");
  }
  if (!text_data_->GetMutableCachedTextEmbeddings().has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::text_data_.cached_text_embeddings_ is not set.");
  }
  return &text_data_->GetMutableCachedTextEmbeddings().value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetVisionEmbeddingsPtr() const {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required for Vision "
        "Embeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> embeddings_ptr_status =
      vision_data_->GetEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(embeddings_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableVisionEmbeddingsPtr() {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required "
        "for Vision Embeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> embeddings_ptr_status =
      vision_data_->GetMutableEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(
        embeddings_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetVisionPerLayerEmbeddingsPtr() const {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required for Vision "
        "PerLayerEmbeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> per_layer_ptr_status =
      vision_data_->GetPerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(per_layer_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableVisionPerLayerEmbeddingsPtr() {
  if (!vision_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::vision_data_ is not set (required "
        "for Vision PerLayerEmbeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> per_layer_ptr_status =
      vision_data_->GetMutablePerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(
        per_layer_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::vision_data_: ",
                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetAudioEmbeddingsPtr() const {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for Audio "
        "Embeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> embeddings_ptr_status =
      audio_data_->GetEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(embeddings_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableAudioEmbeddingsPtr() {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for "
        "Audio Embeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> embeddings_ptr_status =
      audio_data_->GetMutableEmbeddingsPtr();
  if (!embeddings_ptr_status.ok()) {
    return absl::Status(
        embeddings_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                     embeddings_ptr_status.status().message()));
  }
  return embeddings_ptr_status.value();
}

absl::StatusOr<const ::litert::TensorBuffer*>
ExecutorInputs::GetAudioPerLayerEmbeddingsPtr() const {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for Audio "
        "PerLayerEmbeddings).");
  }
  absl::StatusOr<const ::litert::TensorBuffer*> per_layer_ptr_status =
      audio_data_->GetPerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(per_layer_ptr_status.status().code(),
                        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

absl::StatusOr<::litert::TensorBuffer*>
ExecutorInputs::GetMutableAudioPerLayerEmbeddingsPtr() {
  if (!audio_data_.has_value()) {
    return absl::NotFoundError(
        "ExecutorInputs::audio_data_ is not set (required for "
        "Audio PerLayerEmbeddings).");
  }
  absl::StatusOr<::litert::TensorBuffer*> per_layer_ptr_status =
      audio_data_->GetMutablePerLayerEmbeddingsPtr();
  if (!per_layer_ptr_status.ok()) {
    return absl::Status(
        per_layer_ptr_status.status().code(),
        absl::StrCat("Within ExecutorInputs::audio_data_: ",
                     per_layer_ptr_status.status().message()));
  }
  return per_layer_ptr_status.value();
}

void ExecutorInputs::SetTextData(ExecutorTextData&& text_data) {
  text_data_ = std::move(text_data);
}

void ExecutorInputs::SetVisionData(
    std::optional<ExecutorVisionData>&& vision_data) {
  vision_data_ = std::move(vision_data);
}

void ExecutorInputs::SetAudioData(
    std::optional<ExecutorAudioData>&& audio_data) {
  audio_data_ = std::move(audio_data);
}

std::ostream& operator<<(std::ostream& os, const ExecutorInputs& inputs) {
  os << "ExecutorInputs: {\n";

  os << kFieldIndent << "TextData: ";
  absl::StatusOr<const ExecutorTextData*> text_data_status =
      inputs.GetTextDataPtr();
  if (text_data_status.ok()) {
    os << *text_data_status.value();  // Relies on TextData's operator<<
  } else {
    os << "nullopt (" << text_data_status.status().message() << ")";
  }
  os << "\n";

  os << kFieldIndent << "VisionData: ";
  absl::StatusOr<const ExecutorVisionData*> vision_data_status =
      inputs.GetVisionDataPtr();
  if (vision_data_status.ok()) {
    os << *vision_data_status.value();  // Relies on VisionData's operator<<
  } else {
    os << "nullopt (" << vision_data_status.status().message() << ")";
  }
  os << "\n";

  os << kFieldIndent << "AudioData: ";
  absl::StatusOr<const ExecutorAudioData*> audio_data_status =
      inputs.GetAudioDataPtr();
  if (audio_data_status.ok()) {
    os << *audio_data_status.value();  // Relies on AudioData's operator<<
  } else {
    os << "nullopt (" << audio_data_status.status().message() << ")";
  }
  os << "\n"
     << "}";
  return os;
}

// --- ExecutorPrefillParams Implementation ---
ExecutorPrefillParams::ExecutorPrefillParams(
    int current_step, bool wait_for_completion, const std::atomic_bool* cancel,
    std::optional<int> max_prefill_sequence_length)
    : current_step_(current_step),
      wait_for_completion_(wait_for_completion),
      cancel_(cancel),
      max_prefill_sequence_length_(max_prefill_sequence_length) {}

int ExecutorPrefillParams::GetCurrentStep() const { return current_step_; }

void ExecutorPrefillParams::SetCurrentStep(int current_step) {
  current_step_ = current_step;
}

bool ExecutorPrefillParams::GetWaitForCompletion() const {
  return wait_for_completion_;
}

void ExecutorPrefillParams::SetWaitForCompletion(bool wait_for_completion) {
  wait_for_completion_ = wait_for_completion;
}

const std::atomic_bool* ExecutorPrefillParams::GetCancelFlag() const {
  return cancel_;
}

void ExecutorPrefillParams::SetCancelFlag(const std::atomic_bool* cancel) {
  cancel_ = cancel;
}

absl::StatusOr<int> ExecutorPrefillParams::GetMaxPrefillSequenceLength() const {
  if (max_prefill_sequence_length_.has_value()) {
    return max_prefill_sequence_length_.value();
  }
  return absl::NotFoundError(
      "ExecutorPrefillParams::max_prefill_sequence_length_ is not set.");
}

void ExecutorPrefillParams::SetMaxPrefillSequenceLength(
    std::optional<int> max_prefill_sequence_length) {
  max_prefill_sequence_length_ = max_prefill_sequence_length;
}

std::ostream& operator<<(std::ostream& os,
                         const ExecutorPrefillParams& params) {
  os << "ExecutorPrefillParams: {\n"
     << kFieldIndent << "CurrentStep: " << params.GetCurrentStep() << "\n"
     << kFieldIndent << "WaitForCompletion: " << std::boolalpha
     << params.GetWaitForCompletion() << "\n"
     << kFieldIndent << "CancelFlag: ";
  if (params.GetCancelFlag() != nullptr) {
    os << (params.GetCancelFlag()->load(std::memory_order_relaxed)
               ? "true (atomic)"
               : "false (atomic)");
  } else {
    os << "nullptr";
  }
  os << "\n" << kFieldIndent << "MaxPrefillSequenceLength: ";
  absl::StatusOr<int> max_prefill_sequence_length =
      params.GetMaxPrefillSequenceLength();
  if (max_prefill_sequence_length.ok()) {
    os << max_prefill_sequence_length.value();
  } else {
    os << "nullopt";
  }
  os << "\n"
     << "}";
  return os;
}

// --- ExecutorDecodeParams Implementation ---
void ExecutorDecodeParams::SetConstraintDecoder(
    ConstrainedDecoder* constraint) {
  constraint_decoder_ = constraint;
}

bool ExecutorDecodeParams::HasConstraintDecoder() const {
  return constraint_decoder_ != nullptr;
}

ConstrainedDecoder* ExecutorDecodeParams::GetConstraintDecoder() const {
  return constraint_decoder_;
}

std::ostream& operator<<(std::ostream& os, const ExecutorDecodeParams& params) {
  os << "ExecutorDecodeParams: {\n";
  os << kFieldIndent << "ConstraintDecoder: ";
  if (params.HasConstraintDecoder()) {
    os << params.GetConstraintDecoder();
  } else {
    os << "not set";
  }
  os << "\n"
     << "}";
  return os;
}

}  // namespace litert::lm
