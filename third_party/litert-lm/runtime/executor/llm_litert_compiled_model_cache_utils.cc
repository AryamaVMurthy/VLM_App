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

#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

using ::litert::Expected;
using ::litert::TensorBuffer;

namespace {

constexpr absl::string_view kKvSliceKRoot = "kv_slice_k_";
constexpr absl::string_view kKvSliceVRoot = "kv_slice_v_";
constexpr absl::string_view kKvCacheKRoot = "kv_cache_k_";
constexpr absl::string_view kKvCacheVRoot = "kv_cache_v_";

bool IsKnownUnusedTypeMismatchedKvCache(absl::string_view cache_name) {
  return cache_name == "kv_cache_k_23" || cache_name == "kv_cache_v_23" ||
         cache_name == "kv_cache_k_25" || cache_name == "kv_cache_v_25";
}

absl::Status ValidateMatchingTensorLayouts(
    const TensorBuffer& source_buffer, const TensorBuffer& destination_buffer,
    absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto source_type, source_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto destination_type, destination_buffer.TensorType());
  if (source_type.Layout().Dimensions() != destination_type.Layout().Dimensions()) {
    return absl::InternalError(
        absl::StrCat("Tensor shape mismatch during handoff import for ",
                     tensor_name, ": source_dims=[",
                     absl::StrJoin(source_type.Layout().Dimensions(), ","),
                     "] destination_dims=[",
                     absl::StrJoin(destination_type.Layout().Dimensions(), ","),
                     "]"));
  }
  return absl::OkStatus();
}

absl::Status CopyMatchingTensorBufferContents(
    const TensorBuffer& source_buffer, TensorBuffer& destination_buffer,
    absl::string_view tensor_name) {
  RETURN_IF_ERROR(ValidateMatchingTensorLayouts(source_buffer,
                                                destination_buffer,
                                                tensor_name));
  LITERT_ASSIGN_OR_RETURN(auto source_size, source_buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto destination_size, destination_buffer.PackedSize());
  if (source_size != destination_size) {
    return absl::InternalError(
        absl::StrCat("Tensor packed size mismatch during handoff import for ",
                     tensor_name, "."));
  }
  LITERT_ASSIGN_OR_RETURN(
      auto source_lock,
      ::litert::TensorBufferScopedLock::Create(
          *const_cast<TensorBuffer*>(&source_buffer),
          TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto destination_lock,
      ::litert::TensorBufferScopedLock::Create(destination_buffer,
                                               TensorBuffer::LockMode::kWrite));
  std::memcpy(destination_lock.second, source_lock.second, source_size);
  return absl::OkStatus();
}

absl::Status DequantizeInt16ToFloat32(
    const TensorBuffer& source_buffer,
    const KvCacheQuantizationParams& quantization_params,
    TensorBuffer& destination_buffer, absl::string_view tensor_name) {
  RETURN_IF_ERROR(ValidateMatchingTensorLayouts(source_buffer,
                                                destination_buffer,
                                                tensor_name));
  RETURN_IF_ERROR(quantization_params.Validate(tensor_name));

  LITERT_ASSIGN_OR_RETURN(auto source_type, source_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto destination_type, destination_buffer.TensorType());
  if (source_type.ElementType() != ::litert::ElementType::Int16 ||
      destination_type.ElementType() != ::litert::ElementType::Float32) {
    return absl::InternalError(absl::StrCat(
        "Unsupported handoff type conversion for ", tensor_name, ": source=",
        static_cast<int>(source_type.ElementType()), " destination=",
        static_cast<int>(destination_type.ElementType()), "."));
  }
  if (quantization_params.source_element_type != source_type.ElementType()) {
    return absl::InternalError(absl::StrCat(
        "Quantization metadata source element type mismatch for ",
        tensor_name, "."));
  }

  LITERT_ASSIGN_OR_RETURN(auto num_elements, source_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(
      auto source_lock,
      ::litert::TensorBufferScopedLock::Create(
          *const_cast<TensorBuffer*>(&source_buffer),
          TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto destination_lock,
      ::litert::TensorBufferScopedLock::Create(destination_buffer,
                                               TensorBuffer::LockMode::kWrite));
  const auto* source_data = static_cast<const int16_t*>(source_lock.second);
  auto* destination_data = static_cast<float*>(destination_lock.second);
  for (int64_t i = 0; i < num_elements; ++i) {
    destination_data[i] =
        (static_cast<float>(source_data[i]) -
         static_cast<float>(quantization_params.zero_point)) *
        quantization_params.scale;
  }
  ABSL_LOG(INFO) << "Dequantized handoff tensor '" << tensor_name
                 << "' from Int16 to Float32 using scale="
                 << quantization_params.scale
                 << " zero_point=" << quantization_params.zero_point << ".";
  return absl::OkStatus();
}

Expected<int> GetKvCacheUpdateStartPosition(const TensorBuffer& input_pos) {
  LITERT_ASSIGN_OR_RETURN(auto positions, CopyFromTensorBuffer<int32_t>(input_pos));
  if (positions.empty()) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "input_pos must contain at least one value.");
  }
  if (positions.front() < 0) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "input_pos must be non-negative.");
  }
  return positions.front();
}

Expected<std::pair<std::string, int>> MapKvSliceToCacheName(
    absl::string_view slice_name) {
  if (absl::StartsWith(slice_name, kKvSliceKRoot)) {
    return std::make_pair(
        absl::StrCat(kKvCacheKRoot, slice_name.substr(kKvSliceKRoot.size())),
        /*axis=*/2);
  }
  if (absl::StartsWith(slice_name, kKvSliceVRoot)) {
    return std::make_pair(
        absl::StrCat(kKvCacheVRoot, slice_name.substr(kKvSliceVRoot.size())),
        /*axis=*/3);
  }
  return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              absl::StrCat("Unsupported KV slice tensor: ",
                                           slice_name));
}

Expected<void> CopyKvCacheSlice(TensorBuffer& dst_buffer,
                                const TensorBuffer& src_buffer,
                                int start_position, int axis,
                                absl::string_view cache_name) {
  LITERT_ASSIGN_OR_RETURN(auto dst_type, dst_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto src_type, src_buffer.TensorType());
  if (dst_type.ElementType() != src_type.ElementType()) {
    if (IsKnownUnusedTypeMismatchedKvCache(cache_name)) {
      ABSL_LOG(INFO)
          << "Skipping KV cache update for known unused tensor '"
          << cache_name
          << "' because the slice and cache buffer element types differ.";
      return {};
    }
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("KV cache element type mismatch for '", cache_name,
                     "'."));
  }

  const auto dst_shape = dst_type.Layout().Dimensions();
  const auto src_shape = src_type.Layout().Dimensions();
  if (dst_shape.size() != src_shape.size()) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("KV cache rank mismatch for '", cache_name, "'."));
  }
  if (axis < 0 || axis >= src_shape.size()) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("Invalid KV cache axis for '", cache_name, "'."));
  }
  for (int i = 0; i < src_shape.size(); ++i) {
    if (i == axis) {
      continue;
    }
    if (dst_shape[i] != src_shape[i]) {
      return ::litert::Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("KV cache shape mismatch for '", cache_name,
                       "' outside update axis."));
    }
  }
  if (start_position + src_shape[axis] > dst_shape[axis]) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("KV cache update for '", cache_name, "' is out of range."));
  }

  LITERT_ASSIGN_OR_RETURN(auto src_num_elements, src_type.Layout().NumElements());
  if (src_num_elements == 0) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("KV cache slice for '", cache_name, "' is empty."));
  }
  LITERT_ASSIGN_OR_RETURN(auto src_bytes, src_type.Bytes());
  const size_t element_size = src_bytes / src_num_elements;

  int64_t inner_block_size_in_elements = 1;
  for (int i = axis + 1; i < src_shape.size(); ++i) {
    inner_block_size_in_elements *= src_shape[i];
  }
  int64_t outer_block_count = 1;
  for (int i = 0; i < axis; ++i) {
    outer_block_count *= src_shape[i];
  }
  const int64_t src_outer_stride_in_elements =
      src_shape[axis] * inner_block_size_in_elements;
  const int64_t dst_outer_stride_in_elements =
      dst_shape[axis] * inner_block_size_in_elements;
  const size_t copy_size_in_bytes =
      src_shape[axis] * inner_block_size_in_elements * element_size;

  LITERT_ASSIGN_OR_RETURN(
      auto src_lock,
      ::litert::TensorBufferScopedLock::Create(
          *const_cast<TensorBuffer*>(&src_buffer),
          TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto dst_lock,
      ::litert::TensorBufferScopedLock::Create(dst_buffer,
                                               TensorBuffer::LockMode::kWrite));
  const auto* src_data = static_cast<const uint8_t*>(src_lock.second);
  auto* dst_data = static_cast<uint8_t*>(dst_lock.second);

  for (int64_t i = 0; i < outer_block_count; ++i) {
    const auto* src_outer_block_start =
        src_data + i * src_outer_stride_in_elements * element_size;
    auto* dst_outer_block_start =
        dst_data + (i * dst_outer_stride_in_elements +
                    start_position * inner_block_size_in_elements) *
                       element_size;
    std::memcpy(dst_outer_block_start, src_outer_block_start,
                copy_size_in_bytes);
  }
  return {};
}

}  // namespace

absl::Status CopyHandoffKvCacheBuffer(
    const ::litert::TensorBuffer& source_buffer,
    const std::optional<KvCacheQuantizationParams>& quantization_params,
    ::litert::TensorBuffer& destination_buffer, absl::string_view tensor_name) {
  if (source_buffer.Get() == destination_buffer.Get()) {
    ABSL_LOG(INFO) << "Skipping aliased handoff copy for '" << tensor_name
                   << "' because the CPU decode path uses a single-buffer KV "
                      "cache mirror.";
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto source_type, source_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto destination_type, destination_buffer.TensorType());
  if (source_type.ElementType() == destination_type.ElementType()) {
    return CopyMatchingTensorBufferContents(source_buffer, destination_buffer,
                                           tensor_name);
  }
  if (!quantization_params.has_value()) {
    return absl::InternalError(absl::StrCat(
        "Tensor element type mismatch during handoff import for ", tensor_name,
        ": source=", static_cast<int>(source_type.ElementType()),
        " destination=", static_cast<int>(destination_type.ElementType()),
        ". Missing quantization metadata for explicit conversion."));
  }
  return DequantizeInt16ToFloat32(source_buffer, *quantization_params,
                                  destination_buffer, tensor_name);
}

::litert::Expected<bool> ShouldDeleteKVCacheTokens(int current_step,
                                                   int start_position,
                                                   size_t context_size) {
  if (current_step - start_position < 0) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Deleted more tokens than the number of model steps processed.");
  }
  if (current_step < 0) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "current_step step is negative.");
  }
  if (start_position < 0) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "start_position is negative.");
  }
  if (current_step - start_position >= context_size - 1) {
    return true;
  }
  return false;
}

::litert::Expected<void> UpdateKvCacheFromSlices(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        kv_cache_slice_buffers,
    const ::litert::TensorBuffer& input_pos) {
  if (input_kv_cache_buffers == nullptr) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "input_kv_cache_buffers must not be null.");
  }
  LITERT_ASSIGN_OR_RETURN(const int start_position,
                          GetKvCacheUpdateStartPosition(input_pos));

  bool updated_any_buffer = false;
  for (const auto& [slice_name, slice_buffer] : kv_cache_slice_buffers) {
    if (!absl::StartsWith(slice_name, kKvSliceKRoot) &&
        !absl::StartsWith(slice_name, kKvSliceVRoot)) {
      continue;
    }

    LITERT_ASSIGN_OR_RETURN(auto cache_name_and_axis,
                            MapKvSliceToCacheName(slice_name));
    const std::string& cache_name = cache_name_and_axis.first;
    const int axis = cache_name_and_axis.second;
    auto cache_it =
        input_kv_cache_buffers->find(absl::string_view(cache_name));
    if (cache_it == input_kv_cache_buffers->end()) {
      return ::litert::Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("Missing KV cache tensor '", cache_name,
                       "' for slice '", slice_name, "'."));
    }

    LITERT_RETURN_IF_ERROR(CopyKvCacheSlice(cache_it->second, slice_buffer,
                                            start_position, axis, cache_name));
    updated_any_buffer = true;
  }

  if (!updated_any_buffer) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "No KV cache slice tensors were provided.");
  }
  return {};
}

// Function to dump the ring buffer.
::litert::Expected<void> DeleteTokensFromKvCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain) {
  // A k cache buffer is a 4D tensor with shape
  // [1, heads, context_size, embedding_size]
  // A v cache buffer is a 4D tensor with shape
  // [1, heads, embedding_size, context_size]
  for (auto& [input_name, input_buffer] : *input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto type, input_buffer.TensorType());
    const int axis = absl::StrContains(input_name, "cache_k_")   ? 2
                     : absl::StrContains(input_name, "cache_v_") ? 3
                                                                 : -1;
    if (axis == -1) {
      return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                  "Unsupported input name.");
    }
    switch (type.ElementType()) {
      case ::litert::ElementType::Int8:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int8_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Int16:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int16_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Int32:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int32_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Float32:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<float>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      default:
        return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                    "Unsupported element type.");
    }
  }
  return {};
}

::litert::Expected<bool> DeleteTokensIfNeeded(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain, int current_step,
    int& start_position, size_t context_size) {
  LITERT_ASSIGN_OR_RETURN(
      bool should_delete_tokens,
      ShouldDeleteKVCacheTokens(current_step, start_position, context_size));
  if (should_delete_tokens) {
    LITERT_RETURN_IF_ERROR(DeleteTokensFromKvCache(
        input_kv_cache_buffers,
        /*num_tokens_to_drop=*/num_tokens_to_drop,
        /*init_tokens_to_retain=*/init_tokens_to_retain));
    start_position += num_tokens_to_drop;
    return true;
  }
  return false;
}

absl::Status ExpandBuffer(const uint8_t* src_data,
                          absl::Span<const int> src_shape, uint8_t* dst_data,
                          absl::Span<const int> dst_shape,
                          size_t element_size) {
  RET_CHECK_EQ(src_shape.size(), dst_shape.size());
  int expansion_axis = -1;
  for (int i = 0; i < src_shape.size(); ++i) {
    if (src_shape[i] != dst_shape[i]) {
      if (expansion_axis != -1) {
        return absl::InvalidArgumentError(
            "Tensors differ in more than one dimension.");
      }
      if (dst_shape[i] < src_shape[i]) {
        return absl::InvalidArgumentError(
            "Destination tensor dimension is smaller than source along an "
            "axis.");
      }
      expansion_axis = i;
    }
  }
  if (expansion_axis == -1) {
    return absl::InvalidArgumentError("No expansion axis found.");
  }

  int64_t dest_total_elements = 1;
  for (int dim : dst_shape) {
    dest_total_elements *= dim;
  }
  memset(dst_data, 0, dest_total_elements * element_size);

  int64_t inner_block_size_in_elements = 1;
  for (int i = expansion_axis + 1; i < src_shape.size(); ++i) {
    inner_block_size_in_elements *= src_shape[i];
  }
  const size_t inner_block_size_in_bytes =
      inner_block_size_in_elements * element_size;

  int64_t outer_block_count = 1;
  for (int i = 0; i < expansion_axis; ++i) {
    outer_block_count *= src_shape[i];
  }

  int64_t src_outer_block_stride_in_elements =
      src_shape[expansion_axis] * inner_block_size_in_elements;
  int64_t dest_outer_block_stride_in_elements =
      dst_shape[expansion_axis] * inner_block_size_in_elements;

  for (int64_t i = 0; i < outer_block_count; ++i) {
    // Calculate the starting pointer for this outer block
    const uint8_t* src_outer_block_start =
        src_data + i * src_outer_block_stride_in_elements * element_size;
    uint8_t* dest_outer_block_start =
        dst_data + i * dest_outer_block_stride_in_elements * element_size;

    // Copy each inner block from source to destination
    for (int j = 0; j < src_shape[expansion_axis]; ++j) {
      const uint8_t* src_inner_block =
          src_outer_block_start + j * inner_block_size_in_bytes;
      uint8_t* dest_inner_block =
          dest_outer_block_start + j * inner_block_size_in_bytes;
      memcpy(dest_inner_block, src_inner_block, inner_block_size_in_bytes);
    }
  }

  return absl::OkStatus();
};

bool IsKVCacheTensor(absl::string_view tensor_name) {
  return absl::StartsWith(tensor_name, "kv_cache_") ||
         absl::StartsWith(tensor_name, "k_cache_") ||
         absl::StartsWith(tensor_name, "v_cache_");
}

}  // namespace litert::lm
