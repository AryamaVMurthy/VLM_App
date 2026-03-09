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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_H_

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl

namespace litert {
namespace lm {

// Input data type for GPU Convolution and Fully Connected operations.
enum class ConvType {
  kAuto,   // Either float32/16 or int8 depending on the model.
  kFloat,  // Either float32 or float16 depending on the activation data type.
  kInt8,   // int8 quantized. Better latency with risk of less accuracy.
};

struct LiteRtLmSettings {
  std::string backend = "gpu";
  std::optional<std::string> vision_backend = std::nullopt;
  std::optional<std::string> audio_backend = std::nullopt;
  std::string sampler_backend = "";
  std::string model_path;
  bool load_model_from_descriptor = false;
  std::string input_prompt = "What is the tallest building in the world?";
  std::optional<std::string> expected_output = std::nullopt;
  std::optional<std::string> log_sink_file = std::nullopt;
  int max_num_tokens = 0;
  int max_output_tokens = -1;
  int max_num_images = 0;
  int max_visual_tokens = 0;
  std::string visual_token_pruning_strategy = "uniform";
  absl::LogSeverity min_log_level = absl::LogSeverity::kInfo;
  std::set<int> prefill_batch_sizes;
  int num_output_candidates = 1;
  bool benchmark = false;
  int benchmark_prefill_tokens = 0;
  int benchmark_decode_tokens = 0;
  bool async = true;
  bool report_peak_memory_footprint = false;
  bool force_f32 = false;
  bool multi_turns = false;
  int num_cpu_threads = 0;
  bool gpu_external_tensor_mode = false;
  bool configure_magic_numbers = true;
  bool verify_magic_numbers = false;
  bool clear_kv_cache_before_prefill = true;
  int num_logits_to_print_after_decode = 0;
  std::optional<std::string> score_target_text = std::nullopt;
  bool gpu_madvise_original_shared_tensors = true;
  bool disable_cache = false;
  int prefill_chunk_size = -1;
  std::string preferred_device_substr = "";
  int num_threads_to_upload = -1;
  int num_threads_to_compile = -1;
  bool convert_weights_on_gpu = true;
  bool wait_for_weights_conversion_complete_in_benchmark = true;
  bool optimize_shader_compilation = true;
  bool share_constant_tensors = true;
  bool use_session = false;
  int num_iterations = 1;
  std::string litert_dispatch_lib_dir = "";
  bool sampler_handles_input = true;
  ConvType conv_type = ConvType::kAuto;
  bool cache_compiled_shaders_only = false;
  std::string constraint_regex = "";
  bool use_submodel = false;
};

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_H_
