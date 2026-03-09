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

#include "runtime/engine/litert_lm_settings_util.h"

#include <optional>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/engine/io_types.h"

namespace litert {
namespace lm {
std::optional<Backend> GetSamplerBackend(const LiteRtLmSettings& settings) {
  if (settings.sampler_backend.empty()) {
    return std::nullopt;
  }
  const absl::StatusOr<Backend> sampler_backend =
      GetBackendFromString(settings.sampler_backend);
  if (!sampler_backend.ok()) {
    ABSL_LOG(WARNING) << "Ignore invalid sampler backend string: "
                      << sampler_backend.status();
    return std::nullopt;
  }
  return *sampler_backend;
}

SessionConfig CreateSessionConfig(const LiteRtLmSettings& settings) {
  auto session_config = SessionConfig::CreateDefault();
  session_config.SetNumOutputCandidates(settings.num_output_candidates);
  if (const std::optional<Backend> sampler_backend =
          GetSamplerBackend(settings);
      sampler_backend.has_value()) {
    session_config.SetSamplerBackend(*sampler_backend);
  }
  if (settings.vision_backend.has_value()) {
    session_config.SetVisionModalityEnabled(true);
  }
  if (settings.audio_backend.has_value()) {
    session_config.SetAudioModalityEnabled(true);
  }
  session_config.SetMaxVisualTokens(settings.max_visual_tokens);
  session_config.SetVisualTokenPruningStrategy(
      settings.visual_token_pruning_strategy);
  return session_config;
}

std::optional<ConstraintProviderConfig> CreateConversationConstraintProviderConfig(
    const LiteRtLmSettings& settings) {
  if (settings.constraint_regex.empty()) {
    return std::nullopt;
  }
  return ConstraintProviderConfig(LlGuidanceConfig());
}

ConversationOptionalArgs CreateConversationOptionalArgs(
    const LiteRtLmSettings& settings) {
  ConversationOptionalArgs optional_args;
  if (!settings.constraint_regex.empty()) {
    optional_args.decoding_constraint = LlGuidanceConstraintArg{
        .constraint_type = LlgConstraintType::kRegex,
        .constraint_string = settings.constraint_regex,
    };
  }
  if (settings.max_output_tokens > 0) {
    optional_args.max_output_tokens = settings.max_output_tokens;
  }
  return optional_args;
}

}  // namespace lm
}  // namespace litert
