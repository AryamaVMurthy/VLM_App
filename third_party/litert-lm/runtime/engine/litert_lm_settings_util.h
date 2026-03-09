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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_UTIL_H_

#include <optional>

#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/litert_lm_settings.h"

namespace litert {
namespace lm {

std::optional<Backend> GetSamplerBackend(const LiteRtLmSettings& settings);

SessionConfig CreateSessionConfig(const LiteRtLmSettings& settings);

struct ConversationOptionalArgs {
  std::optional<ConstraintArg> decoding_constraint = std::nullopt;
  std::optional<int> max_output_tokens = std::nullopt;
};

std::optional<ConstraintProviderConfig> CreateConversationConstraintProviderConfig(
    const LiteRtLmSettings& settings);

ConversationOptionalArgs CreateConversationOptionalArgs(
    const LiteRtLmSettings& settings);

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_SETTINGS_UTIL_H_
