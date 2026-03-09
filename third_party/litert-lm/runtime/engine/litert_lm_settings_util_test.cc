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

#include <variant>

#include <gtest/gtest.h>
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/engine/litert_lm_settings.h"

namespace litert {
namespace lm {
namespace {

TEST(LiteRtLmSettingsUtilTest, CreateSessionConfigPropagatesMaxVisualTokens) {
  LiteRtLmSettings settings;
  settings.max_visual_tokens = 96;

  SessionConfig session_config = CreateSessionConfig(settings);

  EXPECT_EQ(session_config.GetMaxVisualTokens(), 96);
}

TEST(LiteRtLmSettingsUtilTest, CreateSessionConfigDisablesPruningByDefault) {
  LiteRtLmSettings settings;

  SessionConfig session_config = CreateSessionConfig(settings);

  EXPECT_EQ(session_config.GetMaxVisualTokens(), 0);
  EXPECT_EQ(session_config.GetVisualTokenPruningStrategy(), "uniform");
}

TEST(LiteRtLmSettingsUtilTest,
     CreateSessionConfigPropagatesVisualTokenPruningStrategy) {
  LiteRtLmSettings settings;
  settings.visual_token_pruning_strategy = "prompt_conditioned_v1";

  SessionConfig session_config = CreateSessionConfig(settings);

  EXPECT_EQ(session_config.GetVisualTokenPruningStrategy(),
            "prompt_conditioned_v1");
}

TEST(LiteRtLmSettingsUtilTest,
     CreateConversationOptionalArgsUsesRegexConstraintAndMaxOutputTokens) {
  LiteRtLmSettings settings;
  settings.constraint_regex = " ?[A-Za-z0-9]+(?: [A-Za-z0-9]+)?";
  settings.max_output_tokens = 2;

  auto optional_args = CreateConversationOptionalArgs(settings);

  ASSERT_TRUE(optional_args.decoding_constraint.has_value());
  ASSERT_TRUE(std::holds_alternative<LlGuidanceConstraintArg>(
      *optional_args.decoding_constraint));
  const auto& constraint_arg =
      std::get<LlGuidanceConstraintArg>(*optional_args.decoding_constraint);
  EXPECT_EQ(constraint_arg.constraint_type, LlgConstraintType::kRegex);
  EXPECT_EQ(constraint_arg.constraint_string, settings.constraint_regex);
  ASSERT_TRUE(optional_args.max_output_tokens.has_value());
  EXPECT_EQ(*optional_args.max_output_tokens, 2);
}

TEST(LiteRtLmSettingsUtilTest,
     CreateConversationConstraintProviderConfigUsesLlGuidanceForRegex) {
  LiteRtLmSettings settings;
  settings.constraint_regex = "yes|no";

  auto constraint_provider_config =
      CreateConversationConstraintProviderConfig(settings);

  ASSERT_TRUE(constraint_provider_config.has_value());
  EXPECT_TRUE(
      std::holds_alternative<LlGuidanceConfig>(*constraint_provider_config));
}

}  // namespace
}  // namespace lm
}  // namespace litert
