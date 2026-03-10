// Copyright 2026 The ODML Authors.
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

#include "runtime/engine/visual_token_budget_controller.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace litert::lm {
namespace {

TEST(VisualTokenBudgetControllerTest, LowersBudgetUnderQueuePressure) {
  const auto decision_or =
      ChooseVisualTokenBudget({96, 160, 256},
                              /*queue_depth=*/5,
                              /*queued_image_count=*/5,
                              /*recent_prefill_ms=*/std::nullopt,
                              /*recent_decode_ms=*/std::nullopt,
                              /*answer_mode=*/"none");
  ASSERT_TRUE(decision_or.ok()) << decision_or.status();
  const auto& decision = *decision_or;
  EXPECT_EQ(decision.budget, 160);
  EXPECT_NE(std::find(decision.reason_codes.begin(), decision.reason_codes.end(),
                      "queue_pressure"),
            decision.reason_codes.end());
}

TEST(VisualTokenBudgetControllerTest, MatchesDecodeWindow) {
  const auto decision_or =
      ChooseVisualTokenBudget({96, 160, 256},
                              /*queue_depth=*/0,
                              /*queued_image_count=*/1,
                              /*recent_prefill_ms=*/320.0,
                              /*recent_decode_ms=*/200.0,
                              /*answer_mode=*/"short");
  ASSERT_TRUE(decision_or.ok()) << decision_or.status();
  const auto& decision = *decision_or;
  EXPECT_EQ(decision.budget, 96);
  EXPECT_NE(std::find(decision.reason_codes.begin(), decision.reason_codes.end(),
                      "prefill_above_decode_window"),
            decision.reason_codes.end());
  EXPECT_NE(std::find(decision.reason_codes.begin(), decision.reason_codes.end(),
                      "short_answer_mode"),
            decision.reason_codes.end());
}

TEST(VisualTokenBudgetControllerTest, RejectsInvalidBuckets) {
  EXPECT_FALSE(ChooseVisualTokenBudget({0, 96},
                                       /*queue_depth=*/0,
                                       /*queued_image_count=*/0,
                                       /*recent_prefill_ms=*/std::nullopt,
                                       /*recent_decode_ms=*/std::nullopt)
                   .ok());
}

TEST(VisualTokenBudgetControllerTest, LimitsPerRequestStepChange) {
  const auto decision_or =
      ChooseVisualTokenBudget({32, 64, 128, 256},
                              /*queue_depth=*/8,
                              /*queued_image_count=*/8,
                              /*recent_prefill_ms=*/450.0,
                              /*recent_decode_ms=*/200.0,
                              /*answer_mode=*/"short",
                              /*previous_budget=*/256,
                              /*max_step_change=*/1);
  ASSERT_TRUE(decision_or.ok()) << decision_or.status();
  const auto& decision = *decision_or;
  EXPECT_EQ(decision.budget, 128);
  EXPECT_NE(std::find(decision.reason_codes.begin(), decision.reason_codes.end(),
                      "limited_step_change"),
            decision.reason_codes.end());
}

}  // namespace
}  // namespace litert::lm
