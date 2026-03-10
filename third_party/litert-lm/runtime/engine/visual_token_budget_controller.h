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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_VISUAL_TOKEN_BUDGET_CONTROLLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_VISUAL_TOKEN_BUDGET_CONTROLLER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

struct VisualTokenBudgetDecision {
  int budget = 0;
  std::vector<std::string> reason_codes;
};

absl::StatusOr<VisualTokenBudgetDecision> ChooseVisualTokenBudget(
    absl::Span<const int> allowed_buckets, int queue_depth,
    int queued_image_count, std::optional<double> recent_prefill_ms,
    std::optional<double> recent_decode_ms,
    absl::string_view answer_mode = "none",
    std::optional<int> previous_budget = std::nullopt,
    int max_step_change = 1);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_VISUAL_TOKEN_BUDGET_CONTROLLER_H_
