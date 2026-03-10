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
#include <cstdlib>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::lm {
namespace {

absl::StatusOr<std::vector<int>> NormalizeBuckets(
    absl::Span<const int> allowed_buckets) {
  std::vector<int> buckets(allowed_buckets.begin(), allowed_buckets.end());
  std::sort(buckets.begin(), buckets.end());
  buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
  if (buckets.empty()) {
    return absl::InvalidArgumentError(
        "At least one visual token budget bucket is required.");
  }
  if (buckets.front() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Visual token budget buckets must be positive, got ",
                     buckets.front(), "."));
  }
  return buckets;
}

}  // namespace

absl::StatusOr<VisualTokenBudgetDecision> ChooseVisualTokenBudget(
    absl::Span<const int> allowed_buckets, int queue_depth,
    int queued_image_count, std::optional<double> recent_prefill_ms,
    std::optional<double> recent_decode_ms, absl::string_view answer_mode,
    std::optional<int> previous_budget, int max_step_change) {
  auto buckets_or = NormalizeBuckets(allowed_buckets);
  if (!buckets_or.ok()) {
    return buckets_or.status();
  }
  std::vector<int> buckets = std::move(*buckets_or);
  if (queue_depth < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("queue_depth must be non-negative, got ", queue_depth,
                     "."));
  }
  if (queued_image_count < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "queued_image_count must be non-negative, got ", queued_image_count,
        "."));
  }
  if (max_step_change < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_step_change must be non-negative, got ",
                     max_step_change, "."));
  }
  std::string normalized_answer_mode(answer_mode);
  absl::AsciiStrToLower(&normalized_answer_mode);
  if (normalized_answer_mode != "none" &&
      normalized_answer_mode != "short" &&
      normalized_answer_mode != "long") {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported answer_mode: ", answer_mode));
  }

  int bucket_index = static_cast<int>(buckets.size()) - 1;
  std::vector<std::string> reason_codes;

  if (queue_depth >= 4 || queued_image_count >= 4) {
    bucket_index = std::max(0, bucket_index - 1);
    reason_codes.push_back("queue_pressure");
  }
  if (queue_depth >= 8 || queued_image_count >= 8) {
    bucket_index = 0;
    reason_codes.push_back("heavy_queue_pressure");
  }

  if (recent_prefill_ms.has_value() && recent_decode_ms.has_value()) {
    if (*recent_prefill_ms <= 0.0 || *recent_decode_ms <= 0.0) {
      return absl::InvalidArgumentError(
          "recent_prefill_ms and recent_decode_ms must be positive when "
          "provided.");
    }
    if (*recent_prefill_ms > *recent_decode_ms * 1.10) {
      bucket_index = std::max(0, bucket_index - 1);
      reason_codes.push_back("prefill_above_decode_window");
    } else if (*recent_prefill_ms < *recent_decode_ms * 0.75) {
      bucket_index =
          std::min(static_cast<int>(buckets.size()) - 1, bucket_index + 1);
      reason_codes.push_back("prefill_below_decode_window");
    }
  }

  if (normalized_answer_mode == "short" && bucket_index > 0) {
    bucket_index = std::max(0, bucket_index - 1);
    reason_codes.push_back("short_answer_mode");
  } else if (normalized_answer_mode == "long" &&
             bucket_index < static_cast<int>(buckets.size()) - 1) {
    bucket_index =
        std::min(static_cast<int>(buckets.size()) - 1, bucket_index + 1);
    reason_codes.push_back("long_answer_mode");
  }

  if (previous_budget.has_value()) {
    const auto previous_it =
        std::find(buckets.begin(), buckets.end(), *previous_budget);
    if (previous_it == buckets.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("previous_budget must be one of the allowed buckets, got ",
                       *previous_budget, "."));
    }
    const int previous_index =
        static_cast<int>(std::distance(buckets.begin(), previous_it));
    const int delta = bucket_index - previous_index;
    if (std::abs(delta) > max_step_change) {
      bucket_index = previous_index +
                     (delta > 0 ? max_step_change : -max_step_change);
      reason_codes.push_back("limited_step_change");
    }
  }

  if (reason_codes.empty()) {
    reason_codes.push_back("baseline_bucket");
  }
  return VisualTokenBudgetDecision{
      .budget = buckets[bucket_index],
      .reason_codes = std::move(reason_codes),
  };
}

}  // namespace litert::lm
