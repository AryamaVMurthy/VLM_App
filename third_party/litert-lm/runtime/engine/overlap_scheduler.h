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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_OVERLAP_SCHEDULER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_OVERLAP_SCHEDULER_H_

#include <optional>
#include <string>

#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

struct StageWindow {
  absl::Time start_time;
  absl::Time end_time;
};

struct OverlapAnalysis {
  absl::Duration overlap_duration = absl::ZeroDuration();
  absl::Duration cpu_stall_before_next_decode = absl::ZeroDuration();
  std::optional<std::string> no_overlap_reason;
};

// Computes the overlap and stall characteristics between the previous request's
// CPU decode window and the next request's NPU prefill window.
OverlapAnalysis AnalyzeOverlapWindow(const StageWindow& decode_window,
                                     const StageWindow& prefill_window);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_OVERLAP_SCHEDULER_H_
