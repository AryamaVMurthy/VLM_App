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

#include "runtime/engine/overlap_scheduler.h"

#include <algorithm>

namespace litert::lm {

OverlapAnalysis AnalyzeOverlapWindow(const StageWindow& decode_window,
                                     const StageWindow& prefill_window) {
  OverlapAnalysis analysis;
  const absl::Time overlap_start =
      std::max(decode_window.start_time, prefill_window.start_time);
  const absl::Time overlap_end =
      std::min(decode_window.end_time, prefill_window.end_time);
  if (overlap_end > overlap_start) {
    analysis.overlap_duration = overlap_end - overlap_start;
  } else if (prefill_window.start_time >= decode_window.end_time) {
    analysis.no_overlap_reason =
        "prefill_started_after_previous_decode_finished";
  } else if (decode_window.start_time >= prefill_window.end_time) {
    analysis.no_overlap_reason =
        "previous_decode_started_after_prefill_finished";
  } else {
    analysis.no_overlap_reason = "overlap_window_empty";
  }

  if (decode_window.end_time > prefill_window.end_time) {
    analysis.cpu_stall_before_next_decode =
        decode_window.end_time - prefill_window.end_time;
  }
  return analysis;
}

}  // namespace litert::lm
