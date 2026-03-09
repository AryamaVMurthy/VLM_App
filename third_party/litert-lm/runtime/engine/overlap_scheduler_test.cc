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

#include <gtest/gtest.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

TEST(OverlapSchedulerTest, ReportsOverlapWithoutStall) {
  const absl::Time base = absl::Now();
  const StageWindow decode_window{
      .start_time = base + absl::Milliseconds(10),
      .end_time = base + absl::Milliseconds(50),
  };
  const StageWindow prefill_window{
      .start_time = base + absl::Milliseconds(20),
      .end_time = base + absl::Milliseconds(60),
  };

  const OverlapAnalysis analysis =
      AnalyzeOverlapWindow(decode_window, prefill_window);

  EXPECT_EQ(analysis.overlap_duration, absl::Milliseconds(30));
  EXPECT_EQ(analysis.cpu_stall_before_next_decode, absl::ZeroDuration());
  EXPECT_EQ(analysis.no_overlap_reason, std::nullopt);
}

TEST(OverlapSchedulerTest, ReportsCpuBusyStallWhenPrefillFinishesFirst) {
  const absl::Time base = absl::Now();
  const StageWindow decode_window{
      .start_time = base + absl::Milliseconds(10),
      .end_time = base + absl::Milliseconds(80),
  };
  const StageWindow prefill_window{
      .start_time = base + absl::Milliseconds(20),
      .end_time = base + absl::Milliseconds(40),
  };

  const OverlapAnalysis analysis =
      AnalyzeOverlapWindow(decode_window, prefill_window);

  EXPECT_EQ(analysis.overlap_duration, absl::Milliseconds(20));
  EXPECT_EQ(analysis.cpu_stall_before_next_decode, absl::Milliseconds(40));
  EXPECT_EQ(analysis.no_overlap_reason, std::nullopt);
}

TEST(OverlapSchedulerTest, ReportsNoOverlapWhenPrefillStartsAfterDecodeEnds) {
  const absl::Time base = absl::Now();
  const StageWindow decode_window{
      .start_time = base + absl::Milliseconds(10),
      .end_time = base + absl::Milliseconds(20),
  };
  const StageWindow prefill_window{
      .start_time = base + absl::Milliseconds(30),
      .end_time = base + absl::Milliseconds(50),
  };

  const OverlapAnalysis analysis =
      AnalyzeOverlapWindow(decode_window, prefill_window);

  EXPECT_EQ(analysis.overlap_duration, absl::ZeroDuration());
  EXPECT_EQ(analysis.cpu_stall_before_next_decode, absl::ZeroDuration());
  ASSERT_TRUE(analysis.no_overlap_reason.has_value());
  EXPECT_EQ(*analysis.no_overlap_reason,
            "prefill_started_after_previous_decode_finished");
}

}  // namespace
}  // namespace litert::lm
