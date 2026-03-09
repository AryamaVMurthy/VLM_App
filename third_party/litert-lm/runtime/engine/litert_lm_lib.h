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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_

#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/log/log_entry.h"  // from @com_google_absl
#include "absl/log/log_sink.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/engine/litert_lm_settings.h"

namespace litert {
namespace lm {

class FileLogSink : public absl::LogSink {
 public:
  explicit FileLogSink(const std::string& filename) {
    std::filesystem::path path(filename);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    file_.open(filename, std::ios_base::app);
  }

  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(&mutex_);
    file_ << entry.text_message_with_prefix_and_newline();
  }

 private:
  absl::Mutex mutex_;
  std::ofstream file_;
};

struct LitertLmMetrics {
  std::optional<BenchmarkInfo> benchmark_info;
  float peak_mem_mb = 0.0f;
  float peak_private_mb = 0.0f;
};

// Runs the LLM inference with the given settings.
// If metrics is not null, the metrics will be populated with the metrics from
// the inference. Results from each iteration is saved in the vector.
absl::Status RunLiteRtLm(const LiteRtLmSettings& settings,
                         std::vector<LitertLmMetrics>* metrics = nullptr);

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
