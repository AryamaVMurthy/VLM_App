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

#ifndef THIRD_PARTY_LITERT_LM_RUNTIME_ENGINE_REQUEST_PREPARATION_QUEUE_H_
#define THIRD_PARTY_LITERT_LM_RUNTIME_ENGINE_REQUEST_PREPARATION_QUEUE_H_

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"

namespace litert::lm {

struct RequestPreparationSpec {
  int request_index = -1;
  std::string request_id;
  std::string prompt;
  std::string image_path;
};

struct PreparedRequest {
  RequestPreparationSpec request;
  std::vector<InputData> request_contents;
  absl::Time prepare_start_time;
  absl::Time prepare_end_time;
};

class RequestPreparationQueue {
 public:
  using PrepareFn =
      std::function<absl::StatusOr<PreparedRequest>(const RequestPreparationSpec&)>;

  static absl::StatusOr<std::unique_ptr<RequestPreparationQueue>> Create(
      std::vector<RequestPreparationSpec> requests, int max_queue_size,
      PrepareFn prepare_fn);

  ~RequestPreparationQueue();

  RequestPreparationQueue(const RequestPreparationQueue&) = delete;
  RequestPreparationQueue& operator=(const RequestPreparationQueue&) = delete;

  absl::StatusOr<PreparedRequest> PopNext();
  int queue_size() const;

 private:
  RequestPreparationQueue(std::vector<RequestPreparationSpec> requests,
                          int max_queue_size, PrepareFn prepare_fn);

  bool CanProducerAdvance() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool CanConsumerAdvance() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void ProducerLoop();
  absl::Status StopAndJoin();

  const std::vector<RequestPreparationSpec> requests_;
  const int max_queue_size_;
  const PrepareFn prepare_fn_;

  mutable absl::Mutex mutex_;
  std::deque<PreparedRequest> queue_ ABSL_GUARDED_BY(mutex_);
  bool stop_requested_ ABSL_GUARDED_BY(mutex_) = false;
  bool producer_finished_ ABSL_GUARDED_BY(mutex_) = false;
  std::optional<absl::Status> producer_error_ ABSL_GUARDED_BY(mutex_);

  std::thread producer_thread_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_LITERT_LM_RUNTIME_ENGINE_REQUEST_PREPARATION_QUEUE_H_
