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

#include "runtime/engine/request_preparation_queue.h"

#include <exception>
#include <memory>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::lm {

absl::StatusOr<std::unique_ptr<RequestPreparationQueue>>
RequestPreparationQueue::Create(std::vector<RequestPreparationSpec> requests,
                                int max_queue_size, PrepareFn prepare_fn) {
  if (requests.empty()) {
    return absl::InvalidArgumentError(
        "RequestPreparationQueue requires at least one request.");
  }
  if (max_queue_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RequestPreparationQueue max_queue_size must be > 0, got ",
        max_queue_size));
  }
  if (!prepare_fn) {
    return absl::InvalidArgumentError(
        "RequestPreparationQueue requires a valid prepare callback.");
  }
  auto queue = std::unique_ptr<RequestPreparationQueue>(
      new RequestPreparationQueue(std::move(requests), max_queue_size,
                                  std::move(prepare_fn)));
  queue->producer_thread_ =
      std::thread([queue_ptr = queue.get()]() { queue_ptr->ProducerLoop(); });
  return queue;
}

RequestPreparationQueue::RequestPreparationQueue(
    std::vector<RequestPreparationSpec> requests, int max_queue_size,
    PrepareFn prepare_fn)
    : requests_(std::move(requests)),
      max_queue_size_(max_queue_size),
      prepare_fn_(std::move(prepare_fn)) {}

RequestPreparationQueue::~RequestPreparationQueue() {
  const absl::Status stop_status = StopAndJoin();
  if (!stop_status.ok()) {
    ABSL_LOG(ERROR) << "RequestPreparationQueue shutdown failed: "
                    << stop_status;
  }
}

absl::StatusOr<PreparedRequest> RequestPreparationQueue::PopNext() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &RequestPreparationQueue::CanConsumerAdvance));
  if (!queue_.empty()) {
    PreparedRequest prepared = std::move(queue_.front());
    queue_.pop_front();
    return prepared;
  }
  if (producer_error_.has_value()) {
    return *producer_error_;
  }
  if (producer_finished_) {
    return absl::OutOfRangeError(
        "RequestPreparationQueue has no more prepared requests.");
  }
  return absl::InternalError(
      "RequestPreparationQueue reached an invalid consumer state.");
}

int RequestPreparationQueue::queue_size() const {
  absl::MutexLock lock(&mutex_);
  return static_cast<int>(queue_.size());
}

bool RequestPreparationQueue::CanProducerAdvance() const {
  mutex_.AssertHeld();
  return stop_requested_ || static_cast<int>(queue_.size()) < max_queue_size_;
}

bool RequestPreparationQueue::CanConsumerAdvance() const {
  mutex_.AssertHeld();
  return !queue_.empty() || producer_finished_ || producer_error_.has_value();
}

void RequestPreparationQueue::ProducerLoop() {
  for (const RequestPreparationSpec& request : requests_) {
    {
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &RequestPreparationQueue::CanProducerAdvance));
      if (stop_requested_) {
        producer_finished_ = true;
        return;
      }
    }

    absl::StatusOr<PreparedRequest> prepared;
    try {
      prepared = prepare_fn_(request);
    } catch (const std::exception& e) {
      prepared = absl::InternalError(
          absl::StrCat("Request preparation threw std::exception: ", e.what()));
    } catch (...) {
      prepared = absl::InternalError(
          "Request preparation threw an unknown non-absl exception.");
    }

    absl::MutexLock lock(&mutex_);
    if (stop_requested_) {
      producer_finished_ = true;
      return;
    }
    if (!prepared.ok()) {
      producer_error_ = prepared.status();
      producer_finished_ = true;
      return;
    }
    queue_.push_back(std::move(*prepared));
  }

  absl::MutexLock lock(&mutex_);
  producer_finished_ = true;
}

absl::Status RequestPreparationQueue::StopAndJoin() {
  {
    absl::MutexLock lock(&mutex_);
    stop_requested_ = true;
  }
  if (producer_thread_.joinable()) {
    producer_thread_.join();
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
