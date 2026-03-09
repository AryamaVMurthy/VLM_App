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

#include <chrono>  // NOLINT
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {
namespace {

std::vector<RequestPreparationSpec> MakeRequests() {
  return {
      {.request_index = 0,
       .request_id = "q0",
       .prompt = "Question: q0",
       .image_path = "/tmp/q0.jpg"},
      {.request_index = 1,
       .request_id = "q1",
       .prompt = "Question: q1",
       .image_path = "/tmp/q1.jpg"},
      {.request_index = 2,
       .request_id = "q2",
       .prompt = "Question: q2",
       .image_path = "/tmp/q2.jpg"},
  };
}

PreparedRequest MakePreparedRequest(const RequestPreparationSpec& request) {
  PreparedRequest prepared;
  prepared.request = request;
  prepared.request_contents.emplace_back(InputText(std::string(request.prompt)));
  return prepared;
}

TEST(RequestPreparationQueueTest, PopsPreparedRequestsInManifestOrder) {
  auto queue_or = RequestPreparationQueue::Create(
      MakeRequests(), /*max_queue_size=*/2,
      [](const RequestPreparationSpec& request)
          -> absl::StatusOr<PreparedRequest> {
        return MakePreparedRequest(request);
      });
  ASSERT_TRUE(queue_or.ok()) << queue_or.status();
  auto queue = std::move(*queue_or);

  std::vector<std::string> request_ids;
  for (int i = 0; i < 3; ++i) {
    absl::StatusOr<PreparedRequest> prepared = queue->PopNext();
    ASSERT_TRUE(prepared.ok()) << prepared.status();
    request_ids.push_back(prepared->request.request_id);
  }

  EXPECT_EQ(request_ids[0], "q0");
  EXPECT_EQ(request_ids[1], "q1");
  EXPECT_EQ(request_ids[2], "q2");
}

TEST(RequestPreparationQueueTest, SurfacesProducerFailuresToConsumer) {
  auto queue_or = RequestPreparationQueue::Create(
      MakeRequests(), /*max_queue_size=*/1,
      [](const RequestPreparationSpec& request)
          -> absl::StatusOr<PreparedRequest> {
        if (request.request_id == "q1") {
          return absl::InternalError("simulated prepare failure");
        }
        return MakePreparedRequest(request);
      });
  ASSERT_TRUE(queue_or.ok()) << queue_or.status();
  auto queue = std::move(*queue_or);

  absl::StatusOr<PreparedRequest> first = queue->PopNext();
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->request.request_id, "q0");

  absl::StatusOr<PreparedRequest> second = queue->PopNext();
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().code(), absl::StatusCode::kInternal);
  EXPECT_NE(second.status().message().find("simulated prepare failure"),
            std::string::npos);
}

TEST(RequestPreparationQueueTest, BoundedQueueStillDeliversAllRequests) {
  auto queue_or = RequestPreparationQueue::Create(
      MakeRequests(), /*max_queue_size=*/1,
      [](const RequestPreparationSpec& request)
          -> absl::StatusOr<PreparedRequest> {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return MakePreparedRequest(request);
      });
  ASSERT_TRUE(queue_or.ok()) << queue_or.status();
  auto queue = std::move(*queue_or);

  std::vector<std::string> request_ids;
  for (int i = 0; i < 3; ++i) {
    absl::StatusOr<PreparedRequest> prepared = queue->PopNext();
    ASSERT_TRUE(prepared.ok()) << prepared.status();
    request_ids.push_back(prepared->request.request_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_EQ(request_ids.size(), 3);
  EXPECT_EQ(request_ids[0], "q0");
  EXPECT_EQ(request_ids[1], "q1");
  EXPECT_EQ(request_ids[2], "q2");
}

}  // namespace
}  // namespace litert::lm
