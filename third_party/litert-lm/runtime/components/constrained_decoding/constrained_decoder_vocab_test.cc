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

#include "runtime/components/constrained_decoding/constrained_decoder.h"

#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

TEST(ConstrainedDecoderVocabTest, MaskLogitsMasksExtraModelVocabularyEntries) {
  FakeConstraint constraint(/*token_ids=*/{2, 1}, /*vocabulary_size=*/4);
  ConstrainedDecoder constrained_decoder(&constraint, /*batch_size=*/1);

  std::vector<float> logits_data = {1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 6.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits_tensor_buffer,
      CopyToTensorBuffer<float>(absl::MakeConstSpan(logits_data),
                                /*dims=*/{1, 1, 6}));

  EXPECT_TRUE(constrained_decoder.MaskLogits(logits_tensor_buffer).ok());

  LITERT_ASSERT_OK_AND_ASSIGN(auto masked_logits,
                              CopyFromTensorBuffer<float>(logits_tensor_buffer));
  for (int i = 0; i < masked_logits.size(); ++i) {
    if (i == 2) {
      EXPECT_EQ(masked_logits[i], 1.0f);
    } else {
      EXPECT_EQ(masked_logits[i], std::numeric_limits<float>::lowest());
    }
  }
}

}  // namespace
}  // namespace litert::lm
