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

#include "runtime/engine/graphpilot_retrieval_lib.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_macros.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

TEST(GraphPilotRetrievalLibTest, BuildModelInputAddsBosEosAndPads) {
  auto ids = BuildModelInputTokenIds({11, 12, 13}, /*sequence_length=*/8,
                                     /*bos_id=*/2, /*eos_id=*/1,
                                     /*pad_id=*/0);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_THAT(*ids, ElementsAre(2, 11, 12, 13, 1, 0, 0, 0));
}

TEST(GraphPilotRetrievalLibTest, BuildModelInputTruncatesToSequenceLength) {
  auto ids = BuildModelInputTokenIds({10, 11, 12, 13, 14}, /*sequence_length=*/4,
                                     /*bos_id=*/2, /*eos_id=*/1,
                                     /*pad_id=*/0);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_THAT(*ids, ElementsAre(2, 10, 11, 1));
}

TEST(GraphPilotRetrievalLibTest, NormalizeRejectsZeroVectors) {
  std::vector<float> embedding = {0.0f, 0.0f, 0.0f};
  auto status = NormalizeEmbeddingInPlace(embedding);
  EXPECT_FALSE(status.ok());
}

TEST(GraphPilotRetrievalLibTest, RankDocumentsSortsByCosineSimilarity) {
  std::vector<RetrievalDocument> docs = {
      {.doc_id = "aligned",
       .title = "Aligned",
       .text = "Aligned doc",
       .embedding = {1.0f, 0.0f}},
      {.doc_id = "orthogonal",
       .title = "Orthogonal",
       .text = "Orthogonal doc",
       .embedding = {0.0f, 1.0f}},
      {.doc_id = "diagonal",
       .title = "Diagonal",
       .text = "Diagonal doc",
       .embedding = {0.8f, 0.2f}},
  };
  ASSERT_TRUE(NormalizeDocumentEmbeddingsInPlace(docs).ok());

  auto hits = RankDocumentsBySimilarity({1.0f, 0.0f}, docs, /*top_k=*/2);
  ASSERT_TRUE(hits.ok()) << hits.status();
  ASSERT_EQ(hits->size(), 2);
  EXPECT_EQ((*hits)[0].doc_id, "aligned");
  EXPECT_EQ((*hits)[1].doc_id, "diagonal");
  EXPECT_GT((*hits)[0].score, (*hits)[1].score);
}

TEST(GraphPilotRetrievalLibTest,
     BuildRetrievalEnvironmentOptionsDoesNotRequireRuntimeDirForGpu) {
  auto env_options = BuildRetrievalEnvironmentOptions(
      "gpu", RetrievalEnvironmentConfig{});
  ASSERT_TRUE(env_options.ok()) << env_options.status();
  EXPECT_TRUE(env_options->empty());
}

TEST(GraphPilotRetrievalLibTest,
     BuildRetrievalEnvironmentOptionsPreservesRuntimeDirForGpu) {
  auto env_options = BuildRetrievalEnvironmentOptions(
      "gpu", RetrievalEnvironmentConfig{.runtime_library_dir = "/tmp/runtime"});
  ASSERT_TRUE(env_options.ok()) << env_options.status();
  ASSERT_EQ(env_options->size(), 1);
  EXPECT_EQ((*env_options)[0].tag, Environment::OptionTag::RuntimeLibraryDir);
}

TEST(GraphPilotRetrievalLibTest,
     BuildRetrievalEnvironmentOptionsRequiresDispatchDirForNpu) {
  auto env_options = BuildRetrievalEnvironmentOptions(
      "npu", RetrievalEnvironmentConfig{.runtime_library_dir = "/tmp/runtime"});
  ASSERT_FALSE(env_options.ok());
  EXPECT_THAT(std::string(env_options.status().message()),
              testing::HasSubstr("dispatch_library_dir"));
}

TEST(GraphPilotRetrievalLibTest,
     BuildRetrievalEnvironmentOptionsAddsRuntimeAndDispatchDirsForNpu) {
  auto env_options_or = BuildRetrievalEnvironmentOptions(
      "npu", RetrievalEnvironmentConfig{.runtime_library_dir = "/tmp/runtime",
                                        .dispatch_library_dir = "/tmp/dispatch"});
  ASSERT_TRUE(env_options_or.ok()) << env_options_or.status();
  const auto& env_options = *env_options_or;
  ASSERT_EQ(env_options.size(), 3);
  EXPECT_EQ(env_options[0].tag, Environment::OptionTag::RuntimeLibraryDir);
  EXPECT_EQ(env_options[1].tag, Environment::OptionTag::DispatchLibraryDir);
  EXPECT_EQ(env_options[2].tag,
            Environment::OptionTag::CompilerPluginLibraryDir);
}

TEST(GraphPilotRetrievalLibTest,
     BuildRetrievalCompilationOptionsConfiguresGpuPrecisionAndStorageHints) {
  auto options_or = BuildRetrievalCompilationOptions("gpu");
  ASSERT_TRUE(options_or.ok()) << options_or.status();

  LiteRtHwAcceleratorSet accelerators = 0;
  LITERT_ASSERT_OK(
      LiteRtGetOptionsHardwareAccelerators(options_or->Get(), &accelerators));
  EXPECT_EQ(accelerators, static_cast<LiteRtHwAcceleratorSet>(
                              HwAccelerators::kGpu));

  LITERT_ASSERT_OK_AND_ASSIGN(auto& gpu_options, options_or->GetGpuOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              gpu_options.GetData<LiteRtGpuOptionsPayloadT>());

  bool constant_tensor_sharing = false;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_TRUE(constant_tensor_sharing);

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_EQ(precision, kLiteRtDelegatePrecisionFp32);

  bool prefer_texture_weights = false;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
          &prefer_texture_weights, payload));
  EXPECT_TRUE(prefer_texture_weights);
}

}  // namespace
}  // namespace litert::lm
