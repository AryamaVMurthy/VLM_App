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

#include "runtime/core/litert_env_options_util.h"

#include <filesystem>  // NOLINT
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr absl::string_view kTestModelPathContextLength =
    "magic_test_context_length.tflite";

absl::StatusOr<Model> LoadModelFromFile(absl::string_view model_path) {
  auto model_path_in_srcdir =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/" / std::string(model_path);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(model_path_in_srcdir.string()));
  return model;
}

class ModelResourcesMock : public ModelResources {
 public:
  MOCK_METHOD(absl::StatusOr<const proto::LlmMetadata*>, GetLlmMetadata, (),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Tokenizer>>, GetTokenizer, (),
              (override));
  MOCK_METHOD(absl::StatusOr<absl::string_view>, GetTFLiteModelBuffer,
              (ModelType model_type), (override));
  MOCK_METHOD(std::optional<std::string>, GetTFLiteModelBackendConstraint,
              (ModelType model_type), (override));
  MOCK_METHOD(absl::StatusOr<std::reference_wrapper<ScopedFile>>, GetScopedFile,
              (), (override));
  MOCK_METHOD((absl::StatusOr<std::pair<size_t, size_t>>),
              GetWeightsSectionOffset, (ModelType model_type), (override));

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    return &model_;
  }

  explicit ModelResourcesMock(const Model& model) : model_(model) {}

 private:
  const Model& model_;
};

absl::StatusOr<LlmExecutorSettings> GetLlmExecutorSettings() {
  auto model_assets_or = ModelAssets::Create("dont_care_path");
  if (!model_assets_or.ok()) {
    return model_assets_or.status();
  }
  return LlmExecutorSettings::CreateDefault(std::move(*model_assets_or));
}

std::string GetStringOptionValue(const Environment::Option& option) {
  if (std::holds_alternative<const char*>(option.value)) {
    return std::get<const char*>(option.value);
  }
  if (std::holds_alternative<absl::string_view>(option.value)) {
    return std::string(std::get<absl::string_view>(option.value));
  }
  ADD_FAILURE() << "Option did not contain a string payload.";
  return "";
}

TEST(LiteRtEnvOptionsUtilTest,
     PreservesRuntimeLibraryDirWhenMagicNumbersEnabled) {
  ASSERT_OK_AND_ASSIGN(auto model,
                       LoadModelFromFile(kTestModelPathContextLength));
  ASSERT_OK_AND_ASSIGN(auto executor_settings, GetLlmExecutorSettings());
  executor_settings.SetLitertDispatchLibDir("/tmp/runtime_libs");

  ModelResourcesMock model_resources(model);
  MagicNumberConfigsHelper helper;

  const auto env_options = BuildCpuGpuLiteRtEnvironmentOptions(
      model_resources, executor_settings, helper);

  ASSERT_EQ(env_options.size(), 2);
  EXPECT_EQ(env_options[0].tag, Environment::OptionTag::RuntimeLibraryDir);
  EXPECT_EQ(GetStringOptionValue(env_options[0]), "/tmp/runtime_libs");
  EXPECT_EQ(env_options[1].tag, Environment::OptionTag::MagicNumberConfigs);
}

TEST(LiteRtEnvOptionsUtilTest, RuntimeLibraryDirOnlyWhenMagicNumbersDisabled) {
  ASSERT_OK_AND_ASSIGN(auto model,
                       LoadModelFromFile(kTestModelPathContextLength));
  ASSERT_OK_AND_ASSIGN(auto executor_settings, GetLlmExecutorSettings());
  executor_settings.SetLitertDispatchLibDir("/tmp/runtime_libs");
  AdvancedSettings advanced_settings{.configure_magic_numbers = false};
  executor_settings.SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(model);
  MagicNumberConfigsHelper helper;

  const auto env_options = BuildCpuGpuLiteRtEnvironmentOptions(
      model_resources, executor_settings, helper);

  ASSERT_EQ(env_options.size(), 1);
  EXPECT_EQ(env_options[0].tag, Environment::OptionTag::RuntimeLibraryDir);
  EXPECT_EQ(GetStringOptionValue(env_options[0]), "/tmp/runtime_libs");
}

}  // namespace
}  // namespace litert::lm
