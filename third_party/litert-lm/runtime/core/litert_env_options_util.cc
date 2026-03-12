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

#include <vector>

namespace litert::lm {

std::vector<Environment::Option> BuildCpuGpuLiteRtEnvironmentOptions(
    ModelResources& model_resources,
    const LlmExecutorSettings& executor_settings,
    MagicNumberConfigsHelper& helper) {
  std::vector<Environment::Option> env_options;
  if (!executor_settings.GetLitertDispatchLibDir().empty()) {
    env_options.push_back(::litert::Environment::Option{
        ::litert::Environment::OptionTag::RuntimeLibraryDir,
        executor_settings.GetLitertDispatchLibDir()});
  }
  if (!executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->configure_magic_numbers) {
    auto magic_number_options =
        helper.GetLiteRtEnvOptions(model_resources, executor_settings);
    env_options.insert(env_options.end(), magic_number_options.begin(),
                       magic_number_options.end());
  }
  return env_options;
}

}  // namespace litert::lm
