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

#include "runtime/executor/vision_executor_utils.h"

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<VisionExecutorProperties>
GetVisionExecutorPropertiesFromModelResources(ModelResources& model_resources) {
  VisionExecutorProperties properties;
  ASSIGN_OR_RETURN(
      auto vision_adapter_model,
      model_resources.GetTFLiteModel(ModelType::kTfLiteVisionAdapter));
  LITERT_ASSIGN_OR_RETURN(auto input_tensor_type,
                          vision_adapter_model->GetOutputTensorType(0, 0));
  // Assume the adapter output tensor has shape [batch_size, num_tokens,
  // embedding_size].
  properties.num_tokens_per_image =
      input_tensor_type.Layout()
          .Dimensions()[input_tensor_type.Layout().Dimensions().size() - 2];
  return properties;
}

}  // namespace litert::lm
