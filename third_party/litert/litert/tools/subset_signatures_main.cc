// Copyright 2025 Google LLC.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"

ABSL_FLAG(std::string, model_path, "", "Path to the input TFLite model.");
ABSL_FLAG(std::string, output_path, "", "Path to the output TFLite model.");
ABSL_FLAG(std::vector<std::string>, signature_keys, {},
          "Signature keys to keep, in output order.");

namespace {

using ::litert::Expected;
using ::litert::Unexpected;

Expected<size_t> GetSubgraphIndex(const LiteRtModelT& model,
                                  const LiteRtSubgraphT& subgraph) {
  for (size_t i = 0; i < model.NumSubgraphs(); ++i) {
    if (&model.Subgraph(i) == &subgraph) {
      return i;
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Subgraph not found in model");
}

Expected<LiteRtTensor> FindTensorByName(LiteRtSubgraphT& subgraph,
                                        absl::string_view tensor_name) {
  for (auto tensor : subgraph.Tensors()) {
    if (tensor->Name() == tensor_name) {
      return tensor;
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound,
                    "Tensor not found in selected subgraph");
}

Expected<void> CopySignature(const LiteRtSignatureT& source_signature,
                             LiteRtSubgraphT& dest_subgraph,
                             LiteRtModelT& dest_model) {
  std::vector<std::string> input_names(source_signature.InputNames().begin(),
                                       source_signature.InputNames().end());
  std::vector<LiteRtTensor> input_tensors;
  input_tensors.reserve(input_names.size());
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto* source_tensor = source_signature.GetInputTensor(i);
    if (source_tensor->Name().empty()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Source signature input tensor name is empty");
    }
    LITERT_ASSIGN_OR_RETURN(auto tensor,
                            FindTensorByName(dest_subgraph,
                                             source_tensor->Name()));
    input_tensors.push_back(tensor);
  }

  std::vector<std::string> output_names(source_signature.OutputNames().begin(),
                                        source_signature.OutputNames().end());
  std::vector<LiteRtTensor> output_tensors;
  output_tensors.reserve(output_names.size());
  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto* source_tensor = source_signature.GetOutputTensor(i);
    if (source_tensor->Name().empty()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Source signature output tensor name is empty");
    }
    LITERT_ASSIGN_OR_RETURN(auto tensor,
                            FindTensorByName(dest_subgraph,
                                             source_tensor->Name()));
    output_tensors.push_back(tensor);
  }

  dest_model.EmplaceSignature(&dest_subgraph, std::move(input_names),
                              std::move(input_tensors),
                              std::move(output_names),
                              std::move(output_tensors),
                              std::string(source_signature.Key()));
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string output_path = absl::GetFlag(FLAGS_output_path);
  const std::vector<std::string> signature_keys =
      absl::GetFlag(FLAGS_signature_keys);

  if (model_path.empty()) {
    std::cerr << "--model_path is required\n";
    return 2;
  }
  if (output_path.empty()) {
    std::cerr << "--output_path is required\n";
    return 2;
  }
  if (signature_keys.empty()) {
    std::cerr << "--signature_keys must contain at least one signature\n";
    return 2;
  }

  auto model = litert::internal::LoadModelFromFile(model_path);
  if (!model.HasValue()) {
    std::cerr << "Failed to load model: " << model.Error().Message() << "\n";
    return 1;
  }

  std::vector<size_t> kept_subgraph_indices;
  kept_subgraph_indices.reserve(signature_keys.size());
  std::unordered_map<std::string, size_t> signature_to_subgraph_index;
  for (const auto& signature_key : signature_keys) {
    auto signature = (*model)->FindSignature(signature_key);
    if (!signature.HasValue()) {
      std::cerr << "Missing signature: " << signature_key << "\n";
      return 1;
    }
    auto subgraph_index =
        GetSubgraphIndex(**model, signature->get().GetSubgraph());
    if (!subgraph_index.HasValue()) {
      std::cerr << "Failed to resolve subgraph for signature: "
                << signature_key << "\n";
      return 1;
    }
    signature_to_subgraph_index.emplace(signature_key, *subgraph_index);
    if (std::find(kept_subgraph_indices.begin(), kept_subgraph_indices.end(),
                  *subgraph_index) == kept_subgraph_indices.end()) {
      kept_subgraph_indices.push_back(*subgraph_index);
    }
  }

  LiteRtModelT reduced_model = (*model)->Yank(kept_subgraph_indices);
  std::unordered_map<size_t, size_t> old_to_new_subgraph_index;
  for (size_t new_index = 0; new_index < kept_subgraph_indices.size();
       ++new_index) {
    old_to_new_subgraph_index.emplace(kept_subgraph_indices[new_index],
                                      new_index);
  }

  for (const auto& signature_key : signature_keys) {
    auto source_signature = (*model)->FindSignature(signature_key);
    ABSL_CHECK(source_signature.HasValue());
    const size_t old_subgraph_index =
        signature_to_subgraph_index.at(signature_key);
    auto new_index_it = old_to_new_subgraph_index.find(old_subgraph_index);
    ABSL_CHECK(new_index_it != old_to_new_subgraph_index.end());
    auto& dest_subgraph = reduced_model.Subgraph(new_index_it->second);
    auto status =
        CopySignature(source_signature->get(), dest_subgraph, reduced_model);
    if (!status.HasValue()) {
      std::cerr << "Failed to copy signature " << signature_key << ": "
                << status.Error().Message() << "\n";
      return 1;
    }
  }

  auto serialized = litert::internal::SerializeModel(std::move(reduced_model));
  if (!serialized.HasValue()) {
    std::cerr << "Failed to serialize reduced model: "
              << serialized.Error().Message() << "\n";
    return 1;
  }

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    std::cerr << "Failed to open output path: " << output_path << "\n";
    return 1;
  }
  output.write(reinterpret_cast<const char*>(serialized->Data()),
               serialized->Size());
  output.close();
  if (!output) {
    std::cerr << "Failed to write output model: " << output_path << "\n";
    return 1;
  }

  std::cout << "Wrote reduced model to " << output_path << "\n";
  std::cout << "Kept signatures: [" << absl::StrJoin(signature_keys, ", ")
            << "]\n";
  std::cout << "Kept subgraphs: [" << absl::StrJoin(kept_subgraph_indices, ", ")
            << "]\n";
  return 0;
}
