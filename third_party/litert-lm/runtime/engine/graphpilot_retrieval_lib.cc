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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

using ::nlohmann::json;

absl::StatusOr<std::string> ReadTextFile(absl::string_view path) {
  std::ifstream input{std::string(path)};
  if (!input.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open file: ", path));
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<litert::HwAccelerators> ParseAccelerator(
    absl::string_view accelerator) {
  std::string normalized(accelerator);
  absl::AsciiStrToLower(&normalized);
  if (normalized == "cpu") {
    return litert::HwAccelerators::kCpu;
  }
  if (normalized == "gpu") {
    return litert::HwAccelerators::kGpu;
  }
  if (normalized == "npu") {
    return litert::HwAccelerators::kNpu;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported accelerator: ", accelerator));
}

absl::StatusOr<int> ResolveSequenceLength(
    const litert::CompiledModel& compiled_model) {
  LITERT_ASSIGN_OR_RETURN(auto input_buffers, compiled_model.CreateInputBuffers(0));
  if (input_buffers.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected one retrieval input tensor, got ", input_buffers.size(),
        "."));
  }
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, input_buffers[0].TensorType());
  const auto& dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected a 2D retrieval input tensor, got rank ",
        dimensions.size(), "."));
  }
  if (dimensions[0] != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected retrieval batch dimension 1, got ", dimensions[0], "."));
  }
  if (dimensions[1] <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected positive retrieval sequence length, got ", dimensions[1],
        "."));
  }
  return dimensions[1];
}

}  // namespace

absl::StatusOr<std::vector<int32_t>> BuildModelInputTokenIds(
    absl::Span<const int> token_ids, int sequence_length, int bos_id, int eos_id,
    int pad_id) {
  if (sequence_length <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("sequence_length must be positive, got ",
                     sequence_length, "."));
  }
  const int effective_pad_id = pad_id >= 0 ? pad_id : 0;
  std::vector<int32_t> output;
  output.reserve(sequence_length);
  if (bos_id >= 0) {
    output.push_back(bos_id);
  }
  for (const int token_id : token_ids) {
    if (static_cast<int>(output.size()) >= sequence_length) {
      break;
    }
    output.push_back(token_id);
  }
  if (eos_id >= 0) {
    if (static_cast<int>(output.size()) == sequence_length) {
      output.back() = eos_id;
    } else {
      output.push_back(eos_id);
    }
  }
  if (output.empty()) {
    return absl::InvalidArgumentError(
        "Tokenization produced no usable tokens for the retrieval model.");
  }
  if (static_cast<int>(output.size()) < sequence_length) {
    output.resize(sequence_length, effective_pad_id);
  } else if (static_cast<int>(output.size()) > sequence_length) {
    output.resize(sequence_length);
  }
  return output;
}

absl::Status NormalizeEmbeddingInPlace(std::vector<float>& embedding) {
  if (embedding.empty()) {
    return absl::InvalidArgumentError("Embedding cannot be empty.");
  }
  double squared_norm = 0.0;
  for (const float value : embedding) {
    squared_norm += static_cast<double>(value) * static_cast<double>(value);
  }
  if (squared_norm <= 0.0) {
    return absl::InvalidArgumentError(
        "Embedding norm must be positive; got a zero vector.");
  }
  const double norm = std::sqrt(squared_norm);
  for (float& value : embedding) {
    value = static_cast<float>(value / norm);
  }
  return absl::OkStatus();
}

absl::Status NormalizeDocumentEmbeddingsInPlace(
    std::vector<RetrievalDocument>& documents) {
  for (RetrievalDocument& document : documents) {
    RETURN_IF_ERROR(NormalizeEmbeddingInPlace(document.embedding));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<RetrievalDocument>> LoadRetrievalDocumentsFromJsonFile(
    absl::string_view kb_path) {
  ASSIGN_OR_RETURN(std::string raw_json, ReadTextFile(kb_path));
  json payload;
  try {
    payload = json::parse(raw_json, nullptr, /*allow_exceptions=*/true);
  } catch (const json::exception& exc) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse retrieval KB JSON at ", kb_path, ": ", exc.what()));
  }
  if (!payload.contains("documents") || !payload["documents"].is_array()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Retrieval KB is missing a documents array: ", kb_path));
  }

  std::vector<RetrievalDocument> documents;
  for (const json& item : payload["documents"]) {
    if (!item.contains("doc_id") || !item.contains("title") ||
        !item.contains("text") || !item.contains("embedding")) {
      return absl::InvalidArgumentError(
          "Retrieval KB document is missing required fields.");
    }
    if (!item["embedding"].is_array()) {
      return absl::InvalidArgumentError(
          "Retrieval KB document embedding must be an array.");
    }
    RetrievalDocument document;
    document.doc_id = item["doc_id"].get<std::string>();
    document.title = item["title"].get<std::string>();
    document.text = item["text"].get<std::string>();
    document.embedding = item["embedding"].get<std::vector<float>>();
    documents.push_back(std::move(document));
  }
  RETURN_IF_ERROR(NormalizeDocumentEmbeddingsInPlace(documents));
  return documents;
}

absl::StatusOr<std::vector<RetrievalHit>> RankDocumentsBySimilarity(
    absl::Span<const float> query_embedding,
    absl::Span<const RetrievalDocument> documents, int top_k) {
  if (documents.empty()) {
    return absl::InvalidArgumentError(
        "Cannot rank retrieval documents from an empty corpus.");
  }
  if (top_k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("top_k must be positive, got ", top_k, "."));
  }
  std::vector<float> normalized_query(query_embedding.begin(),
                                      query_embedding.end());
  RETURN_IF_ERROR(NormalizeEmbeddingInPlace(normalized_query));

  std::vector<RetrievalHit> hits;
  hits.reserve(documents.size());
  for (const RetrievalDocument& document : documents) {
    if (document.embedding.size() != normalized_query.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Embedding dimension mismatch for doc_id=", document.doc_id,
          ": expected ", normalized_query.size(), ", got ",
          document.embedding.size(), "."));
    }
    const double score =
        std::inner_product(normalized_query.begin(), normalized_query.end(),
                           document.embedding.begin(), 0.0);
    hits.push_back({
        .doc_id = document.doc_id,
        .title = document.title,
        .text = document.text,
        .score = score,
    });
  }

  std::sort(hits.begin(), hits.end(),
            [](const RetrievalHit& lhs, const RetrievalHit& rhs) {
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              return lhs.doc_id < rhs.doc_id;
            });
  if (static_cast<int>(hits.size()) > top_k) {
    hits.resize(top_k);
  }
  return hits;
}

absl::StatusOr<std::vector<Environment::Option>>
BuildRetrievalEnvironmentOptions(
    absl::string_view accelerator,
    const RetrievalEnvironmentConfig& environment_config) {
  std::string normalized(accelerator);
  absl::AsciiStrToLower(&normalized);

  std::vector<Environment::Option> env_options;
  if (!environment_config.runtime_library_dir.empty()) {
    env_options.push_back(Environment::Option{
        Environment::OptionTag::RuntimeLibraryDir,
        environment_config.runtime_library_dir});
  }

  if (normalized == "cpu") {
    return env_options;
  }
  if (normalized == "gpu") {
    return env_options;
  }
  if (normalized == "npu") {
    if (environment_config.runtime_library_dir.empty()) {
      return absl::InvalidArgumentError(
          "NPU retrieval requires a non-empty runtime_library_dir.");
    }
    if (environment_config.dispatch_library_dir.empty()) {
      return absl::InvalidArgumentError(
          "NPU retrieval requires a non-empty dispatch_library_dir.");
    }
    env_options.push_back(Environment::Option{
        Environment::OptionTag::DispatchLibraryDir,
        environment_config.dispatch_library_dir});
    env_options.push_back(Environment::Option{
        Environment::OptionTag::CompilerPluginLibraryDir,
        environment_config.dispatch_library_dir});
    return env_options;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported accelerator: ", accelerator));
}

absl::StatusOr<Options> BuildRetrievalCompilationOptions(
    absl::string_view accelerator) {
  ASSIGN_OR_RETURN(litert::HwAccelerators hw_accelerator,
                   ParseAccelerator(accelerator));
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());

  if (hw_accelerator == litert::HwAccelerators::kGpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    LITERT_RETURN_IF_ERROR(gpu_options.EnableConstantTensorSharing(true));
    LITERT_RETURN_IF_ERROR(
        gpu_options.SetPrecision(GpuOptions::Precision::kFp32));
    LITERT_RETURN_IF_ERROR(gpu_options.SetPreferTextureWeights(true));
  }

  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(hw_accelerator));
  return options;
}

absl::StatusOr<GraphPilotRetrievalEngine> GraphPilotRetrievalEngine::Create(
    absl::string_view model_path, absl::string_view tokenizer_path,
    absl::string_view accelerator,
    const RetrievalEnvironmentConfig& environment_config) {
  ASSIGN_OR_RETURN(auto env_options,
                   BuildRetrievalEnvironmentOptions(accelerator,
                                                   environment_config));
  LITERT_ASSIGN_OR_RETURN(auto environment, litert::Environment::Create(env_options));
  ASSIGN_OR_RETURN(auto options,
                   BuildRetrievalCompilationOptions(accelerator));
  LITERT_ASSIGN_OR_RETURN(auto compiled_model, litert::CompiledModel::Create(
                                            environment, std::string(model_path),
                                            options));
  ASSIGN_OR_RETURN(int sequence_length,
                   ResolveSequenceLength(compiled_model));
  ASSIGN_OR_RETURN(auto tokenizer,
                   SentencePieceTokenizer::CreateFromFile(tokenizer_path));

  return GraphPilotRetrievalEngine(
      std::make_unique<litert::Environment>(std::move(environment)),
      std::make_unique<litert::CompiledModel>(std::move(compiled_model)),
      std::move(tokenizer), sequence_length, std::string(accelerator));
}

absl::StatusOr<std::vector<float>> GraphPilotRetrievalEngine::EmbedText(
    absl::string_view text) {
  ASSIGN_OR_RETURN(std::vector<int> token_ids, tokenizer_->TextToTokenIds(text));
  ASSIGN_OR_RETURN(
      std::vector<int32_t> model_ids,
      BuildModelInputTokenIds(token_ids, sequence_length_,
                              tokenizer_->GetProcessor().bos_id(),
                              tokenizer_->GetProcessor().eos_id(),
                              tokenizer_->GetProcessor().pad_id()));
  LITERT_ASSIGN_OR_RETURN(auto input_buffers, compiled_model_->CreateInputBuffers(0));
  if (input_buffers.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected one retrieval input tensor, got ", input_buffers.size(),
        "."));
  }
  LITERT_RETURN_IF_ERROR(input_buffers[0].Write<int32_t>(model_ids));
  LITERT_ASSIGN_OR_RETURN(auto output_buffers, compiled_model_->CreateOutputBuffers(0));
  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected one retrieval output tensor, got ", output_buffers.size(),
        "."));
  }
  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(static_cast<size_t>(0), input_buffers, output_buffers));
  LITERT_ASSIGN_OR_RETURN(auto output_type, output_buffers[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto num_elements, output_type.Layout().NumElements());
  std::vector<float> embedding(num_elements);
  LITERT_RETURN_IF_ERROR(output_buffers[0].Read<float>(absl::MakeSpan(embedding)));
  RETURN_IF_ERROR(NormalizeEmbeddingInPlace(embedding));
  return embedding;
}

}  // namespace litert::lm
