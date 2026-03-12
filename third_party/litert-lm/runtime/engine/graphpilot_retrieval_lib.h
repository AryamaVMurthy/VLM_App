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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_GRAPHPILOT_RETRIEVAL_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_GRAPHPILOT_RETRIEVAL_LIB_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"

namespace litert::lm {

struct RetrievalDocument {
  std::string doc_id;
  std::string title;
  std::string text;
  std::vector<float> embedding;
};

struct RetrievalHit {
  std::string doc_id;
  std::string title;
  std::string text;
  double score;
};

struct RetrievalEnvironmentConfig {
  std::string runtime_library_dir;
  std::string dispatch_library_dir;
};

absl::StatusOr<std::vector<int32_t>> BuildModelInputTokenIds(
    absl::Span<const int> token_ids, int sequence_length, int bos_id, int eos_id,
    int pad_id);

absl::Status NormalizeEmbeddingInPlace(std::vector<float>& embedding);
absl::Status NormalizeDocumentEmbeddingsInPlace(
    std::vector<RetrievalDocument>& documents);

absl::StatusOr<std::vector<RetrievalDocument>> LoadRetrievalDocumentsFromJsonFile(
    absl::string_view kb_path);

absl::StatusOr<std::vector<RetrievalHit>> RankDocumentsBySimilarity(
    absl::Span<const float> query_embedding,
    absl::Span<const RetrievalDocument> documents, int top_k);

absl::StatusOr<std::vector<Environment::Option>>
BuildRetrievalEnvironmentOptions(
    absl::string_view accelerator,
    const RetrievalEnvironmentConfig& environment_config);

absl::StatusOr<Options> BuildRetrievalCompilationOptions(
    absl::string_view accelerator);

class GraphPilotRetrievalEngine {
 public:
  static absl::StatusOr<GraphPilotRetrievalEngine> Create(
      absl::string_view model_path, absl::string_view tokenizer_path,
      absl::string_view accelerator,
      const RetrievalEnvironmentConfig& environment_config = {});

  GraphPilotRetrievalEngine(GraphPilotRetrievalEngine&&) = default;
  GraphPilotRetrievalEngine& operator=(GraphPilotRetrievalEngine&&) = default;

  GraphPilotRetrievalEngine(const GraphPilotRetrievalEngine&) = delete;
  GraphPilotRetrievalEngine& operator=(const GraphPilotRetrievalEngine&) = delete;

  absl::StatusOr<std::vector<float>> EmbedText(absl::string_view text);

  int sequence_length() const { return sequence_length_; }
  absl::string_view accelerator() const { return accelerator_; }

 private:
  GraphPilotRetrievalEngine(
      std::unique_ptr<litert::Environment> env,
      std::unique_ptr<litert::CompiledModel> compiled_model,
      std::unique_ptr<SentencePieceTokenizer> tokenizer, int sequence_length,
      std::string accelerator)
      : env_(std::move(env)),
        compiled_model_(std::move(compiled_model)),
        tokenizer_(std::move(tokenizer)),
        sequence_length_(sequence_length),
        accelerator_(std::move(accelerator)) {}

  std::unique_ptr<litert::Environment> env_;
  std::unique_ptr<litert::CompiledModel> compiled_model_;
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
  int sequence_length_;
  std::string accelerator_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_GRAPHPILOT_RETRIEVAL_LIB_H_
