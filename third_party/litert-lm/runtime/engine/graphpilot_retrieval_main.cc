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

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, mode, "", "embed or retrieve");
ABSL_FLAG(std::string, model_path, "", "Path to retrieval TFLite model.");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to SentencePiece tokenizer.");
ABSL_FLAG(std::string, kb_path, "", "Path to retrieval KB JSON.");
ABSL_FLAG(std::string, query_path, "", "Path to UTF-8 query text.");
ABSL_FLAG(std::string, query_b64, "", "Base64-encoded UTF-8 query text.");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu, gpu, or npu");
ABSL_FLAG(std::string, runtime_library_dir, "",
          "Runtime library directory required for GPU/NPU retrieval.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Dispatch/compiler plugin directory required for NPU retrieval.");
ABSL_FLAG(int, top_k, 3, "Top-k documents to return.");

namespace litert::lm {
namespace {

using ::nlohmann::json;

absl::StatusOr<std::string> ReadTextFile(absl::string_view path) {
  std::ifstream input{std::string(path)};
  if (!input.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open text file: ", path));
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<std::string> LoadQueryText() {
  const std::string query_path = absl::GetFlag(FLAGS_query_path);
  const std::string query_b64 = absl::GetFlag(FLAGS_query_b64);
  if (!query_path.empty()) {
    return ReadTextFile(query_path);
  }
  if (!query_b64.empty()) {
    std::string decoded;
    if (!absl::Base64Unescape(query_b64, &decoded)) {
      return absl::InvalidArgumentError("Failed to base64-decode query_b64.");
    }
    return decoded;
  }
  return absl::InvalidArgumentError(
      "Provide either --query_path or --query_b64.");
}

absl::Status Run() {
  const std::string mode = absl::GetFlag(FLAGS_mode);
  if (mode != "embed" && mode != "retrieve") {
    return absl::InvalidArgumentError(
        "Unsupported --mode. Expected embed or retrieve.");
  }
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
  if (model_path.empty() || tokenizer_path.empty()) {
    return absl::InvalidArgumentError(
        "--model_path and --tokenizer_path are required.");
  }
  ASSIGN_OR_RETURN(std::string query, LoadQueryText());
  if (query.empty()) {
    return absl::InvalidArgumentError("Query text is empty.");
  }

  const absl::Time total_start = absl::Now();
  ASSIGN_OR_RETURN(
      auto engine,
      GraphPilotRetrievalEngine::Create(model_path, tokenizer_path,
                                        absl::GetFlag(FLAGS_accelerator),
                                        RetrievalEnvironmentConfig{
                                            .runtime_library_dir =
                                                absl::GetFlag(
                                                    FLAGS_runtime_library_dir),
                                            .dispatch_library_dir =
                                                absl::GetFlag(
                                                    FLAGS_dispatch_library_dir),
                                        }));
  const absl::Time embed_start = absl::Now();
  ASSIGN_OR_RETURN(std::vector<float> embedding, engine.EmbedText(query));
  const double embed_ms =
      absl::ToDoubleMilliseconds(absl::Now() - embed_start);

  json payload = {
      {"mode", mode},
      {"accelerator", std::string(engine.accelerator())},
      {"sequence_length", engine.sequence_length()},
      {"query", query},
      {"timings_ms",
       {{"embed", embed_ms},
        {"total", absl::ToDoubleMilliseconds(absl::Now() - total_start)}}},
      {"embedding", embedding},
  };

  if (mode == "retrieve") {
    const std::string kb_path = absl::GetFlag(FLAGS_kb_path);
    if (kb_path.empty()) {
      return absl::InvalidArgumentError("--kb_path is required for retrieve mode.");
    }
    const absl::Time search_start = absl::Now();
    ASSIGN_OR_RETURN(auto documents,
                     LoadRetrievalDocumentsFromJsonFile(kb_path));
    ASSIGN_OR_RETURN(auto hits,
                     RankDocumentsBySimilarity(embedding, documents,
                                              absl::GetFlag(FLAGS_top_k)));
    json hits_json = json::array();
    for (const RetrievalHit& hit : hits) {
      hits_json.push_back({
          {"doc_id", hit.doc_id},
          {"title", hit.title},
          {"text", hit.text},
          {"score", hit.score},
      });
    }
    payload["hits"] = std::move(hits_json);
    payload["timings_ms"]["search"] =
        absl::ToDoubleMilliseconds(absl::Now() - search_start);
    payload["timings_ms"]["total"] =
        absl::ToDoubleMilliseconds(absl::Now() - total_start);
  }

  std::cout << payload.dump() << std::endl;
  return absl::OkStatus();
}

}  // namespace
}  // namespace litert::lm

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = litert::lm::Run();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << status;
    std::cerr << status.message() << std::endl;
    return 1;
  }
  return 0;
}
