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

#include <filesystem>  // NOLINT
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/litert_lm_settings.h"
#include "runtime/engine/litert_lm_settings_util.h"
#include "runtime/engine/overlap_scheduler.h"
#include "runtime/engine/request_preparation_queue.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/magic_number_configs_helper.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, model_path, "", "Model path to use for overlap runs.");
ABSL_FLAG(std::string, decode_model_path, "",
          "Optional CPU decode model path. Use the raw FastVLM .litertlm when "
          "the prefill model path points at a Qualcomm precompiled bundle.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");
ABSL_FLAG(std::vector<std::string>, image_paths, {},
          "Device-local image paths for queued requests.");
ABSL_FLAG(std::string, request_manifest_path, "",
          "Optional JSONL manifest describing queued requests. Each line must "
          "contain request_id, prompt, and image_path.");
ABSL_FLAG(int, max_num_tokens, 512, "Max context length for overlap run.");
ABSL_FLAG(int, max_output_tokens, -1, "Max generated output tokens.");
ABSL_FLAG(int, max_visual_tokens, 0,
          "Maximum number of projected visual tokens to retain.");
ABSL_FLAG(std::string, visual_token_pruning_strategy, "uniform",
          "Visual token pruning strategy.");
ABSL_FLAG(std::string, constraint_regex, "",
          "Optional regex constraint applied during CPU decode.");
ABSL_FLAG(int, num_cpu_threads, 4, "Number of CPU threads for decode.");
ABSL_FLAG(int, prepare_queue_size, 4,
          "Number of prepared requests to keep buffered ahead of prefill.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory containing LiteRT Qualcomm dispatch/compiler libraries.");

namespace litert::lm {
namespace {

using ::nlohmann::json;

bool IsEventModeEnabled() {
  const char* value = std::getenv("LITERT_LM_EVENT_MODE");
  if (value == nullptr) {
    return false;
  }
  const std::string normalized(value);
  return normalized == "1" || normalized == "true" || normalized == "TRUE" ||
         normalized == "yes" || normalized == "YES";
}

absl::Mutex& EventLogMutex() {
  static auto* mutex = new absl::Mutex();
  return *mutex;
}

void EmitEventLine(const json& event) {
  absl::MutexLock lock(&EventLogMutex());
  std::cout << "VLM_EVENT " << event.dump() << std::endl;
  std::cout.flush();
}

std::string GetInputPrompt() {
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!input_prompt.empty() && !input_prompt_file.empty()) {
    ABSL_LOG(FATAL) << "Only one of --input_prompt and --input_prompt_file can "
                       "be specified.";
  }
  if (!input_prompt.empty()) {
    return input_prompt;
  }
  if (!input_prompt_file.empty()) {
    std::ifstream file(input_prompt_file);
    if (!file.is_open()) {
      ABSL_LOG(FATAL) << "Could not open prompt file: " << input_prompt_file;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
  return "Describe this image in one sentence.";
}

using QueuedRequest = RequestPreparationSpec;

absl::StatusOr<std::vector<QueuedRequest>> LoadQueuedRequests() {
  const std::string request_manifest_path =
      absl::GetFlag(FLAGS_request_manifest_path);
  const std::vector<std::string> image_paths = absl::GetFlag(FLAGS_image_paths);
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!request_manifest_path.empty()) {
    if (!image_paths.empty() || !input_prompt.empty() ||
        !input_prompt_file.empty()) {
      return absl::InvalidArgumentError(
          "--request_manifest_path cannot be combined with --image_paths, "
          "--input_prompt, or --input_prompt_file.");
    }
    std::ifstream file(request_manifest_path);
    if (!file.is_open()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not open request manifest: ", request_manifest_path));
    }
    std::vector<QueuedRequest> requests;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
      ++line_number;
      if (line.empty()) {
        continue;
      }
      json payload;
      try {
        payload = json::parse(line);
      } catch (const std::exception& e) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid JSON in request manifest ", request_manifest_path,
            " at line ", line_number, ": ", e.what()));
      }
      for (absl::string_view field : {"request_id", "prompt", "image_path"}) {
        if (!payload.contains(std::string(field))) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Missing required field '", field, "' in request manifest ",
              request_manifest_path, " at line ", line_number));
        }
      }
      requests.push_back(QueuedRequest{
          .request_index = static_cast<int>(requests.size()),
          .request_id = payload["request_id"].get<std::string>(),
          .prompt = payload["prompt"].get<std::string>(),
          .image_path = payload["image_path"].get<std::string>(),
      });
    }
    if (requests.size() < 2) {
      return absl::InvalidArgumentError(
          "Request manifest must contain at least 2 queued requests.");
    }
    return requests;
  }

  if (image_paths.size() < 2) {
    return absl::InvalidArgumentError(
        "--image_paths must contain at least 2 queued images.");
  }
  const std::string prompt = GetInputPrompt();
  std::vector<QueuedRequest> requests;
  requests.reserve(image_paths.size());
  for (int i = 0; i < image_paths.size(); ++i) {
    requests.push_back(QueuedRequest{
        .request_index = i,
        .request_id = absl::StrCat("request_", i),
        .prompt = prompt,
        .image_path = image_paths[i],
    });
  }
  return requests;
}

absl::StatusOr<std::string> ReadBinaryFile(absl::string_view path) {
  std::ifstream file(std::string(path), std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Could not open image file: ", path));
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  if (!file.good() && !file.eof()) {
    return absl::InternalError(
        absl::StrCat("Failed to read image file: ", path));
  }
  return buffer.str();
}

absl::StatusOr<ImagePreprocessParameter> CreateImagePreprocessParameter(
    const proto::LlmModelType& llm_model_type) {
  int image_tensor_height = 0;
  int image_tensor_width = 0;
  if (llm_model_type.has_gemma3n()) {
    image_tensor_height = llm_model_type.gemma3n().image_tensor_height();
    image_tensor_width = llm_model_type.gemma3n().image_tensor_width();
  } else if (llm_model_type.has_gemma3()) {
    image_tensor_height = llm_model_type.gemma3().image_tensor_height();
    image_tensor_width = llm_model_type.gemma3().image_tensor_width();
  } else {
    return absl::FailedPreconditionError(
        "Overlap runner only supports Gemma3/Gemma3N image preprocessing.");
  }
  if (image_tensor_height <= 0 || image_tensor_width <= 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("Invalid image tensor dimensions from model metadata: ",
                     image_tensor_height, "x", image_tensor_width));
  }
  ImagePreprocessParameter image_parameter;
  image_parameter.SetTargetDimensions(
      Dimensions({1, image_tensor_height, image_tensor_width, 3}));
  return image_parameter;
}

absl::StatusOr<ModelAssets> CreateModelAssets(absl::string_view model_path) {
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  return ModelAssets::Create(model_path);
}

absl::StatusOr<EngineSettings> CreateOverlapEngineSettings(
    const LiteRtLmSettings& settings, Backend backend,
    std::optional<Backend> vision_backend) {
  ASSIGN_OR_RETURN(ModelAssets model_assets, CreateModelAssets(settings.model_path));
  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   EngineSettings::CreateDefault(std::move(model_assets), backend,
                                                 vision_backend,
                                                 /*audio_backend=*/std::nullopt));
  if (settings.max_num_tokens > 0) {
    engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
        settings.max_num_tokens);
  }
  if (backend == Backend::CPU) {
    // Match the NPU prefill path's activation precision as closely as possible
    // so KV cache handoff can remain a real backend interop path.
    engine_settings.GetMutableMainExecutorSettings().SetActivationDataType(
        ActivationDataType::FLOAT16);
  }
  if (!settings.litert_dispatch_lib_dir.empty()) {
    engine_settings.GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
        settings.litert_dispatch_lib_dir);
  }
  if (backend == Backend::CPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(auto cpu_settings,
                     executor_settings.MutableBackendConfig<CpuConfig>());
    if (settings.num_cpu_threads > 0) {
      cpu_settings.number_of_threads = settings.num_cpu_threads;
    }
    executor_settings.SetBackendConfig(cpu_settings);
  }
  AdvancedSettings advanced_settings{
      .configure_magic_numbers = true,
      .clear_kv_cache_before_prefill = true,
  };
  engine_settings.GetMutableMainExecutorSettings().SetAdvancedSettings(
      advanced_settings);
  return engine_settings;
}

absl::StatusOr<Environment> CreateEnvironmentForSettings(
    EngineSettings& engine_settings, ModelResources& model_resources) {
  std::vector<Environment::Option> env_options;
  const auto& main_executor_settings = engine_settings.GetMainExecutorSettings();
  if (main_executor_settings.GetBackend() == Backend::CPU ||
      main_executor_settings.GetBackend() == Backend::GPU) {
    if (!main_executor_settings.GetAdvancedSettings() ||
        main_executor_settings.GetAdvancedSettings()->configure_magic_numbers) {
      MagicNumberConfigsHelper helper;
      env_options =
          helper.GetLiteRtEnvOptions(model_resources, main_executor_settings);
    }
  } else {
#if defined(LITERT_DISABLE_NPU)
    return absl::InvalidArgumentError(
        "NPU backend is disabled in this build.");
#else
    auto configure_npu_library_dirs =
        [&](absl::string_view library_dir,
            absl::string_view source_description) {
          env_options.push_back(::litert::Environment::Option{
              ::litert::Environment::OptionTag::DispatchLibraryDir,
              library_dir});
          env_options.push_back(::litert::Environment::Option{
              ::litert::Environment::OptionTag::CompilerPluginLibraryDir,
              library_dir});
          ABSL_LOG(INFO) << "Setting dispatch library path from "
                         << source_description << ": " << library_dir;
          ABSL_LOG(INFO) << "Setting compiler plugin path from "
                         << source_description << ": " << library_dir;
        };
    if (!main_executor_settings.GetLitertDispatchLibDir().empty()) {
      configure_npu_library_dirs(
          main_executor_settings.GetLitertDispatchLibDir(),
          "main_executor_settings");
    } else {
      ASSIGN_OR_RETURN(auto model_path,
                       main_executor_settings.GetModelAssets().GetPath());
      const std::filesystem::path path(model_path);
      const std::string dispatch_root = path.parent_path().string();
      if (!dispatch_root.empty()) {
        configure_npu_library_dirs(dispatch_root, "model_path parent");
      }
    }
#endif
  }
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create(env_options));
  return env;
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupManager>>
CreatePromptEmbeddingLookupManager(ModelResources& model_resources,
                                   const SessionConfig& session_config) {
  if (session_config.GetMaxVisualTokens() <= 0) {
    return std::unique_ptr<EmbeddingLookupManager>(nullptr);
  }
  if (!absl::EqualsIgnoreCase(session_config.GetVisualTokenPruningStrategy(),
                              "prompt_conditioned_v1")) {
    return std::unique_ptr<EmbeddingLookupManager>(nullptr);
  }
  ASSIGN_OR_RETURN(const litert::Model * text_embedder_model,
                   model_resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  return EmbeddingLookupManager::Create(
      text_embedder_model, /*fully_supports_multi_modal=*/false);
}

struct SessionBundle {
  EngineSettings engine_settings;
  std::unique_ptr<ModelResources> model_resources;
  Environment environment;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<LlmExecutor> executor;
  std::unique_ptr<VisionExecutor> vision_executor;
  std::unique_ptr<ThreadPool> worker_thread_pool;
  std::unique_ptr<SessionBasic> session;
};

absl::StatusOr<SessionBundle> CreatePrefillBundle(
    const LiteRtLmSettings& settings, absl::string_view input_prompt_hint) {
  ASSIGN_OR_RETURN(
      auto engine_settings,
      CreateOverlapEngineSettings(settings, Backend::NPU, Backend::NPU));
  ASSIGN_OR_RETURN(auto model_resources,
                   BuildLiteRtCompiledModelResources(
                       engine_settings.GetMutableMainExecutorSettings()
                           .GetModelAssets()));
  ASSIGN_OR_RETURN(auto tokenizer, model_resources->GetTokenizer());
  ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
      *tokenizer, llm_metadata, input_prompt_hint,
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLitePrefillDecode),
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLiteVisionEncoder),
      std::nullopt));
  ASSIGN_OR_RETURN(auto environment,
                   CreateEnvironmentForSettings(engine_settings,
                                                *model_resources));
  ASSIGN_OR_RETURN(
      auto executor,
      CreateLlmLiteRtCompiledModelExecutor(
          engine_settings.GetMainExecutorSettings(), environment,
          *model_resources));
  ASSIGN_OR_RETURN(
      auto vision_executor,
      VisionLiteRtCompiledModelExecutor::Create(
          engine_settings.GetMutableVisionExecutorSettings().value(),
          environment));
  LiteRtLmSettings session_settings = settings;
  session_settings.backend = "npu";
  session_settings.vision_backend = "npu";
  SessionConfig session_config = CreateSessionConfig(session_settings);
  if (settings.max_output_tokens > 0) {
    session_config.SetMaxOutputTokens(settings.max_output_tokens);
  }
  RETURN_IF_ERROR(session_config.MaybeUpdateAndValidate(engine_settings));
  ASSIGN_OR_RETURN(auto prompt_embedding_lookup_manager,
                   CreatePromptEmbeddingLookupManager(*model_resources,
                                                      session_config));
  auto worker_thread_pool =
      std::make_unique<ThreadPool>("overlap_prefill", /*max_num_threads=*/1);
  ASSIGN_OR_RETURN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer.get(),
                           vision_executor.get(), /*audio_executor=*/nullptr,
                           std::move(prompt_embedding_lookup_manager),
                           session_config, std::nullopt,
                           worker_thread_pool.get()));
  return SessionBundle{
      .engine_settings = std::move(engine_settings),
      .model_resources = std::move(model_resources),
      .environment = std::move(environment),
      .tokenizer = std::move(tokenizer),
      .executor = std::move(executor),
      .vision_executor = std::move(vision_executor),
      .worker_thread_pool = std::move(worker_thread_pool),
      .session = std::move(session),
  };
}

absl::StatusOr<SessionBundle> CreateDecodeBundle(
    const LiteRtLmSettings& settings, absl::string_view input_prompt_hint) {
  ASSIGN_OR_RETURN(
      auto engine_settings,
      CreateOverlapEngineSettings(settings, Backend::CPU, std::nullopt));
  ASSIGN_OR_RETURN(auto model_resources,
                   BuildLiteRtCompiledModelResources(
                       engine_settings.GetMutableMainExecutorSettings()
                           .GetModelAssets()));
  ASSIGN_OR_RETURN(auto tokenizer, model_resources->GetTokenizer());
  ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
      *tokenizer, llm_metadata, input_prompt_hint,
      model_resources->GetTFLiteModelBackendConstraint(
          ModelType::kTfLitePrefillDecode),
      std::nullopt, std::nullopt));
  ASSIGN_OR_RETURN(auto environment,
                   CreateEnvironmentForSettings(engine_settings,
                                                *model_resources));
  ASSIGN_OR_RETURN(
      auto executor,
      CreateLlmLiteRtCompiledModelExecutor(
          engine_settings.GetMainExecutorSettings(), environment,
          *model_resources));
  LiteRtLmSettings session_settings = settings;
  session_settings.backend = "cpu";
  session_settings.vision_backend = std::nullopt;
  SessionConfig session_config = CreateSessionConfig(session_settings);
  if (settings.max_output_tokens > 0) {
    session_config.SetMaxOutputTokens(settings.max_output_tokens);
  }
  RETURN_IF_ERROR(session_config.MaybeUpdateAndValidate(engine_settings));
  auto worker_thread_pool =
      std::make_unique<ThreadPool>("overlap_decode", /*max_num_threads=*/1);
  ASSIGN_OR_RETURN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool.get()));
  return SessionBundle{
      .engine_settings = std::move(engine_settings),
      .model_resources = std::move(model_resources),
      .environment = std::move(environment),
      .tokenizer = std::move(tokenizer),
      .executor = std::move(executor),
      .vision_executor = nullptr,
      .worker_thread_pool = std::move(worker_thread_pool),
      .session = std::move(session),
  };
}

absl::StatusOr<std::vector<InputData>> BuildRequestContents(
    absl::string_view prompt, absl::string_view image_path,
    ImagePreprocessor& image_preprocessor,
    const ImagePreprocessParameter& image_parameter) {
  ASSIGN_OR_RETURN(std::string image_bytes, ReadBinaryFile(image_path));
  ASSIGN_OR_RETURN(
      InputImage processed_image,
      image_preprocessor.Preprocess(InputImage(std::move(image_bytes)),
                                    image_parameter));
  std::vector<InputData> contents;
  contents.emplace_back(InputText(std::string(prompt)));
  contents.emplace_back(std::move(processed_image));
  return contents;
}

absl::StatusOr<PreparedRequest> PrepareRequest(
    const QueuedRequest& request, ImagePreprocessor& image_preprocessor,
    const ImagePreprocessParameter& image_parameter) {
  PreparedRequest prepared_request;
  prepared_request.request = request;
  prepared_request.prepare_start_time = absl::Now();
  if (IsEventModeEnabled()) {
    EmitEventLine({{"type", "PREPARE_START"},
                   {"request_index", request.request_index},
                   {"request_id", request.request_id},
                   {"timestamp_unix_nanos",
                    absl::ToUnixNanos(prepared_request.prepare_start_time)}});
  }
  ASSIGN_OR_RETURN(prepared_request.request_contents,
                   BuildRequestContents(request.prompt, request.image_path,
                                        image_preprocessor, image_parameter));
  prepared_request.prepare_end_time = absl::Now();
  if (IsEventModeEnabled()) {
    EmitEventLine({{"type", "PREPARE_DONE"},
                   {"request_index", request.request_index},
                   {"request_id", request.request_id},
                   {"duration_ms",
                    absl::ToDoubleMilliseconds(
                        prepared_request.prepare_end_time -
                        prepared_request.prepare_start_time)},
                   {"timestamp_unix_nanos",
                    absl::ToUnixNanos(prepared_request.prepare_end_time)}});
  }
  return prepared_request;
}

struct DecodeState {
  int request_index = -1;
  StageWindow decode_window;
  std::optional<std::string> text;
  std::optional<absl::Status> status;
};

struct PendingDecode {
  std::shared_ptr<DecodeState> state;
  std::future<absl::StatusOr<Responses>> future;
  std::thread worker;

  bool Valid() const { return state != nullptr; }
};

absl::StatusOr<std::unique_ptr<Constraint>> CreateRegexConstraint(
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    absl::string_view constraint_regex) {
  ASSIGN_OR_RETURN(
      auto constraint_provider,
      CreateConstraintProvider(ConstraintProviderConfig(LlGuidanceConfig()),
                               tokenizer, stop_token_ids));
  return constraint_provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kRegex,
                              .constraint_string =
                                  std::string(constraint_regex)});
}

PendingDecode StartDecode(SessionBasic& decode_session,
                          const PrefillDecodeHandoff& handoff,
                          const Tokenizer& tokenizer,
                          absl::string_view constraint_regex,
                          int request_index) {
  ABSL_CHECK_OK(decode_session.ResetForReuse());
  ABSL_CHECK_OK(decode_session.ImportPrefillDecodeHandoff(handoff));
  auto promise = std::make_shared<std::promise<absl::StatusOr<Responses>>>();
  PendingDecode pending;
  pending.state = std::make_shared<DecodeState>();
  pending.future = promise->get_future();
  pending.state->request_index = request_index;
  pending.worker = std::thread([&decode_session, promise,
                                &tokenizer,
                                constraint_regex = std::string(constraint_regex),
                                state = pending.state]() mutable {
    state->decode_window.start_time = absl::Now();
    try {
      DecodeConfig decode_config = DecodeConfig::CreateDefault();
      std::unique_ptr<Constraint> constraint;
      if (!constraint_regex.empty()) {
        auto constraint_or = CreateRegexConstraint(
            tokenizer, decode_session.GetSessionConfig().GetStopTokenIds(),
            constraint_regex);
        if (!constraint_or.ok()) {
          state->decode_window.end_time = absl::Now();
          state->status = constraint_or.status();
          promise->set_value(constraint_or.status());
          return;
        }
        constraint = std::move(*constraint_or);
        decode_config.SetConstraint(constraint.get());
      }
      auto responses = decode_session.RunDecode(decode_config);
      state->decode_window.end_time = absl::Now();
      if (responses.ok() && !responses->GetTexts().empty()) {
        state->text = responses->GetTexts()[0];
      } else if (!responses.ok()) {
        state->status = responses.status();
      }
      promise->set_value(std::move(responses));
    } catch (const std::exception& e) {
      state->decode_window.end_time = absl::Now();
      const absl::Status status = absl::InternalError(
          absl::StrCat("Decode worker threw std::exception: ", e.what()));
      ABSL_LOG(ERROR) << status;
      state->status = status;
      promise->set_value(status);
    } catch (...) {
      state->decode_window.end_time = absl::Now();
      const absl::Status status = absl::InternalError(
          "Decode worker threw an unknown non-absl exception.");
      ABSL_LOG(ERROR) << status;
      state->status = status;
      promise->set_value(status);
    }
  });
  return pending;
}

absl::Status FinalizeDecode(PendingDecode& pending) {
  ABSL_CHECK(pending.Valid());
  absl::StatusOr<Responses> responses;
  try {
    responses = pending.future.get();
  } catch (const std::exception& e) {
    if (pending.worker.joinable()) {
      pending.worker.join();
    }
    const absl::Status status = absl::InternalError(
        absl::StrCat("Decode future get failed: ", e.what()));
    pending.state->status = status;
    return status;
  } catch (...) {
    if (pending.worker.joinable()) {
      pending.worker.join();
    }
    const absl::Status status =
        absl::InternalError("Decode future get failed with unknown exception.");
    pending.state->status = status;
    return status;
  }
  if (pending.worker.joinable()) {
    pending.worker.join();
  }
  if (!responses.ok()) {
    return responses.status();
  }
  if (!pending.state->status.has_value()) {
    pending.state->status = absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string decode_model_path = absl::GetFlag(FLAGS_decode_model_path);
  const std::string constraint_regex = absl::GetFlag(FLAGS_constraint_regex);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("--model_path must be provided.");
  }
  ASSIGN_OR_RETURN(const std::vector<QueuedRequest> requests,
                   LoadQueuedRequests());
  const std::string input_prompt_hint = requests.front().prompt;

  LiteRtLmSettings common_settings;
  common_settings.max_num_tokens = absl::GetFlag(FLAGS_max_num_tokens);
  common_settings.max_output_tokens = absl::GetFlag(FLAGS_max_output_tokens);
  common_settings.max_num_images = 1;
  common_settings.max_visual_tokens = absl::GetFlag(FLAGS_max_visual_tokens);
  common_settings.visual_token_pruning_strategy =
      absl::GetFlag(FLAGS_visual_token_pruning_strategy);
  common_settings.num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  common_settings.litert_dispatch_lib_dir =
      absl::GetFlag(FLAGS_litert_dispatch_lib_dir);
  LiteRtLmSettings prefill_settings = common_settings;
  prefill_settings.model_path = model_path;
  LiteRtLmSettings decode_settings = common_settings;
  decode_settings.model_path =
      decode_model_path.empty() ? model_path : decode_model_path;
  ASSIGN_OR_RETURN(auto prefill_bundle,
                   CreatePrefillBundle(prefill_settings, input_prompt_hint));
  ASSIGN_OR_RETURN(auto decode_bundle,
                   CreateDecodeBundle(decode_settings, input_prompt_hint));
  ASSIGN_OR_RETURN(auto* prefill_llm_metadata,
                   prefill_bundle.model_resources->GetLlmMetadata());
  ASSIGN_OR_RETURN(
      const ImagePreprocessParameter image_preprocess_parameter,
      CreateImagePreprocessParameter(prefill_llm_metadata->llm_model_type()));
  StbImagePreprocessor image_preprocessor;
  ASSIGN_OR_RETURN(
      auto prepared_request_queue,
      RequestPreparationQueue::Create(
          requests, absl::GetFlag(FLAGS_prepare_queue_size),
          [&](const QueuedRequest& request) -> absl::StatusOr<PreparedRequest> {
            return PrepareRequest(request, image_preprocessor,
                                  image_preprocess_parameter);
          }));

  ABSL_LOG(INFO) << "Overlap runtime configured with real FastVLM sessions: "
                 << "prefill_backend=npu decode_backend=cpu";
  ABSL_LOG(INFO) << "Overlap model paths: prefill=" << prefill_settings.model_path
                 << " decode=" << decode_settings.model_path;
  if (IsEventModeEnabled()) {
    EmitEventLine({{"type", "OVERLAP_CONFIG"},
                   {"prefill_backend", "npu"},
                   {"decode_backend", "cpu"},
                   {"prefill_model_path", prefill_settings.model_path},
                   {"decode_model_path", decode_settings.model_path},
                   {"queue_size", requests.size()},
                   {"prepare_queue_size", absl::GetFlag(FLAGS_prepare_queue_size)},
                   {"max_visual_tokens", common_settings.max_visual_tokens},
                   {"visual_token_pruning_strategy",
                    common_settings.visual_token_pruning_strategy}});
  }

  std::optional<PendingDecode> pending_decode = std::nullopt;
  std::vector<std::string> final_texts(requests.size());
  std::vector<StageWindow> prefill_windows(requests.size());

  for (int i = 0; i < requests.size(); ++i) {
    const QueuedRequest& request = requests[i];
    const int queue_depth = static_cast<int>(requests.size()) - i;
    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "QUEUE_ADMISSION"},
                     {"request_index", i},
                     {"request_id", request.request_id},
                     {"queue_depth", queue_depth},
                     {"image_path", request.image_path}});
    }

    RETURN_IF_ERROR(prefill_bundle.session->ResetForReuse());
    const absl::Time prepare_wait_start = absl::Now();
    ASSIGN_OR_RETURN(PreparedRequest prepared_request,
                     prepared_request_queue->PopNext());
    const absl::Duration prepare_queue_wait =
        absl::Now() - prepare_wait_start;
    if (prepared_request.request.request_index != i) {
      return absl::InternalError(absl::StrCat(
          "Prepared request order mismatch: expected index ", i,
          " but received ", prepared_request.request.request_index, "."));
    }
    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "PREPARE_QUEUE_WAIT"},
                     {"request_index", i},
                     {"request_id", request.request_id},
                     {"duration_ms",
                      absl::ToDoubleMilliseconds(prepare_queue_wait)}});
      EmitEventLine(
          {{"type", "PREPARE_QUEUE_STATE"},
           {"request_index", i},
           {"request_id", request.request_id},
           {"queue_depth", prepared_request_queue->queue_size()}});
    }
    prefill_windows[i].start_time = absl::Now();
    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "PREFILL_START"},
                     {"request_index", i},
                     {"request_id", request.request_id},
                     {"timestamp_unix_nanos",
                      absl::ToUnixNanos(prefill_windows[i].start_time)}});
    }
    RETURN_IF_ERROR(
        prefill_bundle.session->RunPrefill(prepared_request.request_contents));
    prefill_windows[i].end_time = absl::Now();
    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "PREFILL_DONE"},
                     {"request_index", i},
                     {"request_id", request.request_id},
                     {"duration_ms", absl::ToDoubleMilliseconds(
                                         prefill_windows[i].end_time -
                                         prefill_windows[i].start_time)},
                     {"timestamp_unix_nanos",
                      absl::ToUnixNanos(prefill_windows[i].end_time)}});
    }
    ASSIGN_OR_RETURN(PrefillDecodeHandoff handoff,
                     prefill_bundle.session->ExportPrefillDecodeHandoff());

    if (pending_decode.has_value()) {
      const bool decode_ready =
          pending_decode->future.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready;
      if (!decode_ready && IsEventModeEnabled()) {
        EmitEventLine({{"type", "STALL"},
                       {"request_index", i},
                       {"request_id", request.request_id},
                       {"stall_reason", "decode_worker_busy_after_prefill"}});
      }
      RETURN_IF_ERROR(FinalizeDecode(*pending_decode));
      const OverlapAnalysis analysis = AnalyzeOverlapWindow(
          pending_decode->state->decode_window, prefill_windows[i]);
      if (IsEventModeEnabled()) {
        EmitEventLine(
            {{"type", "OVERLAP_ANALYSIS"},
             {"previous_request_index", pending_decode->state->request_index},
             {"previous_request_id",
              requests[pending_decode->state->request_index].request_id},
             {"next_request_index", i},
             {"next_request_id", request.request_id},
             {"overlap_ms",
              absl::ToDoubleMilliseconds(analysis.overlap_duration)},
             {"cpu_stall_before_next_decode_ms",
              absl::ToDoubleMilliseconds(
                  analysis.cpu_stall_before_next_decode)},
             {"no_overlap_reason",
              analysis.no_overlap_reason.value_or(std::string())}});
      }
      final_texts[pending_decode->state->request_index] =
          pending_decode->state->text.value_or("");
      pending_decode.reset();
    }

    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "DECODE_START"},
                     {"request_index", i},
                     {"request_id", request.request_id},
                     {"backend", "cpu"}});
    }
    pending_decode = StartDecode(*decode_bundle.session, handoff,
                                 *decode_bundle.tokenizer, constraint_regex, i);
  }

  if (!pending_decode.has_value()) {
    return absl::InternalError("No decode request was launched.");
  }
  RETURN_IF_ERROR(FinalizeDecode(*pending_decode));
  if (IsEventModeEnabled()) {
    EmitEventLine(
        {{"type", "DECODE_DONE"},
         {"request_index", pending_decode->state->request_index},
         {"request_id",
          requests[pending_decode->state->request_index].request_id},
         {"duration_ms",
          absl::ToDoubleMilliseconds(
              pending_decode->state->decode_window.end_time -
              pending_decode->state->decode_window.start_time)},
         {"timestamp_unix_nanos",
          absl::ToUnixNanos(pending_decode->state->decode_window.end_time)}});
  }
  final_texts[pending_decode->state->request_index] =
      pending_decode->state->text.value_or("");

  for (int i = 0; i < final_texts.size(); ++i) {
    ABSL_LOG(INFO) << "Overlap response[" << i << "]: " << final_texts[i];
    std::cout << "OVERLAP_RESPONSE[" << i << "] " << final_texts[i]
              << std::endl;
    if (IsEventModeEnabled()) {
      EmitEventLine({{"type", "OVERLAP_RESPONSE"},
                     {"request_index", i},
                     {"request_id", requests[i].request_id},
                     {"text", final_texts[i]}});
    }
  }
  if (IsEventModeEnabled()) {
    EmitEventLine({{"type", "DONE"}});
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace litert::lm

int main(int argc, char** argv) {
  ABSL_CHECK_OK(litert::lm::MainHelper(argc, argv));
  return 0;
}
