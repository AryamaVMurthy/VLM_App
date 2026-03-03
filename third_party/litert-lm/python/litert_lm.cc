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

#include <deque>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"   // IWYU pragma: keep
#include "nanobind/stl/variant.h"      // IWYU pragma: keep
#include "nanobind/stl/vector.h"       // IWYU pragma: keep
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nanobind_json/nanobind_json.hpp"  // from @nanobind_json  // IWYU pragma: keep
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "tflite/core/c/c_api_types.h"  // from @litert
#include "tflite/logger.h"  // from @litert
#include "tflite/minimal_logging.h"  // from @litert

#define VALUE_OR_THROW(status_or)                                   \
  ([&]() {                                                          \
    auto status_or_value = (status_or);                             \
    if (!status_or_value.ok()) {                                    \
      std::stringstream ss;                                         \
      ss << __FILE__ << ":" << __LINE__ << ": " << __func__ << ": " \
         << status_or_value.status();                               \
      throw std::runtime_error(ss.str());                           \
    }                                                               \
    return std::move(status_or_value).value();                      \
  }())

namespace litert::lm {

namespace nb = nanobind;

// Note: Consider move to C++ API.
enum class LogSeverity {
  VERBOSE = 0,
  DEBUG = 1,
  INFO = 2,
  WARNING = 3,
  ERROR = 4,
  FATAL = 5,
  SILENT = 1000,
};

// MessageIterator bridges the asynchronous, callback-based C++ API
// (Conversation::SendMessageAsync) to Python's synchronous iterator protocol
// (__iter__ / __next__).
//
// It provides a thread-safe queue where the background C++ inference thread
// pushes generated message chunks. The Python main thread can then safely
// pull these chunks one by one by iterating over this object.
//
// This design keeps the C++ background thread completely free from Python's
// Global Interpreter Lock (GIL), maximizing concurrency and preventing
// deadlocks.
class MessageIterator {
 public:
  MessageIterator() = default;

  MessageIterator(const MessageIterator&) = delete;
  MessageIterator& operator=(const MessageIterator&) = delete;

  void Push(absl::StatusOr<Message> message) {
    absl::MutexLock lock(&mutex_);
    queue_.push_back(std::move(message));
  }

  nlohmann::json Next() {
    absl::StatusOr<Message> msg;
    {
      nb::gil_scoped_release release;
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &MessageIterator::HasData));
      msg = std::move(queue_.front());
      queue_.pop_front();
    }

    if (!msg.ok()) {
      if (absl::IsCancelled(msg.status())) {
        throw nb::stop_iteration();
      }
      throw std::runtime_error(msg.status().ToString());
    }

    if (!std::holds_alternative<JsonMessage>(*msg)) {
      throw std::runtime_error(
          "SendMessageAsync did not return a JsonMessage.");
    }

    auto& json_msg = std::get<JsonMessage>(*msg);
    if (json_msg.empty()) {
      throw nb::stop_iteration();
    }

    return static_cast<nlohmann::json>(json_msg);
  }

  bool HasData() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !queue_.empty();
  }

 private:
  absl::Mutex mutex_;
  std::deque<absl::StatusOr<Message>> queue_ ABSL_GUARDED_BY(mutex_);
};

struct PyBenchmarkInfo {
  double init_time_in_second;
  double time_to_first_token_in_second;
  int last_prefill_token_count;
  double last_prefill_tokens_per_second;
  int last_decode_token_count;
  double last_decode_tokens_per_second;
};

NB_MODULE(litert_lm_ext, m) {
  nb::enum_<LogSeverity>(m, "LogSeverity")
      .value("VERBOSE", LogSeverity::VERBOSE)
      .value("DEBUG", LogSeverity::DEBUG)
      .value("INFO", LogSeverity::INFO)
      .value("WARNING", LogSeverity::WARNING)
      .value("ERROR", LogSeverity::ERROR)
      .value("FATAL", LogSeverity::FATAL)
      .value("SILENT", LogSeverity::SILENT)
      .export_values();

  nb::enum_<Backend>(m, "Backend")
      .value("CPU", Backend::CPU)
      .value("GPU", Backend::GPU)
      .value("UNSPECIFIED", Backend::UNSPECIFIED)
      .export_values();

  nb::class_<ModelAssets>(m, "ModelAssets")
      .def_static(
          "create",
          [](absl::string_view model_path) {
            return VALUE_OR_THROW(ModelAssets::Create(model_path));
          },
          nb::arg("model_path"));

  nb::class_<EngineSettings>(m, "EngineSettings")
      .def_static(
          "create_default",
          [](const ModelAssets& model_assets, Backend backend,
             std::optional<Backend> vision_backend,
             std::optional<Backend> audio_backend) {
            return VALUE_OR_THROW(EngineSettings::CreateDefault(
                model_assets, backend, vision_backend, audio_backend));
          },
          nb::arg("model_assets"), nb::arg("backend") = Backend::CPU,
          nb::arg("vision_backend") = std::nullopt,
          nb::arg("audio_backend") = std::nullopt)
      .def(
          "set_cache_dir",
          [](EngineSettings& self, absl::string_view cache_dir) {
            self.GetMutableMainExecutorSettings().SetCacheDir(
                std::string(cache_dir));
          },
          nb::arg("cache_dir"))
      .def(
          "set_max_num_tokens",
          [](EngineSettings& self, int max_num_tokens) {
            self.GetMutableMainExecutorSettings().SetMaxNumTokens(
                max_num_tokens);
          },
          nb::arg("max_num_tokens"));

  m.def(
      "create_default_engine",
      [](const EngineSettings& engine_settings,
         absl::string_view input_prompt_as_hint) {
        return VALUE_OR_THROW(EngineFactory::CreateDefault(
            engine_settings, input_prompt_as_hint));
      },
      nb::arg("engine_settings"), nb::arg("input_prompt_as_hint") = "");

  m.def(
      "set_min_log_severity",
      [](LogSeverity log_severity) {
        absl::LogSeverityAtLeast absl_log_severity;
        LiteRtLogSeverity litert_log_severity;
        tflite::LogSeverity tflite_log_severity;

        switch (log_severity) {
          case LogSeverity::VERBOSE:
            absl_log_severity = absl::LogSeverityAtLeast::kInfo;
            litert_log_severity = kLiteRtLogSeverityVerbose;
            tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
            break;
          case LogSeverity::DEBUG:
            absl_log_severity = absl::LogSeverityAtLeast::kInfo;
            litert_log_severity = kLiteRtLogSeverityDebug;
            tflite_log_severity = tflite::TFLITE_LOG_VERBOSE;
            break;
          case LogSeverity::INFO:
            absl_log_severity = absl::LogSeverityAtLeast::kInfo;
            litert_log_severity = kLiteRtLogSeverityInfo;
            tflite_log_severity = tflite::TFLITE_LOG_INFO;
            break;
          case LogSeverity::WARNING:
            absl_log_severity = absl::LogSeverityAtLeast::kWarning;
            litert_log_severity = kLiteRtLogSeverityWarning;
            tflite_log_severity = tflite::TFLITE_LOG_WARNING;
            break;
          case LogSeverity::ERROR:
            absl_log_severity = absl::LogSeverityAtLeast::kError;
            litert_log_severity = kLiteRtLogSeverityError;
            tflite_log_severity = tflite::TFLITE_LOG_ERROR;
            break;
          case LogSeverity::FATAL:
            absl_log_severity = absl::LogSeverityAtLeast::kFatal;
            litert_log_severity = kLiteRtLogSeverityError;
            tflite_log_severity = tflite::TFLITE_LOG_ERROR;
            break;
          default:  // infinity
            absl_log_severity = absl::LogSeverityAtLeast::kInfinity;
            litert_log_severity = kLiteRtLogSeveritySilent;
            tflite_log_severity = tflite::TFLITE_LOG_SILENT;
            break;
        }

        absl::SetMinLogLevel(absl_log_severity);
        LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(),
                                   litert_log_severity);
        tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
            tflite_log_severity);
      },
      nb::arg("log_severity"));

  nb::class_<Engine>(m, "Engine")
      // Support for Python context managers (with statement).
      // __enter__ returns the object itself.
      .def("__enter__", [](nb::handle self) { return self; })
      // __exit__ immediately destroys the underlying C++ instance to free
      // resources deterministically, instead of waiting for garbage collection.
      .def(
          "__exit__",
          [](nb::handle self, nb::handle exc_type, nb::handle exc_value,
             nb::handle traceback) { nb::inst_destruct(self); },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none());

  nb::class_<ConversationConfig>(m, "ConversationConfig")
      .def_static(
          "create_default",
          [](const Engine& engine) {
            return VALUE_OR_THROW(ConversationConfig::CreateDefault(engine));
          },
          nb::arg("engine"));

  nb::class_<Conversation>(m, "Conversation")
      // Support for Python context managers (with statement).
      // __enter__ returns the object itself.
      .def("__enter__", [](nb::handle self) { return self; })
      // __exit__ immediately destroys the underlying C++ instance to free
      // resources deterministically, instead of waiting for garbage collection.
      .def(
          "__exit__",
          [](nb::handle self, nb::handle exc_type, nb::handle exc_value,
             nb::handle traceback) { nb::inst_destruct(self); },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def("cancel_process", &Conversation::CancelProcess)
      .def_static(
          "create",
          [](Engine& engine, const ConversationConfig& config) {
            return VALUE_OR_THROW(Conversation::Create(engine, config));
          },
          nb::arg("engine"), nb::arg("config"))
      .def(
          "send_message",
          [](Conversation& self, const nb::dict& message) {
            nlohmann::json json_message = message;
            absl::StatusOr<Message> result = self.SendMessage(json_message);
            Message message_variant = VALUE_OR_THROW(std::move(result));

            if (!std::holds_alternative<JsonMessage>(message_variant)) {
              throw std::runtime_error(
                  "SendMessage did not return a JsonMessage.");
            }

            return static_cast<nlohmann::json>(
                std::get<JsonMessage>(message_variant));
          },
          nb::arg("message"))
      .def(
          "send_message_async",
          [](Conversation& self, const nb::dict& message) {
            nlohmann::json json_message = message;
            auto iterator = std::make_shared<MessageIterator>();

            absl::Status status = self.SendMessageAsync(
                json_message, [iterator](absl::StatusOr<Message> msg) {
                  iterator->Push(std::move(msg));
                });

            if (!status.ok()) {
              std::stringstream ss;
              ss << "SendMessageAsync failed: " << status;
              throw std::runtime_error(ss.str());
            }
            return iterator;
          },
          nb::arg("message"));

  // Expose the MessageIterator to Python so that it can be used in a
  // standard `for chunk in stream:` loop. We bind Python's iterator protocol
  // (__iter__ and __next__) to our C++ implementation.
  nb::class_<MessageIterator>(m, "MessageIterator")
      .def("__iter__", [](nb::handle self) { return self; })
      .def("__next__", &MessageIterator::Next);

  m.def(
      "benchmark",
      [](absl::string_view model_path, Backend backend, int prefill_tokens,
         int decode_tokens, absl::string_view cache_dir) {
        auto model_assets = VALUE_OR_THROW(ModelAssets::Create(model_path));
        auto settings = VALUE_OR_THROW(
            EngineSettings::CreateDefault(model_assets, backend));

        if (!cache_dir.empty()) {
          settings.GetMutableMainExecutorSettings().SetCacheDir(
              std::string(cache_dir));
        }

        auto& benchmark_params = settings.GetMutableBenchmarkParams();
        benchmark_params.set_num_prefill_tokens(prefill_tokens);
        benchmark_params.set_num_decode_tokens(decode_tokens);

        auto engine =
            VALUE_OR_THROW(EngineFactory::CreateDefault(std::move(settings)));

        auto conversation_config =
            VALUE_OR_THROW(ConversationConfig::CreateDefault(*engine));
        auto conversation =
            VALUE_OR_THROW(Conversation::Create(*engine, conversation_config));

        // Trigger benchmark
        nlohmann::json dummy_message = {
            {"role", "user"},
            {"content", "Engine ignore this message in this mode."}};
        (void)VALUE_OR_THROW(conversation->SendMessage(dummy_message));

        auto benchmark_info = VALUE_OR_THROW(conversation->GetBenchmarkInfo());

        PyBenchmarkInfo result;

        double total_init_time_ms = 0.0;
        for (const auto& phase : benchmark_info.GetInitPhases()) {
          total_init_time_ms += absl::ToDoubleMilliseconds(phase.second);
        }
        result.init_time_in_second = total_init_time_ms / 1000.0;
        result.time_to_first_token_in_second =
            benchmark_info.GetTimeToFirstToken();

        int last_prefill_token_count = 0;
        double last_prefill_tokens_per_second = 0.0;
        if (benchmark_info.GetTotalPrefillTurns() > 0) {
          int last_index =
              static_cast<int>(benchmark_info.GetTotalPrefillTurns()) - 1;
          auto turn = benchmark_info.GetPrefillTurn(last_index);
          if (turn.ok()) {
            last_prefill_token_count = static_cast<int>(turn->num_tokens);
          }
          last_prefill_tokens_per_second =
              benchmark_info.GetPrefillTokensPerSec(last_index);
        }
        result.last_prefill_token_count = last_prefill_token_count;
        result.last_prefill_tokens_per_second = last_prefill_tokens_per_second;

        int last_decode_token_count = 0;
        double last_decode_tokens_per_second = 0.0;
        if (benchmark_info.GetTotalDecodeTurns() > 0) {
          int last_index =
              static_cast<int>(benchmark_info.GetTotalDecodeTurns()) - 1;
          auto turn = benchmark_info.GetDecodeTurn(last_index);
          if (turn.ok()) {
            last_decode_token_count = static_cast<int>(turn->num_tokens);
          }
          last_decode_tokens_per_second =
              benchmark_info.GetDecodeTokensPerSec(last_index);
        }
        result.last_decode_token_count = last_decode_token_count;
        result.last_decode_tokens_per_second = last_decode_tokens_per_second;

        return result;
      },
      nb::arg("model_path"), nb::arg("backend"),
      nb::arg("prefill_tokens") = 256, nb::arg("decode_tokens") = 256,
      nb::arg("cache_dir") = "");

  nb::class_<PyBenchmarkInfo>(m, "BenchmarkInfo",
                              "Data class to hold benchmark information.")
      .def_rw("init_time_in_second", &PyBenchmarkInfo::init_time_in_second,
              "The time in seconds to initialize the engine and the "
              "conversation.")
      .def_rw("time_to_first_token_in_second",
              &PyBenchmarkInfo::time_to_first_token_in_second,
              "The time in seconds to the first token.")
      .def_rw(
          "last_prefill_token_count",
          &PyBenchmarkInfo::last_prefill_token_count,
          "The number of tokens in the last prefill. Returns 0 if there was "
          "no prefill.")
      .def_rw("last_prefill_tokens_per_second",
              &PyBenchmarkInfo::last_prefill_tokens_per_second,
              "The number of tokens processed per second in the last prefill.")
      .def_rw("last_decode_token_count",
              &PyBenchmarkInfo::last_decode_token_count,
              "The number of tokens in the last decode. Returns 0 if there was "
              "no decode.")
      .def_rw("last_decode_tokens_per_second",
              &PyBenchmarkInfo::last_decode_tokens_per_second,
              "The number of tokens processed per second in the last decode.");
}

}  // namespace litert::lm
