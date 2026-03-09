#include "runtime/executor/qualcomm_npu_options.h"

#include "litert/cc/litert_macros.h"

namespace litert::lm {
namespace {

void ApplyDefaultQualcommNpuSettings(
    ::litert::qualcomm::QualcommOptions& qualcomm_options) {
  qualcomm_options.SetLogLevel(
      ::litert::qualcomm::QualcommOptions::LogLevel::kInfo);
  qualcomm_options.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
}

}  // namespace

::litert::Expected<void> ConfigureDefaultQualcommNpuLiteRtOptions(
    ::litert::Options& options) {
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(::litert::HwAccelerators::kNpu));
  LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options, options.GetQualcommOptions());
  ApplyDefaultQualcommNpuSettings(qualcomm_options);
  return {};
}

::litert::Expected<::litert::Options> CreateDefaultQualcommNpuLiteRtOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  LITERT_RETURN_IF_ERROR(ConfigureDefaultQualcommNpuLiteRtOptions(options));
  return options;
}

::litert::Expected<::litert::qualcomm::QualcommOptions>
CreateDefaultQualcommNpuOptions() {
  LITERT_ASSIGN_OR_RETURN(auto qualcomm_options,
                          ::litert::qualcomm::QualcommOptions::Create());
  ApplyDefaultQualcommNpuSettings(qualcomm_options);
  return qualcomm_options;
}

}  // namespace litert::lm
