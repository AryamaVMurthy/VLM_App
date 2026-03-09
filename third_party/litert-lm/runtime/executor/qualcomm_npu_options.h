#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QUALCOMM_NPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QUALCOMM_NPU_OPTIONS_H_

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"

namespace litert::lm {

::litert::Expected<void> ConfigureDefaultQualcommNpuLiteRtOptions(
    ::litert::Options& options);

::litert::Expected<::litert::Options> CreateDefaultQualcommNpuLiteRtOptions();

::litert::Expected<::litert::qualcomm::QualcommOptions>
CreateDefaultQualcommNpuOptions();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_QUALCOMM_NPU_OPTIONS_H_
