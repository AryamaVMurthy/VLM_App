#include "runtime/executor/qualcomm_npu_options.h"

#include "gtest/gtest.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"

namespace litert::lm {
namespace {

TEST(QualcommNpuOptionsTest, CreatesLiteRtOptionsForNpuExecution) {
  auto options = CreateDefaultQualcommNpuLiteRtOptions();
  ASSERT_TRUE(options.HasValue()) << options.Error();

  auto accelerators = options->GetHardwareAccelerators();
  ASSERT_TRUE(accelerators.HasValue()) << accelerators.Error();
  EXPECT_EQ(*accelerators,
            static_cast<LiteRtHwAcceleratorSet>(
                ::litert::HwAccelerators::kNpu));

  auto qualcomm_options = options->GetQualcommOptions();
  ASSERT_TRUE(qualcomm_options.HasValue()) << qualcomm_options.Error();
  EXPECT_EQ(qualcomm_options->GetLogLevel(),
            ::litert::qualcomm::QualcommOptions::LogLevel::kInfo);
  EXPECT_EQ(qualcomm_options->GetHtpPerformanceMode(),
            ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
}

TEST(QualcommNpuOptionsTest, UsesBurstHtpPerformanceMode) {
  auto options = CreateDefaultQualcommNpuOptions();
  ASSERT_TRUE(options.HasValue()) << options.Error();
  EXPECT_EQ(options->GetHtpPerformanceMode(),
            ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
}

TEST(QualcommNpuOptionsTest, ConfiguresExistingLiteRtOptionsForNpuExecution) {
  auto options = ::litert::Options::Create();
  ASSERT_TRUE(options.HasValue()) << options.Error();

  auto status = ConfigureDefaultQualcommNpuLiteRtOptions(*options);
  ASSERT_TRUE(status.HasValue()) << status.Error();

  auto accelerators = options->GetHardwareAccelerators();
  ASSERT_TRUE(accelerators.HasValue()) << accelerators.Error();
  EXPECT_EQ(*accelerators,
            static_cast<LiteRtHwAcceleratorSet>(
                ::litert::HwAccelerators::kNpu));
}

}  // namespace
}  // namespace litert::lm
