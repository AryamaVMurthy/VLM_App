// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tflite/types/half.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

TEST(EmbeddingLookupOpBuilderTest,
     QuantInt8TableToFloat32OutputMaterializesFp16TableAndCast) {
  TensorPool tensor_pool;

  const std::vector<float> scales = {0.1f, 0.2f, 0.3f, 0.4f};
  const std::vector<std::int32_t> zero_points = {0, 0, 0, 0};
  const AxisScaleOffsetQuantizeParamsWrapper table_quant_params(
      /*axis=*/0, scales, zero_points);

  const std::vector<std::int8_t> table_data = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
  };
  auto& table_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, table_quant_params, {4, 3},
      sizeof(std::int8_t) * table_data.size(), table_data.data());

  const std::vector<std::int32_t> indices_data = {0, 2};
  auto& indices_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, UndefinedQuantizeParamsWrapper{}, {2},
      sizeof(std::int32_t) * indices_data.size(), indices_data.data());

  auto& output_tensor = tensor_pool.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, UndefinedQuantizeParamsWrapper{}, {2, 3},
      "embedding_lookup");

  auto ops = BuildEmbeddingLookupOp(tensor_pool, {indices_tensor, table_tensor},
                                    {output_tensor});

  ASSERT_EQ(ops.size(), 2);
  EXPECT_TRUE(ops[0].IsOpCode(QnnOpCode::kGather));
  EXPECT_TRUE(ops[1].IsOpCode(QnnOpCode::kCast));

  const auto& fp16_table = ops[0].GetInputTensor(0);
  EXPECT_TRUE(fp16_table.IsTensorStatic());
  EXPECT_TRUE(fp16_table.IsF16());
  EXPECT_FALSE(fp16_table.IsQuant());
  const auto* fp16_table_data = static_cast<const std::uint16_t*>(
      fp16_table.GetQnnTensor().v2.clientBuf.data);
  ASSERT_NE(fp16_table_data, nullptr);
  const std::vector<float> expected_dequantized = {
      0.1f, 0.2f, 0.3f,
      0.8f, 1.0f, 1.2f,
      2.1f, 2.4f, 2.7f,
      4.0f, 4.4f, 4.8f,
  };
  ASSERT_EQ(fp16_table.GetQnnTensor().v2.clientBuf.dataSize,
            sizeof(std::uint16_t) * expected_dequantized.size());
  for (size_t i = 0; i < expected_dequantized.size(); ++i) {
    EXPECT_EQ(fp16_table_data[i], tflite::half(expected_dequantized[i]).to_bits());
  }

  const auto& fp16_output = ops[0].GetOutputTensor(0);
  EXPECT_TRUE(fp16_output.IsF16());
  EXPECT_FALSE(fp16_output.IsQuant());
  EXPECT_EQ(fp16_output.GetDimensions(), std::vector<std::uint32_t>({2, 3}));

  EXPECT_EQ(&ops[1].GetInputTensor(0), &fp16_output);
  EXPECT_EQ(&ops[1].GetOutputTensor(0), &output_tensor);
}

}  // namespace
}  // namespace qnn
