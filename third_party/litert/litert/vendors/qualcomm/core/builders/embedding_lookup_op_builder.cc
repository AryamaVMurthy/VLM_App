// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "tflite/types/half.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {
constexpr int kTableIdx = 1;
constexpr int kIndicesIdx = 0;
constexpr int kOutputIdx = 0;

size_t NumElementsForAxisPrefix(const std::vector<std::uint32_t>& dimensions,
                                const std::int32_t axis) {
  return std::accumulate(dimensions.begin(), dimensions.begin() + axis, 1ULL,
                         std::multiplies<size_t>());
}

size_t NumElementsForAxisSuffix(const std::vector<std::uint32_t>& dimensions,
                                const std::int32_t axis) {
  return std::accumulate(dimensions.begin() + axis + 1, dimensions.end(), 1ULL,
                         std::multiplies<size_t>());
}

std::optional<std::vector<std::uint16_t>> DequantizeInt8TableToFp16Bits(
    const TensorWrapper& table_tensor) {
  if (!table_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR(
        "Embedding lookup int8->float32 lowering requires a static table.");
    return std::nullopt;
  }

  auto int8_data = table_tensor.GetTensorData<std::int8_t>();
  if (!int8_data.has_value()) {
    QNN_LOG_ERROR("Embedding lookup get int8 table failed.");
    return std::nullopt;
  }

  std::vector<std::uint16_t> fp16_table_bits(int8_data->size());
  const auto encode_value = [&fp16_table_bits, &int8_data](const size_t index,
                                                           const float scale,
                                                           const std::int32_t zero_point) {
    const float dequantized =
        (static_cast<std::int32_t>((*int8_data)[index]) - zero_point) * scale;
    fp16_table_bits[index] = tflite::half(dequantized).to_bits();
  };

  if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
          table_tensor.GetQuantParams())) {
    const auto& quant_params =
        std::get<ScaleOffsetQuantizeParamsWrapper>(table_tensor.GetQuantParams());
    for (size_t i = 0; i < fp16_table_bits.size(); ++i) {
      encode_value(i, quant_params.GetScale(), quant_params.GetZeroPoint());
    }
    return fp16_table_bits;
  }

  if (!std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          table_tensor.GetQuantParams())) {
    QNN_LOG_ERROR(
        "Embedding lookup int8->float32 lowering requires scale-offset or "
        "axis scale-offset quantization.");
    return std::nullopt;
  }

  const auto& quant_params =
      std::get<AxisScaleOffsetQuantizeParamsWrapper>(table_tensor.GetQuantParams());
  const auto axis = quant_params.GetAxis();
  const auto& dimensions = table_tensor.GetDimensions();
  if (axis < 0 || axis >= static_cast<std::int32_t>(dimensions.size())) {
    QNN_LOG_ERROR("Embedding lookup quantized axis is out of bounds.");
    return std::nullopt;
  }

  std::vector<float> scales;
  std::vector<std::int32_t> zero_points;
  quant_params.GetScales(scales);
  quant_params.GetZeroPoints(zero_points);
  if (scales.size() != dimensions[axis] || zero_points.size() != scales.size()) {
    QNN_LOG_ERROR(
        "Embedding lookup per-axis quantization metadata does not match table "
        "dimensions.");
    return std::nullopt;
  }

  const size_t axis_stride = NumElementsForAxisSuffix(dimensions, axis);
  const size_t num_axis_blocks = NumElementsForAxisPrefix(dimensions, axis);
  for (size_t block = 0; block < num_axis_blocks; ++block) {
    for (size_t axis_index = 0; axis_index < scales.size(); ++axis_index) {
      for (size_t offset = 0; offset < axis_stride; ++offset) {
        const size_t flat_index =
            (block * scales.size() + axis_index) * axis_stride + offset;
        encode_value(flat_index, scales[axis_index], zero_points[axis_index]);
      }
    }
  }

  return fp16_table_bits;
}

TensorWrapper* CreateFp16TableTensor(TensorPool& tensor_pool,
                                     const TensorWrapper& table_tensor) {
  auto fp16_table_bits = DequantizeInt8TableToFp16Bits(table_tensor);
  if (!fp16_table_bits.has_value()) {
    return nullptr;
  }

  return &tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_16, UndefinedQuantizeParamsWrapper{},
      table_tensor.GetDimensions(),
      sizeof(std::uint16_t) * fp16_table_bits->size(), fp16_table_bits->data());
}
}  // namespace

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& table_tensor = inputs[kTableIdx];
  TensorWrapper& indices_tensor = inputs[kIndicesIdx];
  TensorWrapper& output_tensor = outputs[kOutputIdx];

  auto& gather_op = CreateOpWrapper(res, QNN_OP_GATHER);
  // Case: QInt8 table with QInt16 output
  if (table_tensor.IsQuantI8() && output_tensor.IsQuantI16()) {
    QNN_LOG_WARNING(
        "The data type of embedding lookup table is int8, but output data type "
        "is int16. Int8 table will be cast to int16.");
    std::vector<std::int16_t> int16_data;
    size_t data_len = table_tensor.GetTensorNumElements();
    auto int8_data = table_tensor.GetTensorData<std::int8_t>();
    if (!int8_data.has_value()) {
      QNN_LOG_ERROR("Embedding lookup get int8 table failed.");
      return res;
    }
    int16_data.reserve(data_len);
    for (int i = 0; i < data_len; ++i) {
      int16_data.emplace_back(static_cast<std::int16_t>((*int8_data)[i]));
    }

    TensorWrapper& int16_table_tensor = tensor_pool.CreateStaticTensor(
        output_tensor.GetDataType(), table_tensor.GetQuantParams(),
        table_tensor.GetDimensions(),
        sizeof(decltype(int16_data)::value_type) * int16_data.size(),
        reinterpret_cast<void*>(int16_data.data()));

    gather_op.AddInputTensor(int16_table_tensor);
  } else if (table_tensor.IsQuantI8() && output_tensor.IsF32()) {
    // Materialize the embedding table to fp16 so Gather runs in a datatype
    // combination QNN HTP accepts, then cast the gathered output back to f32.
    TensorWrapper* fp16_table_tensor =
        CreateFp16TableTensor(tensor_pool, table_tensor);
    if (fp16_table_tensor == nullptr) {
      return res;
    }
    auto& fp16_output = tensor_pool.CreateNativeTensor(
        QNN_DATATYPE_FLOAT_16, UndefinedQuantizeParamsWrapper{},
        output_tensor.GetDimensions());
    gather_op.AddInputTensor(*fp16_table_tensor);
    gather_op.AddInputTensor(indices_tensor);
    gather_op.AddOutputTensor(fp16_output);
    gather_op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, 0);

    auto dequantize_ops =
        BuildDequantizeOp(tensor_pool, {fp16_output}, {output_tensor});
    std::move(dequantize_ops.begin(), dequantize_ops.end(),
              std::back_inserter(res));
    return res;
  } else {
    gather_op.AddInputTensor(table_tensor);
  }

  gather_op.AddInputTensor(indices_tensor);
  gather_op.AddOutputTensor(output_tensor);
  gather_op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, 0);
  return res;
}

}  // namespace qnn
