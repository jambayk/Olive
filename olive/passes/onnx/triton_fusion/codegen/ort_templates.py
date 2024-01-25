# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

CUSTOM_OP_SKELETON = """
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include <cuda.h>
#include <iostream>

#include "core/providers/cuda/cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

// Include custom kernels headers
{custom_kernel_includes}


using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

void ValidateElementwiseShapes(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape) {
  // currently, we only support limited one-directional broadcasting
  // 1s can only be prepended to the second input
  // second input withou leading 1s must be a suffix of the first input

  // check that the shapes are compatible
  CUSTOM_ENFORCE(b_shape.size() <= a_shape.size(), "Second input cannot have more dimensions than the first input");

  // check that the trailing dimensions match
  bool leading_ones = true;
  bool mismatch = false;
  for (size_t i = 0; i < b_shape.size(); ++i) {
    int64_t a_shape_i = a_shape[a_shape.size() - b_shape.size() + i];
    int64_t b_shape_i = b_shape[i];
    // skip leading ones in the second input
    if (leading_ones && b_shape_i == 1) continue;
    // once we see a non-one dimension, we are no longer skipping 1s in the second input
    leading_ones = false;
    // check that the dimensions match
    if (a_shape_i != b_shape_i) {
      mismatch = true;
      break;
    }
  }
  CUSTOM_ENFORCE(!mismatch, "Input shapes are not compatible");
}

namespace OliveTritonFusion {

// Define Custom Ops
{custom_op_defs}


// Register Custom Ops
void RegisterOps(Ort::CustomOpDomain& domain) {
  {custom_op_registrations}
}

} // namespace OliveTritonFusion
"""

CUSTOM_KERNEL_INCLUDE = '#include "{kernel_name}.h"'

CUSTOM_OP_REGISTRATION = """
  // Register {custom_op_name}
  static const std::unique_ptr<OrtLiteCustomOp> c_{custom_op_name}{
    Ort::Custom::CreateLiteCustomOp("{custom_op_name}", "CUDAExecutionProvider", {custom_op_name})
  };
  domain.Add(c_CustomOpOne.get());
"""

MATMUL_TEMPLATE = """
void {custom_op_name}(
    const Ort::Custom::CudaContext& cuda_ctx,
    // matmul inputs
    const Ort::Custom::Tensor<{dtype}>& A,
    const Ort::Custom::Tensor<{dtype}>& B,
    // fused Op inputs
    {fused_input_params}
    // fused Op attributes
    {fused_attr_params}
    // output
    Ort::Custom::Tensor<float>& Y) {

  // get shape of A: M1 X ... X Mn X K.
  // Ex. batch_size X seq_len X K
  auto a_shape = A.Shape();
  // stack all dimensions except the last one
  int64_t M = std::accumulate(a_shape.begin(), a_shape.end() - 1, 1, std::multiplies<int64_t>());
  // last dimension
  int64_t K = a_shape.back();

  // currently, we will only support 2D tensors for B
  // shape of B: K X N
  auto b_shape = B.Shape();
  CUSTOM_ENFORCE(b_shape.size() == 2, "B must be a 2D tensor");
  CUSTOM_ENFORCE(b_shape[0] == K, "First dimension of B must be equal to last dimension of A");
  int64_t N = b_shape[1];

  // shape of output: M1 X ... X Mn X N
  std::vector<int64_t> y_shape(a_shape.size());
  std::copy(a_shape.begin(), a_shape.end() - 1, y_shape.begin());
  y_shape.back() = N;

  // validate shapes of fused inputs
  {fused_input_shape_validation}

  // allocate output tensor
  auto y_raw = Y.Allocate(y_shape);

  // call the kernel
  load_{kernel_name}();
  CUresult ret = {kernel_name}(
      cuda_ctx.cuda_stream,
      reinterpret_cast<CUdeviceptr>(X.DataRaw()),
      reinterpret_cast<CUdeviceptr>(Y.DataRaw()),
      {fused_input_args}
      reinterpret_cast<CUdeviceptr>(y_raw),
      M, N, K,
      {fused_numel_args}
      {fused_attr_args}
      0);
  CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "{kernel_name}_default failed");
  unload_{kernel_name}();
}
"""

ELEMENTWISE_TEMPLATE = """
void {custom_op_name}(
    const Ort::Custom::CudaContext& cuda_ctx,
    // base operation inputs
    const Ort::Custom::Tensor<{dtype}>& A,
    {base_input_param}
    // fused operations inputs
    {fused_input_params}
    // base operation attributes
    {base_attr_params}
    // fused operations attributes
    {fused_attr_params}
    // output
    Ort::Custom::Tensor<{dtype}>& Y) {

  // output shape is the same as input shape
  // true because we only support limited one-directional broadcasting
  auto y_shape = A.Shape();

  // validate shape of base operation's second input
  {base_input_shape_validation}

  // validate shapes of fused inputs
  {fused_input_shape_validation}

  // allocate output tensor
  auto y_raw = Y.Allocate(y_shape);

  // call the kernel
  load_{kernel_name}();
  CUresult ret = {kernel_name}_default(
      cuda_ctx.cuda_stream,
      reinterpret_cast<CUdeviceptr>(A.DataRaw()),
      {base_input_args}
      {fused_input_args}
      reinterpret_cast<CUdeviceptr>(y_raw),
      A.NumberOfElement(),
      {base_numel_arg}
      {fused_numel_args}
      {base_attr_args}
      {fused_attr_args}
      0);
  CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "{kernel_name}_default failed");
  unload_{kernel_name}();
}
"""

INPUT_PARAM = "const Ort::Custom::Tensor<{dtype}>& {input_name}"

ATTR_PARAM = "{attr_dtype} {attr_name}"

INPUT_SHAPE_VALIDATION = "ValidateElementwiseShapes(y_shape, {input_name}.Shape());"

INPUT_ARG = "reinterpret_cast<CUdeviceptr>({input_name}.DataRaw())"

NUMEL_ARG = "{input_name}.NumberOfElement()"
