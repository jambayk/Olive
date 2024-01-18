// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include <cuda.h>

#include "core/providers/cuda/cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

// TODO(jambayk): This will be templated based on the triton kernels generated
#include "matmul_fp32.h"


using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace OliveTritonFusion {

void TritonMatMul(const Ort::Custom::CudaContext& cuda_ctx,
                  const Ort::Custom::Tensor<float>& X,
                  const Ort::Custom::Tensor<float>& Y,
                // TODO(jambayk): Add support for arguments based on fused epilogues/activations
                  Ort::Custom::Tensor<float>& Z) {
    CUSTOM_ENFORCE(cuda_ctx.cuda_stream, "failed to fetch cuda stream");

    // TODO(jambayk): create helper for shape and stride computations
    // get shape of A: M1 X ... X Mn X K.
    // Ex. batch_size X seq_len X K
    auto x_shape = X.Shape();
    // stack all dimensions except the last one
    int64_t M = std::accumulate(x_shape.begin(), x_shape.end() - 1, 1, std::multiplies<int64_t>());
    // last dimension
    int64_t K = x_shape.back();

    // currently, we will only support 2D tensors for B
    // shape of B: K X N
    auto y_shape = Y.Shape();
    CUSTOM_ENFORCE(y_shape.size() == 2, "B must be a 2D tensor");
    CUSTOM_ENFORCE(y_shape[0] == K, "B must have the same last dimension as A");
    int64_t N = y_shape[1];

    // allocate output tensor
    auto z_raw = Z.Allocate({M, N});

    // call the kernel
    CUresult ret = matmul_fp32_default(cuda_ctx.cuda_stream,
                        reinterpret_cast<CUdeviceptr>(X.DataRaw()),
                        reinterpret_cast<CUdeviceptr>(Y.DataRaw()),
                        reinterpret_cast<CUdeviceptr>(z_raw),
                        M, N, K,
                        K, 1,
                        N, 1,
                        N, 1);
    CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "matmul_fp32_default failed");
}

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("TritonMatMul", "CUDAExecutionProvider", TritonMatMul)};
  domain.Add(c_CustomOpOne.get());
}

} // namespace OliveTritonFusion