#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: matmul_fp32_128x256x64x8_warps8xstages3
CUresult matmul_fp32_3fdb966f_01234567891011(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);

CUresult matmul_fp32_128x256x64x8_warps8xstages3(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn){
if (1)
    return matmul_fp32_3fdb966f_01234567891011(stream, a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: matmul_fp32_128x256x64x8_warps8xstages3
void load_matmul_fp32_3fdb966f_01234567891011();
void load_matmul_fp32_128x256x64x8_warps8xstages3() {
  load_matmul_fp32_3fdb966f_01234567891011();
}

// unload for: matmul_fp32_128x256x64x8_warps8xstages3
void unload_matmul_fp32_3fdb966f_01234567891011();
void unload_matmul_fp32_128x256x64x8_warps8xstages3() {
  unload_matmul_fp32_3fdb966f_01234567891011();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);
kernel_func_t matmul_fp32_kernels[] = {
  matmul_fp32_128x256x64x8_warps8xstages3,
};

int matmul_fp32_get_num_algos(void){
  return (int)sizeof(matmul_fp32_kernels);
}

CUresult matmul_fp32(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn, int algo_id){
  assert (algo_id < (int)sizeof(matmul_fp32_kernels));
  return matmul_fp32_kernels[algo_id](stream, a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
}

void load_matmul_fp32(void){
  load_matmul_fp32_128x256x64x8_warps8xstages3();
}

void unload_matmul_fp32(void){
  unload_matmul_fp32_128x256x64x8_warps8xstages3();
}


CUresult matmul_fp32_default(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn){
  return matmul_fp32(stream, a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, 0);
}
